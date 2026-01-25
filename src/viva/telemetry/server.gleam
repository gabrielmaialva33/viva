//// Telemetry Server
////
//// Real-time HTTP/JSON server for World visualization.
//// Uses polling (REST API) for reliability across mist versions.
////
//// Endpoints:
////   /           - Embedded frontend (Three.js + D3.js)
////   /api/health - Health check
////   /api/world  - Current world state JSON
////   /api/metrics - Aggregated metrics JSON
////   /api/graph  - D3.js force graph JSON

import gleam/bool
import gleam/bytes_tree
import gleam/dict
import gleam/erlang/process.{type Subject}
import gleam/float
import gleam/http/request.{type Request}
import gleam/http/response.{type Response}
import gleam/int
import gleam/json
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import gleam/set.{type Set}
import mist.{type Connection, type ResponseData}
import viva/memory/world.{type World}
import viva/telemetry/metrics
import viva/telemetry/perf
import viva/telemetry/system
import viva/telemetry/world_json

// =============================================================================
// TYPES
// =============================================================================

pub type TelemetryMessage {
  Subscribe(Subject(String))
  Unsubscribe(Subject(String))
  Broadcast(String)
  UpdateWorld(World)
  GetWorld(Subject(Option(World)))
}

pub type TelemetryState {
  TelemetryState(
    subscribers: Set(Subject(String)),
    current_world: Option(World),
  )
}

pub type Broadcaster =
  Subject(TelemetryMessage)

// =============================================================================
// BROADCASTER ACTOR
// =============================================================================

fn telemetry_loop(
  state: TelemetryState,
  message: TelemetryMessage,
) -> actor.Next(TelemetryState, TelemetryMessage) {
  case message {
    Subscribe(client) -> {
      actor.continue(TelemetryState(
        ..state,
        subscribers: set.insert(state.subscribers, client),
      ))
    }

    Unsubscribe(client) -> {
      actor.continue(TelemetryState(
        ..state,
        subscribers: set.delete(state.subscribers, client),
      ))
    }

    Broadcast(data) -> {
      let _ = set.fold(state.subscribers, Nil, fn(_, client) {
        process.send(client, data)
        Nil
      })
      actor.continue(state)
    }

    UpdateWorld(world) -> {
      let data = world_json.world_to_string(world)
      let _ = set.fold(state.subscribers, Nil, fn(_, client) {
        process.send(client, data)
        Nil
      })
      actor.continue(TelemetryState(..state, current_world: Some(world)))
    }

    GetWorld(reply_to) -> {
      process.send(reply_to, state.current_world)
      actor.continue(state)
    }
  }
}

pub fn start_broadcaster() -> Result(Broadcaster, actor.StartError) {
  let initial_state = TelemetryState(
    subscribers: set.new(),
    current_world: None,
  )

  let builder =
    actor.new(initial_state)
    |> actor.on_message(telemetry_loop)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Broadcast world state update
pub fn broadcast_world(broadcaster: Broadcaster, world: World) {
  process.send(broadcaster, UpdateWorld(world))
}

/// Legacy broadcast function
pub fn broadcast(broadcaster: Broadcaster, data: String) -> Nil {
  process.send(broadcaster, Broadcast(data))
}

// =============================================================================
// HTTP HANDLERS
// =============================================================================

fn json_response(body: String) -> Response(ResponseData) {
  response.new(200)
  |> response.set_header("content-type", "application/json")
  |> response.set_header("access-control-allow-origin", "*")
  |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
}

fn html_response(body: String) -> Response(ResponseData) {
  response.new(200)
  |> response.set_header("content-type", "text/html; charset=utf-8")
  |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
}

fn not_found() -> Response(ResponseData) {
  response.new(404)
  |> response.set_body(mist.Bytes(bytes_tree.from_string("Not Found")))
}

fn csv_response(body: String) -> Response(ResponseData) {
  response.new(200)
  |> response.set_header("content-type", "text/csv; charset=utf-8")
  |> response.set_header("access-control-allow-origin", "*")
  |> response.set_header("content-disposition", "attachment; filename=\"viva_export.csv\"")
  |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
}

/// Convert World state to CSV format for R/Python analysis
fn world_to_csv(w: World) -> String {
  let header = "id,label,x,y,z,w,energy,sleeping,island_id\n"
  let rows = dict.fold(w.bodies, "", fn(acc, id, body) {
    let pos = body.position.data
    let x = float.to_string(list_get_float(pos, 0))
    let y = float.to_string(list_get_float(pos, 1))
    let z = float.to_string(list_get_float(pos, 2))
    let w_coord = float.to_string(list_get_float(pos, 3))
    let energy = float.to_string(body.energy)
    let sleeping = bool.to_string(body.sleeping)
    let island = int.to_string(body.island_id)

    acc <> int.to_string(id) <> ","
        <> body.label <> ","
        <> x <> "," <> y <> "," <> z <> "," <> w_coord <> ","
        <> energy <> "," <> sleeping <> "," <> island <> "\n"
  })
  header <> rows
}

fn list_get_float(l: List(Float), idx: Int) -> Float {
  case list.drop(l, idx) {
    [x, ..] -> x
    [] -> 0.0
  }
}

fn get_world_sync(broadcaster: Broadcaster) -> Option(World) {
  let reply_subject = process.new_subject()
  process.send(broadcaster, GetWorld(reply_subject))
  case process.receive(reply_subject, 1000) {
    Ok(world_opt) -> world_opt
    Error(_) -> None
  }
}

// =============================================================================
// MAIN SERVER
// =============================================================================

pub fn start(port: Int, broadcaster: Broadcaster) {
  let handler = fn(req: Request(Connection)) {
    let path = request.path_segments(req)

    case path {
      // Health check
      ["api", "health"] ->
        json_response("{\"status\":\"ok\",\"service\":\"viva-telemetry\"}")

      // World state
      ["api", "world"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> json_response(world_json.world_to_string(world))
          None -> json_response("{\"error\":\"no_world_state\",\"bodies\":[],\"islands\":[],\"tick\":0}")
        }
      }

      // Metrics
      ["api", "metrics"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> {
            let m = metrics.collect(world)
            json_response(metrics.metrics_to_string(m))
          }
          None -> json_response("{\"error\":\"no_world_state\"}")
        }
      }

      // Force graph for D3.js
      ["api", "graph"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> {
            let graph = metrics.to_force_graph(world, 0.3)
            json_response(json.to_string(graph))
          }
          None -> json_response("{\"nodes\":[],\"links\":[]}")
        }
      }

      // CSV export for R/Python analysis
      ["api", "export", "csv"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> csv_response(world_to_csv(world))
          None -> csv_response("id,label,x,y,z,w,energy,sleeping,island_id\n")
        }
      }

      // System metrics (CPU, Memory, GPU)
      ["api", "system"] -> {
        let sys = system.collect()
        json_response(system.to_string(sys))
      }

      // Performance metrics (tick time, GC, memory, reductions)
      ["api", "perf"] -> {
        let p = perf.collect(0, 0.0)
        json_response(perf.to_string(p))
      }

      // Embedded frontend
      [] -> html_response(frontend_html(port))

      _ -> not_found()
    }
  }

  mist.new(handler)
  |> mist.port(port)
  |> mist.start
}

// =============================================================================
// EMBEDDED FRONTEND - Oblivion/Her Style (Minimal, Full-screen 3D)
// =============================================================================

fn frontend_html(port: Int) -> String {
  let port_str = int.to_string(port)
  "<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>VIVA</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link href=\"https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap\" rel=\"stylesheet\">
  <script src=\"https://unpkg.com/three@0.160.0/build/three.min.js\"></script>
  <script src=\"https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js\"></script>
  <script src=\"https://unpkg.com/three@0.160.0/examples/js/postprocessing/EffectComposer.js\"></script>
  <script src=\"https://unpkg.com/three@0.160.0/examples/js/postprocessing/RenderPass.js\"></script>
  <script src=\"https://unpkg.com/three@0.160.0/examples/js/postprocessing/UnrealBloomPass.js\"></script>
  <script src=\"https://unpkg.com/three@0.160.0/examples/js/shaders/LuminosityHighPassShader.js\"></script>
  <script src=\"https://unpkg.com/three@0.160.0/examples/js/shaders/CopyShader.js\"></script>
  <style>
    :root {
      --bg: #0a0a0f;
      --bg-glass: rgba(18, 18, 26, 0.7);
      --border: rgba(91, 192, 190, 0.15);
      --border-active: rgba(91, 192, 190, 0.4);
      --text: #fafaff;
      --text-muted: #5a5a6a;
      --cyan: #5bc0be;
      --cyan-dim: #3a8a88;
      --orange: #ee6c4d;
      --glow-cyan: rgba(91, 192, 190, 0.3);
      --font-mono: 'Space Mono', monospace;
      --font-ui: 'Inter', system-ui, sans-serif;
    }
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
      }
    }
    body {
      font-family: var(--font-ui);
      font-size: 14px;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
      height: 100vh;
      height: 100dvh;
    }
    .skip-link {
      position: absolute;
      top: -100px;
      left: 0;
      background: var(--cyan);
      color: var(--bg);
      padding: 8px 16px;
      z-index: 1000;
    }
    .skip-link:focus { top: 0; }

    /* Full-screen canvas */
    #canvas-container {
      position: fixed;
      inset: 0;
      z-index: 0;
    }
    #canvas-container canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    /* Dot grid overlay */
    #canvas-container::before {
      content: '';
      position: absolute;
      inset: 0;
      background-image: radial-gradient(circle, var(--text-muted) 1px, transparent 1px);
      background-size: 50px 50px;
      opacity: 0.03;
      pointer-events: none;
      z-index: 1;
    }

    /* Floating header */
    .header-floating {
      position: fixed;
      top: 24px;
      left: 24px;
      z-index: 10;
    }
    .logo {
      font-family: var(--font-mono);
      font-size: 18px;
      font-weight: 700;
      color: var(--cyan);
      letter-spacing: 4px;
    }

    /* Status overlay */
    .status-overlay {
      position: fixed;
      top: 24px;
      right: 24px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 16px;
      background: var(--bg-glass);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 24px;
      z-index: 10;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
      transition: all 0.3s;
    }
    .status-dot[data-status=\"live\"] {
      background: var(--cyan);
      box-shadow: 0 0 12px var(--cyan);
    }
    .status-dot[data-status=\"error\"] { background: var(--orange); }
    .status-label {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
    }
    .status-time {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
    }

    /* Soul card */
    .soul-card {
      position: fixed;
      bottom: 100px;
      left: 24px;
      width: 180px;
      padding: 16px;
      background: var(--bg-glass);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 12px;
      z-index: 10;
      opacity: 0.85;
      transition: all 0.3s ease;
    }
    .soul-card:hover {
      opacity: 1;
      transform: translateY(-4px);
      border-color: var(--border-active);
    }
    .soul-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .soul-name {
      font-weight: 600;
      font-size: 13px;
      color: var(--text);
    }
    .soul-state {
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 10px;
      background: var(--cyan);
      color: var(--bg);
    }
    .soul-state[data-state=\"dormant\"] {
      background: var(--text-muted);
    }
    .metric-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .metric-row:last-child { margin-bottom: 0; }
    .metric-label {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      width: 14px;
    }
    .bar-track {
      flex: 1;
      height: 4px;
      background: rgba(255,255,255,0.1);
      border-radius: 2px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--cyan), var(--orange));
      border-radius: 2px;
      transition: width 0.3s;
      width: 0%;
    }
    .metric-value {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      width: 28px;
      text-align: right;
    }

    /* Timeline overlay */
    .timeline-overlay {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      height: 70px;
      padding: 12px 24px;
      background: linear-gradient(transparent, rgba(10,10,15,0.9));
      z-index: 10;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
    }
    .timeline-left {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .tick-counter {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
    }
    .tick-value { color: var(--cyan); }
    .fps-counter {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
    }
    .timeline-right {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn-drawer {
      width: 36px;
      height: 36px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--bg-glass);
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .btn-drawer:hover {
      border-color: var(--border-active);
      color: var(--cyan);
    }
    .btn-drawer svg {
      width: 18px;
      height: 18px;
    }

    /* Metrics drawer */
    .metrics-drawer {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: 300px;
      max-width: 90vw;
      padding: 24px;
      background: rgba(12, 12, 18, 0.95);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-left: 1px solid var(--border);
      z-index: 20;
      transform: translateX(100%);
      transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
      overflow-y: auto;
    }
    .metrics-drawer.open { transform: translateX(0); }
    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }
    .drawer-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 2px;
    }
    .btn-close {
      width: 28px;
      height: 28px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 20px;
      cursor: pointer;
    }
    .btn-close:hover { color: var(--cyan); }
    .drawer-section {
      margin-bottom: 24px;
    }
    .drawer-section h3 {
      font-size: 10px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1.5px;
      margin-bottom: 12px;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric-card {
      background: rgba(91, 192, 190, 0.05);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .metric-card-label {
      font-size: 10px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .metric-card-value {
      font-family: var(--font-mono);
      font-size: 18px;
      font-weight: 700;
      color: var(--cyan);
    }

    /* Pulse animation */
    @keyframes pulse-glow {
      0%, 100% { box-shadow: 0 0 0 0 var(--glow-cyan); }
      50% { box-shadow: 0 0 20px 4px var(--glow-cyan); }
    }
    .soul-card[data-active=\"true\"] {
      animation: pulse-glow 2.5s ease-in-out infinite;
    }

    /* Mobile */
    @media (max-width: 768px) {
      .header-floating { top: 16px; left: 16px; }
      .status-overlay { top: 16px; right: 16px; padding: 8px 12px; }
      .soul-card { left: 16px; width: 150px; padding: 12px; }
      .timeline-overlay { padding: 10px 16px; height: 60px; }
    }

    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      border: 0;
    }
  </style>
</head>
<body>
<a href=\"#canvas-container\" class=\"skip-link\">Skip to visualization</a>

<div id=\"canvas-container\" role=\"img\" aria-describedby=\"canvas-desc\">
  <p id=\"canvas-desc\" class=\"sr-only\">
    Interactive 3D visualization of VIVA's holographic memory space.
  </p>
</div>

<header class=\"header-floating\" role=\"banner\">
  <h1 class=\"logo\">VIVA</h1>
</header>

<div class=\"status-overlay\" role=\"status\" aria-live=\"polite\">
  <span class=\"status-dot\" id=\"status-dot\" data-status=\"connecting\"></span>
  <span class=\"status-label\" id=\"status-label\">Connecting</span>
  <span class=\"status-time\" id=\"status-time\">t=0</span>
</div>

<div class=\"soul-card\" id=\"soul-card\" data-active=\"false\" role=\"region\" aria-label=\"Soul status\">
  <div class=\"soul-header\">
    <span class=\"soul-name\">Soul</span>
    <span class=\"soul-state\" id=\"soul-state\" data-state=\"active\">Active</span>
  </div>
  <div class=\"metric-row\" role=\"meter\" aria-label=\"Energy\">
    <span class=\"metric-label\">E</span>
    <div class=\"bar-track\"><div class=\"bar-fill\" id=\"energy-bar\"></div></div>
    <span class=\"metric-value\" id=\"energy-value\">0.00</span>
  </div>
  <div class=\"metric-row\">
    <span class=\"metric-label\">N</span>
    <div class=\"bar-track\"><div class=\"bar-fill\" id=\"bodies-bar\" style=\"background:var(--cyan)\"></div></div>
    <span class=\"metric-value\" id=\"bodies-value\">0</span>
  </div>
</div>

<div class=\"timeline-overlay\">
  <div class=\"timeline-left\">
    <span class=\"tick-counter\">tick: <span class=\"tick-value\" id=\"tick-value\">0</span></span>
    <span class=\"fps-counter\" id=\"fps-counter\">60 fps</span>
  </div>
  <div class=\"timeline-right\">
    <button class=\"btn-drawer\" id=\"btn-drawer\" aria-expanded=\"false\" aria-controls=\"metrics-drawer\" title=\"Show metrics\">
      <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\">
        <path d=\"M4 6h16M4 12h16M4 18h16\"/>
      </svg>
    </button>
  </div>
</div>

<aside class=\"metrics-drawer\" id=\"metrics-drawer\" aria-hidden=\"true\">
  <div class=\"drawer-header\">
    <span class=\"drawer-title\">Metrics</span>
    <button class=\"btn-close\" id=\"btn-close\" aria-label=\"Close\">×</button>
  </div>
  <div class=\"drawer-section\">
    <h3>Population</h3>
    <div class=\"metrics-grid\">
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Bodies</div>
        <div class=\"metric-card-value\" id=\"m-bodies\">0</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Clusters</div>
        <div class=\"metric-card-value\" id=\"m-islands\">0</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Active</div>
        <div class=\"metric-card-value\" id=\"m-awake\">0</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Dormant</div>
        <div class=\"metric-card-value\" id=\"m-sleeping\">0</div>
      </div>
    </div>
  </div>
  <div class=\"drawer-section\">
    <h3>Energy</h3>
    <div class=\"metrics-grid\">
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Mean ε̄</div>
        <div class=\"metric-card-value\" id=\"m-energy\">0.00</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Variance σ²</div>
        <div class=\"metric-card-value\" id=\"m-variance\">0.00</div>
      </div>
    </div>
  </div>
  <div class=\"drawer-section\">
    <h3>System</h3>
    <div class=\"metrics-grid\">
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">CPU</div>
        <div class=\"metric-card-value\" id=\"m-cpu\">--%</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">RAM</div>
        <div class=\"metric-card-value\" id=\"m-ram\">--%</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">GPU</div>
        <div class=\"metric-card-value\" id=\"m-gpu\">--%</div>
      </div>
      <div class=\"metric-card\">
        <div class=\"metric-card-label\">Temp</div>
        <div class=\"metric-card-value\" id=\"m-temp\">--°C</div>
      </div>
    </div>
  </div>
</aside>

<script>
window.addEventListener('load', function() {
  if(typeof THREE==='undefined'){document.getElementById('status-label').textContent='THREE.JS FAILED';return}

  const API='http://localhost:" <> port_str <> "',POLL=50;
  const container=document.getElementById('canvas-container');
  let W=window.innerWidth,H=window.innerHeight;

  // Drawer toggle
  const drawer=document.getElementById('metrics-drawer');
  const btnDrawer=document.getElementById('btn-drawer');
  const btnClose=document.getElementById('btn-close');
  btnDrawer.onclick=()=>{
    const open=drawer.classList.toggle('open');
    drawer.setAttribute('aria-hidden',!open);
    btnDrawer.setAttribute('aria-expanded',open);
  };
  btnClose.onclick=()=>{
    drawer.classList.remove('open');
    drawer.setAttribute('aria-hidden','true');
    btnDrawer.setAttribute('aria-expanded','false');
  };

  // === THREE.JS SCENE ===
  const scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0a0a0f);

  const camera=new THREE.PerspectiveCamera(60,W/H,0.1,1000);
  camera.position.set(0,3,10);

  const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
  renderer.setSize(W,H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.5;
  container.appendChild(renderer.domElement);

  // Post-processing
  let composer=null;
  if(THREE.EffectComposer && THREE.RenderPass && THREE.UnrealBloomPass){
    composer=new THREE.EffectComposer(renderer);
    composer.addPass(new THREE.RenderPass(scene,camera));
    const bloom=new THREE.UnrealBloomPass(new THREE.Vector2(W,H),0.3,0.1,0.9);
    composer.addPass(bloom);
  }

  // Controls
  let controls;
  if(THREE.OrbitControls){
    controls=new THREE.OrbitControls(camera,renderer.domElement);
    controls.target.set(0,0.5,0);
    controls.enableDamping=true;
    controls.dampingFactor=0.05;
    controls.minDistance=3;
    controls.maxDistance=30;
    controls.autoRotate=true;
    controls.autoRotateSpeed=0.3;
  }

  // === BACKGROUND STARS - PIXEL STYLE ===
  const bgParticleCount=200;
  const bgGeom=new THREE.BufferGeometry();
  const bgPositions=new Float32Array(bgParticleCount*3);
  for(let i=0;i<bgParticleCount;i++){
    const i3=i*3;
    const r=15+Math.random()*25;
    const theta=Math.random()*Math.PI*2;
    const phi=Math.acos(2*Math.random()-1);
    bgPositions[i3]=r*Math.sin(phi)*Math.cos(theta);
    bgPositions[i3+1]=r*Math.sin(phi)*Math.sin(theta);
    bgPositions[i3+2]=r*Math.cos(phi);
  }
  bgGeom.setAttribute('position',new THREE.BufferAttribute(bgPositions,3));
  const bgMat=new THREE.PointsMaterial({size:0.15,color:0x5bc0be,transparent:true,opacity:0.9,sizeAttenuation:true});
  const bgParticles=new THREE.Points(bgGeom,bgMat);
  scene.add(bgParticles);

  // === CENTRAL CORE - PIXEL CUBE ===
  const coreGeom=new THREE.BoxGeometry(1.5,1.5,1.5);
  const coreMat=new THREE.MeshBasicMaterial({color:0x5bc0be});
  const core=new THREE.Mesh(coreGeom,coreMat);
  scene.add(core);
  // Wireframe outline
  const coreWireMat=new THREE.MeshBasicMaterial({color:0x1a1a2a,wireframe:true});
  const coreWire=new THREE.Mesh(coreGeom.clone(),coreWireMat);
  coreWire.scale.set(1.02,1.02,1.02);
  scene.add(coreWire);

  // === MEMORY BODIES ===
  const bodies=new Map();
  const connections=new THREE.Group();
  scene.add(connections);

  // Pixel art color palette
  const pixelColors=[0x5bc0be,0x3a8a88,0xee6c4d,0xff9f1c,0xe71d36,0x7209b7,0x4361ee];

  function createBody(b){
    const group=new THREE.Group();
    // PIXEL ART: Use BoxGeometry (cubes!)
    const geom=new THREE.BoxGeometry(0.6,0.6,0.6);
    const energy=b.energy||0;
    // Pick color from palette based on energy
    const colorIdx=Math.floor(energy*6)%pixelColors.length;
    const color=b.sleeping?0x3a3a4a:pixelColors[colorIdx];
    const mat=new THREE.MeshBasicMaterial({color:color});
    const mesh=new THREE.Mesh(geom,mat);
    group.add(mesh);
    // Wireframe outline for pixel look
    const wireMat=new THREE.MeshBasicMaterial({color:0x1a1a2a,wireframe:true});
    const wire=new THREE.Mesh(geom.clone(),wireMat);
    wire.scale.set(1.02,1.02,1.02);
    group.add(wire);
    group.userData={id:b.id,label:b.label,energy:energy};
    return group;
  }

  function updateBodies(bodiesData){
    const currentIds=new Set(bodiesData.map(b=>b.id));
    bodies.forEach((obj,id)=>{
      if(!currentIds.has(id)){scene.remove(obj);bodies.delete(id);}
    });
    bodiesData.forEach(b=>{
      let obj=bodies.get(b.id);
      if(!obj){obj=createBody(b);scene.add(obj);bodies.set(b.id,obj);}
      const p=b.position||[0,0,0,0];
      obj.position.set(p[0]||0,p[1]||0,p[2]||0);
      const energy=b.energy||0;
      const s=0.4+energy*0.6;
      obj.scale.set(s,s,s);
      const mesh=obj.children[0];
      if(mesh && mesh.material){
        const colorIdx=Math.floor(energy*6)%pixelColors.length;
        mesh.material.color.set(b.sleeping?0x3a3a4a:pixelColors[colorIdx]);
      }
    });
    updateConnections(bodiesData);
  }

  function updateConnections(bodiesData){
    while(connections.children.length>0){connections.remove(connections.children[0]);}
    const awakeBodies=bodiesData.filter(b=>!b.sleeping);
    for(let i=0;i<awakeBodies.length;i++){
      for(let j=i+1;j<awakeBodies.length;j++){
        const a=awakeBodies[i],bb=awakeBodies[j];
        const pa=a.position||[0,0,0],pb=bb.position||[0,0,0];
        const dx=pa[0]-pb[0],dy=pa[1]-pb[1],dz=pa[2]-pb[2];
        const dist=Math.sqrt(dx*dx+dy*dy+dz*dz);
        if(dist<5){
          // Pixel style: dashed/dotted line
          const points=[new THREE.Vector3(pa[0],pa[1],pa[2]),new THREE.Vector3(pb[0],pb[1],pb[2])];
          const geom=new THREE.BufferGeometry().setFromPoints(points);
          const mat=new THREE.LineDashedMaterial({color:0x5bc0be,dashSize:0.3,gapSize:0.2,linewidth:2});
          const line=new THREE.Line(geom,mat);
          line.computeLineDistances();
          connections.add(line);
        }
      }
    }
  }

  // === ANIMATION ===
  let time=0;
  let frameCount=0,lastFpsTime=performance.now();

  function animate(){
    requestAnimationFrame(animate);
    time+=0.016;
    frameCount++;

    bgParticles.rotation.y+=0.0001;
    bgParticles.rotation.x+=0.00005;

    // Pixel art: step rotation (not smooth)
    if(Math.floor(time*4)%2===0){
      core.rotation.y+=0.02;
      core.rotation.x+=0.01;
      coreWire.rotation.copy(core.rotation);
    }

    bodies.forEach(obj=>{
      // Pixel bobbing
      obj.position.y+=Math.sin(time*3+obj.userData.id)*0.002;
    });

    if(controls)controls.update();
    if(composer){composer.render();}else{renderer.render(scene,camera);}

    const now=performance.now();
    if(now-lastFpsTime>=1000){
      document.getElementById('fps-counter').textContent=frameCount+' fps';
      frameCount=0;
      lastFpsTime=now;
    }
  }
  animate();

  // === DATA POLLING ===
  const statusDot=document.getElementById('status-dot');
  const statusLabel=document.getElementById('status-label');
  const soulCard=document.getElementById('soul-card');
  let lastTick=-1;

  async function poll(){
    try{
      const r=await fetch(API+'/api/world');
      if(!r.ok)throw new Error('HTTP '+r.status);
      const d=await r.json();

      if(d.error&&!d.bodies){
        statusDot.dataset.status='connecting';
        statusLabel.textContent='No Data';
        return;
      }

      statusDot.dataset.status='live';
      statusLabel.textContent='Live';

      if(d.tick!==lastTick){
        lastTick=d.tick;
        document.getElementById('tick-value').textContent=d.tick||0;
        document.getElementById('status-time').textContent='t='+d.tick;

        if(d.bodies){
          updateBodies(d.bodies);

          const awake=d.bodies.filter(b=>!b.sleeping);
          const avgEnergy=d.bodies.length>0?d.bodies.reduce((s,b)=>s+(b.energy||0),0)/d.bodies.length:0;

          // Soul card
          document.getElementById('energy-bar').style.width=(avgEnergy*100)+'%';
          document.getElementById('energy-value').textContent=avgEnergy.toFixed(2);
          document.getElementById('bodies-bar').style.width=Math.min(100,d.bodies.length*10)+'%';
          document.getElementById('bodies-value').textContent=d.bodies.length;
          document.getElementById('soul-state').textContent=awake.length>0?'Active':'Dormant';
          document.getElementById('soul-state').dataset.state=awake.length>0?'active':'dormant';
          soulCard.dataset.active=awake.length>0?'true':'false';

          // Drawer metrics
          document.getElementById('m-bodies').textContent=d.bodies.length;
          document.getElementById('m-islands').textContent=d.islands?d.islands.length:0;
          document.getElementById('m-awake').textContent=awake.length;
          document.getElementById('m-sleeping').textContent=d.bodies.length-awake.length;
          document.getElementById('m-energy').textContent=avgEnergy.toFixed(3);
          const variance=d.bodies.length>0?d.bodies.reduce((s,b)=>s+Math.pow((b.energy||0)-avgEnergy,2),0)/d.bodies.length:0;
          document.getElementById('m-variance').textContent=variance.toFixed(4);
        }
      }
    }catch(e){
      statusDot.dataset.status='error';
      statusLabel.textContent='Error';
      console.error(e);
    }
  }
  setInterval(poll,POLL);
  poll();

  // Hardware polling
  async function pollHardware(){
    try{
      const r=await fetch(API+'/api/system');
      if(!r.ok)return;
      const sys=await r.json();
      if(sys.cpu)document.getElementById('m-cpu').textContent=(sys.cpu.usage_percent||0).toFixed(0)+'%';
      if(sys.memory)document.getElementById('m-ram').textContent=sys.memory.percent.toFixed(0)+'%';
      if(sys.gpu){
        document.getElementById('m-gpu').textContent=sys.gpu.usage_percent.toFixed(0)+'%';
        document.getElementById('m-temp').textContent=sys.gpu.temp_celsius+'°C';
      }
    }catch(e){}
  }
  setInterval(pollHardware,2000);
  pollHardware();

  // Resize
  window.addEventListener('resize',()=>{
    W=window.innerWidth;H=window.innerHeight;
    camera.aspect=W/H;
    camera.updateProjectionMatrix();
    renderer.setSize(W,H);
    if(composer)composer.setSize(W,H);
  });
});
</script>
</body>
</html>"
}
