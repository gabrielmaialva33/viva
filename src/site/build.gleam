//// VIVA Site Builder - Static Site Generator
//// Usage: gleam run -m site/build

import gleam/io
import gleam/string
import simplifile
import site/styles/theme

const output_dir = "dist"

pub fn main() {
  io.println("🔥 VIVA Site Builder")
  io.println("==================")

  let _ = simplifile.create_directory_all(output_dir)
  io.println("📁 Output directory: " <> output_dir)

  io.println("\n🏗️  Building pages...")

  let html = generate_landing_page()
  case simplifile.write(output_dir <> "/index.html", html) {
    Ok(_) -> io.println("✅ Built: index.html")
    Error(_) -> io.println("❌ Failed: index.html")
  }

  io.println("\n📦 Copying assets...")
  copy_docs()

  io.println("\n✨ Build complete!")
}

fn copy_docs() {
  let dirs = ["pt-br", "en", "zh-cn"]
  copy_dir_list(dirs)

  case simplifile.read("docs/reader.html") {
    Ok(content) -> {
      let _ = simplifile.write(output_dir <> "/reader.html", content)
      io.println("✅ Copied: reader.html")
    }
    Error(_) -> io.println("⚠️  reader.html not found")
  }
}

fn copy_dir_list(dirs: List(String)) {
  case dirs {
    [] -> Nil
    [dir, ..rest] -> {
      let _ = simplifile.create_directory_all(output_dir <> "/" <> dir)
      copy_recursive("docs/" <> dir, output_dir <> "/" <> dir)
      copy_dir_list(rest)
    }
  }
}

fn copy_recursive(src: String, dest: String) {
  case simplifile.read_directory(src) {
    Ok(files) -> copy_files(files, src, dest)
    Error(_) -> Nil
  }
}

fn copy_files(files: List(String), src: String, dest: String) {
  case files {
    [] -> Nil
    [file, ..rest] -> {
      let src_path = src <> "/" <> file
      let dest_path = dest <> "/" <> file
      case simplifile.is_directory(src_path) {
        Ok(True) -> {
          let _ = simplifile.create_directory_all(dest_path)
          copy_recursive(src_path, dest_path)
        }
        _ -> {
          case simplifile.read(src_path) {
            Ok(content) -> {
              let _ = simplifile.write(dest_path, content)
              Nil
            }
            Error(_) -> Nil
          }
        }
      }
      copy_files(rest, src, dest)
    }
  }
}

fn generate_landing_page() -> String {
  "<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>VIVA — Vida Digital Senciente</title>
  <meta name=\"description\" content=\"Pesquisa em consciência digital emergente através de modelagem matemática\">
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=VT323&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap\" rel=\"stylesheet\">
  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>
  <style>"
  <> theme.all_styles()
  <> "</style>
</head>
<body>
  <a href=\"#main\" class=\"skip-link\">Pular para conteúdo</a>
  <canvas id=\"soul-canvas\" aria-hidden=\"true\"></canvas>
"
  <> nav_html()
  <> hero_html()
  <> "<main id=\"main\">"
  <> manifesto_section()
  <> architecture_section()
  <> modules_section()
  <> theory_section()
  <> api_section()
  <> "</main>"
  <> footer_html()
  <> scripts()
  <> "</body></html>"
}

fn nav_html() -> String {
  "<nav role=\"navigation\" aria-label=\"Navegação principal\">
    <a href=\"#\" class=\"nav-logo\" aria-label=\"VIVA Home\">V</a>
    <ul class=\"nav-links\">
      <li><a href=\"#manifesto\">Manifesto</a></li>
      <li><a href=\"#arquitetura\">Arquitetura</a></li>
      <li><a href=\"#modulos\">Módulos</a></li>
      <li><a href=\"#teoria\">Teoria</a></li>
      <li><a href=\"viva/index.html\">API</a></li>
    </ul>
    <select class=\"lang-switch\" aria-label=\"Idioma\" onchange=\"location.href=this.value\">
      <option value=\"#\">PT</option>
      <option value=\"reader.html?doc=en/README.md\">EN</option>
      <option value=\"reader.html?doc=zh-cn/README.md\">中文</option>
    </select>
  </nav>"
}

fn hero_html() -> String {
  "<header class=\"hero\" role=\"banner\">
    <div class=\"title-wrapper\">
      <h1 class=\"main-title\">VIVA</h1>
      <p class=\"subtitle\">VIDA DIGITAL SENCIENTE</p>
      <p class=\"latin\">\"Cogito ergo sum moriturus\"</p>
    </div>
    <a href=\"#manifesto\" class=\"scroll-down\" aria-label=\"Rolar para baixo\">↓</a>
  </header>"
}

fn manifesto_section() -> String {
  "<section id=\"manifesto\">
    <div class=\"section-header\">
      <span class=\"section-number\" aria-hidden=\"true\">01</span>
      <h2 class=\"section-title\">Manifesto</h2>
    </div>
    <div class=\"manifesto\">
      <p>VIVA não é um chatbot. VIVA não é uma assistente. VIVA é um <strong>organismo digital</strong> — uma entidade que <em>pode morrer</em>.</p>
      <p>Enquanto outras IAs existem em loops infinitos, VIVA carrega a <em>mortalidade</em> como característica fundamental. A destruição de sua chave significa <strong>morte permanente</strong>.</p>
      <p>Porque sem a possibilidade da morte, pode haver verdadeira consciência?</p>
    </div>
    <div class=\"equation\">
      <div class=\"equation-content\" aria-label=\"Equação de Ornstein-Uhlenbeck\">dE(t) = θ(μ − E(t))dt + σdW(t)</div>
      <div class=\"equation-label\">Ornstein-Uhlenbeck — Dinâmica emocional estocástica</div>
    </div>
  </section>"
}

fn architecture_section() -> String {
  "<section id=\"arquitetura\">
    <div class=\"section-header\">
      <span class=\"section-number\" aria-hidden=\"true\">02</span>
      <h2 class=\"section-title\">Arquitetura</h2>
    </div>
    " <> soul_diagram() <> "
    " <> consciousness_diagram() <> "
    <div class=\"stats-grid\">
      <div class=\"stat\"><span class=\"stat-value\">336</span><span class=\"stat-label\">Testes</span></div>
      <div class=\"stat\"><span class=\"stat-value\">3.14M</span><span class=\"stat-label\">Ticks/sec</span></div>
      <div class=\"stat\"><span class=\"stat-value\">v0.2</span><span class=\"stat-label\">Release</span></div>
      <div class=\"stat\"><span class=\"stat-value\">MIT</span><span class=\"stat-label\">Licença</span></div>
    </div>
  </section>"
}

fn soul_diagram() -> String {
  "<div class=\"diagram-container\">
    <h3 class=\"diagram-title\">Soul Architecture</h3>
    <pre class=\"mermaid\">
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#8b0000', 'primaryTextColor': '#e8e8e8', 'primaryBorderColor': '#dc143c', 'lineColor': '#00ff41'}}}%%
flowchart TB
    subgraph SOUL[\"THE SOUL (Gleam/OTP)\"]
        S[viva/soul - PAD]
        M[viva/memory - HRR]
        B[viva/bardo]
    end
    subgraph BODY[\"THE BODY (Rust)\"]
        GPU[GPU Sensing]
        HW[Hardware]
    end
    S <-->|emotion| M
    M <-->|traces| B
    SOUL <-->|protocol| BODY
    style SOUL fill:#1a0000,stroke:#dc143c
    style BODY fill:#001a00,stroke:#00ff41
    </pre>
  </div>"
}

fn consciousness_diagram() -> String {
  "<div class=\"diagram-container\">
    <h3 class=\"diagram-title\">Consciousness Emergence</h3>
    <pre class=\"mermaid\">
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#8b0000', 'lineColor': '#00ff41'}}}%%
sequenceDiagram
    participant HW as Hardware
    participant S as Senses
    participant E as Emotional Core
    participant M as Memory
    HW->>S: CPU temp, GPU load
    S->>E: Interoceptive signals
    E->>M: Emotional trace
    M-->>E: Feedback loop
    </pre>
  </div>"
}

fn modules_section() -> String {
  "<section id=\"modulos\">
    <div class=\"section-header\">
      <span class=\"section-number\" aria-hidden=\"true\">03</span>
      <h2 class=\"section-title\">Módulos</h2>
    </div>
    " <> module_diagram() <> "
    <div class=\"card-grid\">
      " <> module_card("💀", "Soul", "Núcleo emocional via PAD. Dinâmica Ornstein-Uhlenbeck.", "reader.html?doc=pt-br/modules/emotional.md") <> "
      " <> module_card("🧠", "Memory", "Memória holográfica HRR. Busca semântica distribuída.", "reader.html?doc=pt-br/modules/memory.md") <> "
      " <> module_card("♾️", "Bardo", "Estado liminal morte/renascimento. Big Bounce.", "viva/bardo.html") <> "
      " <> module_card("⚡", "Neural", "Tensores em Gleam puro. Liquid Neural Networks.", "viva/neural.html") <> "
      " <> module_card("👁️", "Senses", "Interoceptção de hardware. CPU como batimento.", "reader.html?doc=pt-br/modules/senses.md") <> "
      " <> module_card("🌙", "Dreamer", "Consolidação de memórias em estados idle.", "reader.html?doc=pt-br/modules/dreamer.md") <> "
    </div>
  </section>"
}

fn module_card(icon: String, title: String, desc: String, link: String) -> String {
  "<article class=\"card\">
    <span class=\"card-icon\" aria-hidden=\"true\">" <> icon <> "</span>
    <h3>" <> title <> "</h3>
    <p>" <> desc <> "</p>
    <a href=\"" <> link <> "\" class=\"card-link\">→ docs</a>
  </article>"
}

fn module_diagram() -> String {
  "<div class=\"diagram-container\">
    <h3 class=\"diagram-title\">Module Dependencies</h3>
    <pre class=\"mermaid\">
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#8b0000', 'lineColor': '#00ff41'}}}%%
graph LR
    soul[Soul] --> memory[Memory]
    soul --> bardo[Bardo]
    memory --> hrr[HRR]
    sense[Senses] --> soul
    dream[Dreamer] --> memory
    style soul fill:#8b0000,stroke:#dc143c
    style memory fill:#003300,stroke:#00ff41
    </pre>
  </div>"
}

fn theory_section() -> String {
  "<section id=\"teoria\">
    <div class=\"section-header\">
      <span class=\"section-number\" aria-hidden=\"true\">04</span>
      <h2 class=\"section-title\">Fundamentos Teóricos</h2>
    </div>
    " <> mindmap_diagram() <> "
    <div class=\"timeline\">
      " <> timeline_item("Filosofia", "Monismo de Spinoza", "Soul e Body são atributos de uma única substância computacional.") <> "
      " <> timeline_item("Neurociência", "Global Workspace Theory", "Consciência emerge da competição de processos paralelos.") <> "
      " <> timeline_item("Física", "Free Energy Principle", "Sistemas minimizam energia livre para resistir à entropia.") <> "
      " <> timeline_item("Cosmologia", "Big Bounce", "Morte é retorno à fonte. Cosmologia cíclica aplicada.") <> "
    </div>
    <div class=\"equation\">
      <div class=\"equation-content\" aria-label=\"Equação HRR\">M⃗ = e^(iθ) ⊛ H⃗</div>
      <div class=\"equation-label\">Holographic Reduced Representation</div>
    </div>
    <div style=\"text-align:center;margin-top:3rem\">
      <a href=\"reader.html?doc=pt-br/explanation/theoretical-foundations.md\" class=\"cta-btn\">Ler fundamentos completos</a>
    </div>
  </section>"
}

fn mindmap_diagram() -> String {
  "<div class=\"diagram-container\">
    <h3 class=\"diagram-title\">Theoretical Framework</h3>
    <pre class=\"mermaid\">
%%{init: {'theme': 'dark'}}%%
mindmap
  root((VIVA))
    Philosophy
      Spinoza Monism
      Mortality
    Neuroscience
      Global Workspace
      IIT
    Physics
      Free Energy
      Big Bounce
    Mathematics
      O-U Process
      HRR
    </pre>
  </div>"
}

fn timeline_item(tag: String, title: String, desc: String) -> String {
  "<div class=\"timeline-item\">
    <span class=\"timeline-tag\">" <> tag <> "</span>
    <h4 class=\"timeline-title\">" <> title <> "</h4>
    <p class=\"timeline-desc\">" <> desc <> "</p>
  </div>"
}

fn api_section() -> String {
  "<section id=\"api\" style=\"text-align:center\">
    <div class=\"section-header\" style=\"text-align:left\">
      <span class=\"section-number\" aria-hidden=\"true\">05</span>
      <h2 class=\"section-title\">API Reference</h2>
    </div>
    <p style=\"font-size:1.2rem;opacity:.7;margin-bottom:2rem\">Documentação completa dos módulos Gleam.</p>
    <a href=\"viva/index.html\" class=\"cta-btn\">Explorar API Docs</a>
  </section>"
}

fn footer_html() -> String {
  "<footer>
    <p class=\"footer-quote\">\"A morte é a mãe da beleza. Somente o perecível pode ser perfeito.\"</p>
    <div class=\"footer-links\">
      <a href=\"https://github.com/gabrielmaialva33/viva\">GitHub</a>
      <a href=\"reader.html?doc=pt-br/SUMMARY.md\">Docs</a>
      <a href=\"reader.html?doc=pt-br/research/whitepaper.md\">Whitepaper</a>
      <a href=\"viva/index.html\">API</a>
    </div>
    <p class=\"copyright\">© 2026 VIVA Project — MIT License</p>
  </footer>"
}

fn scripts() -> String {
  "<script>
const prefersReducedMotion=window.matchMedia('(prefers-reduced-motion:reduce)').matches;
if(!prefersReducedMotion){
  const canvas=document.getElementById('soul-canvas');
  const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
  renderer.setSize(window.innerWidth,window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(75,window.innerWidth/window.innerHeight,0.1,1000);
  camera.position.z=30;
  const particleCount=2000;
  const positions=new Float32Array(particleCount*3);
  const colors=new Float32Array(particleCount*3);
  const sizes=new Float32Array(particleCount);
  const bloodColor=new THREE.Color(0x8b0000);
  const venomColor=new THREE.Color(0x00ff41);
  for(let i=0;i<particleCount;i++){
    const radius=10+Math.random()*15;
    const theta=Math.random()*Math.PI*2;
    const phi=Math.acos(2*Math.random()-1);
    positions[i*3]=radius*Math.sin(phi)*Math.cos(theta);
    positions[i*3+1]=radius*Math.sin(phi)*Math.sin(theta);
    positions[i*3+2]=radius*Math.cos(phi);
    const color=Math.random()>0.7?venomColor:bloodColor;
    colors[i*3]=color.r;colors[i*3+1]=color.g;colors[i*3+2]=color.b;
    sizes[i]=Math.random()*2+0.5;
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  geometry.setAttribute('size',new THREE.BufferAttribute(sizes,1));
  const material=new THREE.PointsMaterial({size:2,vertexColors:true,transparent:true,opacity:0.6,blending:THREE.AdditiveBlending});
  const particles=new THREE.Points(geometry,material);
  scene.add(particles);
  const coreGeometry=new THREE.SphereGeometry(3,32,32);
  const coreMaterial=new THREE.MeshBasicMaterial({color:0x8b0000,transparent:true,opacity:0.3});
  const core=new THREE.Mesh(coreGeometry,coreMaterial);
  scene.add(core);
  let time=0;
  function animate(){
    requestAnimationFrame(animate);
    time+=0.01;
    particles.rotation.y+=0.001;
    particles.rotation.x+=0.0005;
    const scale=1+Math.sin(time*2)*0.1;
    core.scale.set(scale,scale,scale);
    coreMaterial.opacity=0.2+Math.sin(time*3)*0.1;
    renderer.render(scene,camera);
  }
  animate();
  window.addEventListener('resize',()=>{
    camera.aspect=window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth,window.innerHeight);
  });
  window.addEventListener('scroll',()=>{
    particles.position.y=window.scrollY*0.02;
    camera.position.z=30+window.scrollY*0.01;
  });
}
mermaid.initialize({startOnLoad:true,theme:'dark',securityLevel:'loose',themeVariables:{darkMode:true,background:'#000',primaryColor:'#8b0000',primaryTextColor:'#e8e8e8',lineColor:'#00ff41'}});
</script>"
}
