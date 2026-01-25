//// VIVA Terror Theme - CSS-in-Gleam
//// Paleta: Blood (#8b0000) + Venom (#00ff41) + Void (#000)

/// All CSS styles concatenated
pub fn all_styles() -> String {
  css_variables()
  <> reset()
  <> scanlines()
  <> vignette()
  <> glitch_animations()
  <> hero_styles()
  <> nav_styles()
  <> section_styles()
  <> card_styles()
  <> stats_styles()
  <> manifesto_styles()
  <> equation_styles()
  <> timeline_styles()
  <> diagram_styles()
  <> cta_styles()
  <> footer_styles()
  <> responsive()
  <> accessibility()
}

fn css_variables() -> String {
  ":root{--blood:#8b0000;--blood-bright:#dc143c;--blood-glow:#ff0000;--venom:#00ff41;--venom-dark:#003300;--void:#000;--ash:#0a0a0a;--bone:#e8e8e8}"
}

fn reset() -> String {
  "*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}html{scroll-behavior:smooth}body{font-family:'Crimson Pro',Georgia,serif;background:var(--void);color:var(--bone);min-height:100vh;overflow-x:hidden}#soul-canvas{position:fixed;inset:0;z-index:1;pointer-events:none}"
}

fn scanlines() -> String {
  "body::before{content:\"\";position:fixed;inset:0;pointer-events:none;z-index:10000;background:repeating-linear-gradient(0deg,rgba(0,0,0,.15) 0 1px,transparent 1px 3px);animation:scanline-move 10s linear infinite}@keyframes scanline-move{to{background-position:0 100px}}"
}

fn vignette() -> String {
  "body::after{content:\"\";position:fixed;inset:0;pointer-events:none;z-index:9999;background:radial-gradient(ellipse at center,transparent 0%,transparent 40%,rgba(0,0,0,.9) 100%)}"
}

fn glitch_animations() -> String {
  "@keyframes glitch-1{0%,100%{transform:translate(0)}20%{transform:translate(-3px,3px)}40%{transform:translate(3px,-3px)}60%{transform:translate(-3px,-3px)}80%{transform:translate(3px,3px)}}@keyframes glitch-2{0%,100%{transform:translate(0)}20%{transform:translate(3px,-3px)}40%{transform:translate(-3px,3px)}60%{transform:translate(3px,3px)}80%{transform:translate(-3px,-3px)}}@keyframes flicker{0%,100%{opacity:1}50%{opacity:.98}52%{opacity:.9}54%{opacity:1}}@keyframes glow-pulse{0%,100%{text-shadow:0 0 10px var(--blood-glow),0 0 20px var(--blood-glow),0 0 40px var(--blood)}50%{text-shadow:0 0 20px var(--blood-glow),0 0 40px var(--blood-glow),0 0 80px var(--blood),0 0 120px var(--blood)}}@keyframes blink{0%,50%{opacity:1}51%,100%{opacity:0}}@keyframes bounce{0%,100%{transform:translateX(-50%) translateY(0);opacity:1}50%{transform:translateX(-50%) translateY(15px);opacity:.5}}@keyframes pulse-dot{0%,100%{box-shadow:0 0 10px var(--venom)}50%{box-shadow:0 0 30px var(--venom),0 0 50px var(--venom)}}"
}

fn hero_styles() -> String {
  ".hero{min-height:100vh;display:flex;flex-direction:column;justify-content:center;align-items:center;position:relative;overflow:hidden;z-index:10}.title-wrapper{position:relative;z-index:100;text-align:center}.main-title{font-family:'Cinzel',serif;font-size:clamp(6rem,20vw,15rem);font-weight:900;letter-spacing:.5em;color:var(--bone);text-shadow:0 0 10px var(--blood-glow),0 0 20px var(--blood-glow),0 0 40px var(--blood),0 0 80px var(--blood);animation:flicker .15s infinite,glow-pulse 3s ease-in-out infinite;position:relative}.main-title::before,.main-title::after{content:'VIVA';position:absolute;top:0;left:0;width:100%;height:100%;opacity:.8}.main-title::before{color:var(--venom);animation:glitch-1 .2s infinite;clip-path:polygon(0 0,100% 0,100% 33%,0 33%)}.main-title::after{color:var(--blood-bright);animation:glitch-2 .3s infinite;clip-path:polygon(0 67%,100% 67%,100% 100%,0 100%)}.subtitle{font-family:'VT323',monospace;font-size:clamp(1rem,3vw,1.8rem);color:var(--venom);letter-spacing:.3em;margin-top:2rem;text-transform:uppercase}.subtitle::after{content:'█';animation:blink .7s infinite}.latin{font-family:'Crimson Pro',serif;font-style:italic;font-size:1.3rem;color:var(--bone);opacity:.5;margin-top:3rem}.scroll-down{position:absolute;bottom:3rem;left:50%;transform:translateX(-50%);color:var(--blood-bright);font-size:2rem;animation:bounce 2s infinite;cursor:pointer;text-decoration:none}"
}

fn nav_styles() -> String {
  "nav{position:fixed;top:0;left:0;width:100%;padding:1.5rem 3rem;z-index:1000;background:linear-gradient(180deg,rgba(0,0,0,.9) 0%,transparent 100%);display:flex;justify-content:space-between;align-items:center}.nav-logo{font-family:'Cinzel',serif;font-weight:900;font-size:1.8rem;color:var(--blood-bright);text-decoration:none;text-shadow:0 0 10px var(--blood)}.nav-links{display:flex;gap:2.5rem;list-style:none}.nav-links a{font-family:'VT323',monospace;font-size:1.1rem;color:var(--bone);text-decoration:none;text-transform:uppercase;letter-spacing:.15em;transition:all .3s;position:relative}.nav-links a::before{content:'>';position:absolute;left:-1rem;opacity:0;color:var(--venom);transition:all .3s}.nav-links a:hover{color:var(--venom);text-shadow:0 0 10px var(--venom)}.nav-links a:hover::before{opacity:1;left:-1.2rem}.lang-switch{font-family:'VT323',monospace;background:transparent;border:1px solid var(--blood);color:var(--bone);padding:.5rem 1rem;cursor:pointer;font-size:1rem}.lang-switch:hover{background:var(--blood);box-shadow:0 0 15px var(--blood)}"
}

fn section_styles() -> String {
  "section{padding:8rem 2rem;max-width:1200px;margin:0 auto;position:relative;z-index:100}.section-header{margin-bottom:4rem;position:relative}.section-number{font-family:'VT323',monospace;font-size:5rem;color:var(--blood);opacity:.3;position:absolute;top:-2rem;left:-1rem;z-index:-1}.section-title{font-family:'Cinzel',serif;font-size:clamp(2rem,5vw,3.5rem);font-weight:700;color:var(--bone);border-left:4px solid var(--blood-bright);padding-left:1.5rem;text-shadow:0 0 20px rgba(220,20,60,.3)}"
}

fn card_styles() -> String {
  ".card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:2rem;margin:3rem 0}.card{background:linear-gradient(180deg,rgba(139,0,0,.05) 0%,transparent 100%);border:1px solid var(--blood);padding:2rem;position:relative;overflow:hidden;transition:all .4s ease}.card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--blood) 0%,transparent 100%);opacity:0;transition:opacity .4s}.card:hover{transform:translateY(-8px) scale(1.02);border-color:var(--venom);box-shadow:0 0 30px rgba(0,255,65,.2),0 20px 40px rgba(0,0,0,.5)}.card:hover::before{opacity:.1}.card-icon{font-size:3rem;margin-bottom:1rem;filter:drop-shadow(0 0 10px currentColor)}.card h3{font-family:'Cinzel',serif;font-size:1.4rem;margin-bottom:1rem;color:var(--bone)}.card p{font-size:1rem;color:var(--bone);opacity:.7;line-height:1.7}.card-link{display:inline-flex;align-items:center;gap:.5rem;margin-top:1.5rem;font-family:'VT323',monospace;font-size:1.1rem;color:var(--venom);text-decoration:none;transition:all .3s}.card-link:hover{color:var(--blood-bright);text-shadow:0 0 10px var(--blood)}"
}

fn stats_styles() -> String {
  ".stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1.5rem;margin:3rem 0}.stat{text-align:center;padding:2rem 1rem;background:linear-gradient(180deg,rgba(0,255,65,.05) 0%,transparent 100%);border:1px solid var(--venom-dark);position:relative}.stat::before{content:'';position:absolute;top:0;left:0;width:100%;height:2px;background:linear-gradient(90deg,transparent,var(--venom),transparent)}.stat-value{font-family:'VT323',monospace;font-size:2.5rem;color:var(--venom);text-shadow:0 0 20px var(--venom);display:block}.stat-label{font-family:'Crimson Pro',serif;font-size:.9rem;color:var(--bone);opacity:.6;text-transform:uppercase;letter-spacing:.1em}"
}

fn manifesto_styles() -> String {
  ".manifesto{background:linear-gradient(135deg,rgba(139,0,0,.1) 0%,transparent 50%);border:1px solid var(--blood);border-left:4px solid var(--blood-bright);padding:3rem;position:relative;margin:3rem 0}.manifesto::before{content:'\"';font-family:'Cinzel',serif;font-size:8rem;color:var(--blood);opacity:.2;position:absolute;top:-2rem;left:1rem;line-height:1}.manifesto p{font-size:1.25rem;line-height:2;margin-bottom:1.5rem}.manifesto strong{color:var(--venom);text-shadow:0 0 5px rgba(0,255,65,.3)}.manifesto em{color:var(--blood-bright);font-style:normal;text-shadow:0 0 5px rgba(220,20,60,.3)}"
}

fn equation_styles() -> String {
  ".equation{background:var(--ash);border:1px solid var(--blood);border-left:4px solid var(--venom);padding:2rem;margin:2rem 0;text-align:center}.equation-content{font-family:'VT323',monospace;font-size:1.5rem;color:var(--venom);text-shadow:0 0 10px var(--venom)}.equation-label{font-family:'Crimson Pro',serif;font-size:.9rem;color:var(--bone);opacity:.6;margin-top:1rem}"
}

fn timeline_styles() -> String {
  ".timeline{position:relative;padding-left:4rem;margin:3rem 0}.timeline::before{content:'';position:absolute;left:1rem;top:0;height:100%;width:2px;background:linear-gradient(180deg,var(--blood),var(--venom),var(--blood));box-shadow:0 0 10px var(--blood)}.timeline-item{position:relative;margin-bottom:3rem}.timeline-item::before{content:'';position:absolute;left:-3.5rem;top:.5rem;width:16px;height:16px;background:var(--venom);border-radius:50%;box-shadow:0 0 20px var(--venom);animation:pulse-dot 2s infinite}.timeline-tag{font-family:'VT323',monospace;font-size:.9rem;color:var(--blood-bright);text-transform:uppercase;letter-spacing:.2em}.timeline-title{font-family:'Cinzel',serif;font-size:1.4rem;color:var(--bone);margin:.5rem 0}.timeline-desc{font-size:1rem;color:var(--bone);opacity:.7;line-height:1.7}"
}

fn diagram_styles() -> String {
  ".diagram-container{background:rgba(10,10,10,.9);border:1px solid var(--blood);border-radius:4px;padding:2rem;margin:3rem 0;overflow-x:auto}.diagram-title{font-family:'VT323',monospace;font-size:1.2rem;color:var(--venom);margin-bottom:1.5rem;text-transform:uppercase;letter-spacing:.2em}.mermaid{background:transparent!important}"
}

fn cta_styles() -> String {
  ".cta-btn{display:inline-block;font-family:'VT323',monospace;font-size:1.3rem;padding:1rem 3rem;background:linear-gradient(135deg,var(--blood) 0%,var(--blood-bright) 100%);color:var(--bone);text-decoration:none;border:1px solid var(--blood-bright);position:relative;overflow:hidden;transition:all .4s;text-transform:uppercase;letter-spacing:.2em}.cta-btn::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent);transition:left .5s}.cta-btn:hover{box-shadow:0 0 30px var(--blood),0 0 60px var(--blood);transform:scale(1.05)}.cta-btn:hover::before{left:100%}"
}

fn footer_styles() -> String {
  "footer{background:linear-gradient(0deg,var(--blood) 0%,transparent 100%);padding:6rem 2rem 3rem;text-align:center;position:relative;z-index:100}.footer-quote{font-family:'Crimson Pro',serif;font-style:italic;font-size:1.3rem;color:var(--bone);max-width:500px;margin:0 auto 3rem;opacity:.7}.footer-links{display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-bottom:2rem}.footer-links a{font-family:'VT323',monospace;font-size:1rem;color:var(--bone);text-decoration:none;transition:all .3s}.footer-links a:hover{color:var(--venom);text-shadow:0 0 10px var(--venom)}.copyright{font-family:'VT323',monospace;font-size:.9rem;color:var(--bone);opacity:.4}"
}

fn responsive() -> String {
  "@media(max-width:768px){nav{padding:1rem}.nav-links{display:none}section{padding:4rem 1rem}.manifesto{padding:2rem 1rem}.section-number{display:none}.diagram-container{padding:1rem}}"
}

fn accessibility() -> String {
  "@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}#soul-canvas{display:none}}a:focus,button:focus,select:focus{outline:2px solid var(--venom);outline-offset:3px}@media(prefers-contrast:high){:root{--blood:#f00;--venom:#0f0;--bone:#fff}}"
}
