//// VIVA Terminal Theme - CSS-in-Gleam
//// Paleta: Danger (#ff4d4d) + Success (#00ff66) + Void (#0a0a0a)
//// WCAG AA Compliant + Cyberpunk Animations

/// All CSS styles concatenated
pub fn all_styles() -> String {
  css_variables()
  <> reset()
  <> animations()
  <> scanlines()
  <> nav_styles()
  <> hero_styles()
  <> section_styles()
  <> philo_styles()
  <> concepts_styles()
  <> tech_styles()
  <> cta_styles()
  <> footer_styles()
  <> responsive()
  <> accessibility()
}

fn css_variables() -> String {
  ":root{--danger:#ff4d4d;--danger-glow:rgba(255,77,77,.4);--success:#00ff66;--success-glow:rgba(0,255,102,.4);--warning:#ffaa00;--info:#00aaff;--bg-primary:#0a0a0a;--bg-secondary:#0f0f0f;--bg-card:#111111;--text-primary:#ffffff;--text-secondary:#a3a3a3;--text-muted:#888888;--text-ghost:#525252;--border:#262626;--font-mono:'JetBrains Mono',monospace}"
}

fn reset() -> String {
  "*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}html{scroll-behavior:smooth}body{font-family:var(--font-mono);background:var(--bg-primary);color:var(--text-primary);line-height:1.6;min-height:100vh;overflow-x:hidden}a{color:inherit;text-decoration:none}.skip-link{position:absolute;top:-100px;left:0;background:var(--success);color:var(--bg-primary);padding:1rem 2rem;z-index:100000;font-weight:600}.skip-link:focus{top:0}.red{color:var(--danger)}.green{color:var(--success)}"
}

fn animations() -> String {
  // Keyframes for all animations
  "@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
@keyframes blink{0%,49%{opacity:1}50%,100%{opacity:0}}
@keyframes glitch{0%{transform:translate(0)}20%{transform:translate(-3px,3px)}40%{transform:translate(-3px,-3px)}60%{transform:translate(3px,3px)}80%{transform:translate(3px,-3px)}100%{transform:translate(0)}}
@keyframes glitch-skew{0%{transform:skew(0deg)}20%{transform:skew(-2deg)}40%{transform:skew(2deg)}60%{transform:skew(-1deg)}80%{transform:skew(1deg)}100%{transform:skew(0deg)}}
@keyframes glow-pulse{0%,100%{box-shadow:0 0 5px var(--success-glow),0 0 10px var(--success-glow)}50%{box-shadow:0 0 20px var(--success-glow),0 0 40px var(--success-glow)}}
@keyframes glow-pulse-red{0%,100%{box-shadow:0 0 5px var(--danger-glow),0 0 10px var(--danger-glow)}50%{box-shadow:0 0 20px var(--danger-glow),0 0 40px var(--danger-glow)}}
@keyframes text-glow{0%,100%{text-shadow:0 0 10px var(--danger-glow),0 0 20px var(--danger-glow),0 0 30px var(--danger-glow)}50%{text-shadow:0 0 20px var(--danger-glow),0 0 40px var(--danger-glow),0 0 60px var(--danger-glow)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes scan{0%{top:-100%}100%{top:100%}}
@keyframes typing{from{width:0}to{width:100%}}
@keyframes grid-move{0%{background-position:0 0}100%{background-position:50px 50px}}"
}

fn scanlines() -> String {
  // CRT scanlines overlay + animated grid background
  ".scanlines{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9999;background:repeating-linear-gradient(0deg,rgba(0,0,0,.1) 0px,rgba(0,0,0,.1) 1px,transparent 1px,transparent 2px);opacity:.15}
.grid-bg{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:-1;background-image:linear-gradient(rgba(0,255,102,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,102,.03) 1px,transparent 1px);background-size:50px 50px;animation:grid-move 20s linear infinite}
.scan-line{position:fixed;left:0;width:100%;height:4px;background:linear-gradient(transparent,rgba(0,255,102,.1),transparent);pointer-events:none;z-index:9998;animation:scan 8s linear infinite}"
}

fn nav_styles() -> String {
  ".nav{display:flex;justify-content:space-between;align-items:center;padding:0 80px;height:72px;border-bottom:1px solid var(--border);position:sticky;top:0;background:rgba(10,10,10,.95);backdrop-filter:blur(10px);z-index:1000}
.nav-logo{display:flex;align-items:center;gap:4px;font-size:20px;font-weight:700;transition:all .3s}
.nav-logo:hover{text-shadow:0 0 10px var(--danger-glow)}
.nav-links{display:flex;align-items:center;gap:32px;list-style:none}
.nav-links a{color:var(--text-secondary);font-size:12px;transition:all .3s;position:relative}
.nav-links a::after{content:'';position:absolute;bottom:-4px;left:0;width:0;height:1px;background:var(--success);transition:width .3s}
.nav-links a:hover{color:var(--success)}
.nav-links a:hover::after{width:100%}
.nav-cta{background:var(--danger);color:var(--bg-primary)!important;padding:10px 20px;font-size:12px;font-weight:600;border-bottom:2px solid #cc2222;transition:all .3s!important}
.nav-cta::after{display:none!important}
.nav-cta:hover{transform:translateY(-2px);box-shadow:0 4px 20px var(--danger-glow)}
.nav-status{display:flex;align-items:center;gap:8px}
.nav-status-dot{width:8px;height:8px;background:var(--success);border-radius:50%;animation:pulse 2s infinite;box-shadow:0 0 10px var(--success-glow)}
.nav-status-text{color:var(--success);font-size:11px;letter-spacing:2px}
.nav-deco{color:var(--danger);font-size:12px;opacity:.5;animation:blink 1s infinite}"
}

fn hero_styles() -> String {
  ".hero{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:32px;padding:120px 80px;min-height:100vh;position:relative;overflow:hidden}
.hero-tag{padding:8px 16px;border:1px solid var(--danger);opacity:.9;animation:glow-pulse-red 3s infinite}
.hero-tag-text{color:var(--danger);font-size:11px;letter-spacing:3px}
.hero-headline{display:flex;flex-direction:column;align-items:center;gap:8px}
.hero-h1{color:var(--text-secondary);font-size:72px;font-weight:700;letter-spacing:8px}
.hero-h2{color:var(--danger);font-size:96px;font-weight:700;letter-spacing:12px;animation:text-glow 2s ease-in-out infinite,glitch-skew 10s infinite;position:relative}
.hero-h2::before,.hero-h2::after{content:'MORTAL';position:absolute;top:0;left:0;width:100%;height:100%}
.hero-h2::before{color:var(--success);animation:glitch .3s infinite;clip-path:polygon(0 0,100% 0,100% 45%,0 45%);transform:translate(-2px,-2px)}
.hero-h2::after{color:var(--info);animation:glitch .3s infinite reverse;clip-path:polygon(0 55%,100% 55%,100% 100%,0 100%);transform:translate(2px,2px)}
.hero-sub{color:var(--text-secondary);font-size:18px;text-align:center;line-height:1.8;max-width:600px}
.hero-cursor{display:inline-block;width:12px;height:24px;background:var(--success);margin-left:8px;animation:blink .8s infinite}
.hero-ctas{display:flex;gap:16px;padding-top:24px}
.btn-primary{background:var(--success);color:var(--bg-primary);padding:14px 28px;font-size:13px;font-weight:600;border:none;border-bottom:3px solid #00aa44;transition:all .3s;position:relative;overflow:hidden}
.btn-primary::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent);transition:left .5s}
.btn-primary:hover::before{left:100%}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 4px 30px var(--success-glow)}
.btn-outline{padding:14px 28px;font-size:13px;color:var(--text-muted);border:1px solid var(--text-muted);transition:all .3s;position:relative}
.btn-outline:hover{color:var(--success);border-color:var(--success);box-shadow:0 0 20px var(--success-glow)}
.hero-stats{display:flex;gap:48px;padding-top:48px}
.stat{display:flex;flex-direction:column;align-items:center;gap:4px;padding:16px;background:var(--bg-secondary);border:1px solid var(--border);transition:all .3s}
.stat:hover{border-color:var(--success);box-shadow:0 0 20px var(--success-glow)}
.stat-value{color:var(--success);font-size:28px;font-weight:700;text-shadow:0 0 10px var(--success-glow)}
.stat-label{color:var(--text-muted);font-size:11px;letter-spacing:1px}
.stat-pulse{width:6px;height:6px;background:var(--success);border-radius:50%;animation:pulse 1.5s infinite}
.hero-glitch{position:absolute;font-size:120px;font-weight:700;letter-spacing:16px;opacity:.03;pointer-events:none;animation:float 6s ease-in-out infinite}
.hero-glitch-1{color:var(--danger);top:20%;left:5%}
.hero-glitch-2{color:var(--success);bottom:20%;right:5%;animation-delay:-3s}
.hero-line{position:absolute;height:1px;opacity:.6}
.hero-line-1{background:linear-gradient(90deg,transparent,var(--danger),transparent);width:200px;bottom:25%;left:5%;animation:pulse 3s infinite}
.hero-line-2{background:linear-gradient(90deg,transparent,var(--success),transparent);width:150px;bottom:20%;right:8%;animation:pulse 3s infinite 1.5s}
.hero-version{position:absolute;bottom:40px;right:80px;color:var(--text-ghost);font-size:11px;letter-spacing:1px}"
}

fn section_styles() -> String {
  "section{padding:100px 80px;position:relative}
.section-alt{background:var(--bg-secondary)}
.section-header{display:flex;flex-direction:column;align-items:center;gap:16px;margin-bottom:64px}
.section-tag{font-size:11px;letter-spacing:3px;padding:4px 12px;border:1px solid currentColor}
.section-tag.green{color:var(--success)}
.section-tag.red{color:var(--danger)}
.section-title{font-size:42px;font-weight:700;text-align:center;background:linear-gradient(135deg,var(--text-primary),var(--text-secondary));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.section-sub{color:var(--text-secondary);font-size:14px;text-align:center;line-height:1.8;max-width:700px}
.section-deco{color:var(--text-ghost);font-size:10px;letter-spacing:3px;margin-top:48px;opacity:.5}
.separator{display:flex;align-items:center;justify-content:center;gap:16px;padding-top:48px}
.separator-line{width:200px;height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent)}
.separator-dot{font-size:10px;animation:pulse 2s infinite}
.separator-dot.red{color:var(--danger)}
.separator-dot.green{color:var(--success)}"
}

fn philo_styles() -> String {
  ".philo-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
.philo-card{background:var(--bg-card);padding:32px;display:flex;flex-direction:column;gap:16px;transition:all .4s;position:relative;overflow:hidden}
.philo-card::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:2px;transition:left .4s}
.philo-card:hover::before{left:100%}
.philo-card.red{border-top:2px solid var(--danger);border-left:1px solid rgba(255,77,77,.3)}
.philo-card.red::before{background:var(--danger)}
.philo-card.red:hover{box-shadow:0 0 30px var(--danger-glow);border-color:var(--danger)}
.philo-card.green{border-top:2px solid var(--success);border-right:1px solid rgba(0,255,102,.3)}
.philo-card.green::before{background:var(--success)}
.philo-card.green:hover{box-shadow:0 0 30px var(--success-glow);border-color:var(--success)}
.philo-card-title{font-size:14px;font-weight:700;letter-spacing:3px;transition:text-shadow .3s}
.philo-card.red .philo-card-title{color:var(--danger)}
.philo-card.green .philo-card-title{color:var(--success)}
.philo-card:hover .philo-card-title{text-shadow:0 0 20px currentColor}
.philo-card-desc{color:var(--text-secondary);font-size:13px;line-height:1.8}
.philo-card-dots{display:flex;gap:6px;padding-top:12px}
.philo-dot{width:4px;height:4px;border-radius:50%;transition:all .3s}
.philo-dot.red{background:var(--danger)}
.philo-dot.green{background:var(--success)}
.philo-card:hover .philo-dot{box-shadow:0 0 10px currentColor;transform:scale(1.5)}"
}

fn concepts_styles() -> String {
  ".concepts-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}
.concepts-col{display:flex;flex-direction:column;gap:24px}
.concept-card{background:var(--bg-card);padding:32px;border:1px solid var(--border);display:flex;flex-direction:column;gap:16px;transition:all .4s;position:relative}
.concept-card::after{content:'';position:absolute;bottom:0;left:0;width:0;height:2px;background:var(--success);transition:width .4s}
.concept-card:hover::after{width:100%}
.concept-card:hover{transform:translateY(-4px);border-color:var(--success)}
.concept-card.highlight{border-color:var(--danger);border-left-width:3px}
.concept-card.highlight::after{background:var(--danger)}
.concept-card.highlight:hover{border-color:var(--danger);box-shadow:0 0 30px var(--danger-glow)}
.concept-card-header{display:flex;justify-content:space-between;align-items:center}
.concept-card-title{font-size:18px;font-weight:700;transition:text-shadow .3s}
.concept-card-title.red{color:var(--danger)}
.concept-card-title.green{color:var(--success)}
.concept-card:hover .concept-card-title{text-shadow:0 0 15px currentColor}
.concept-card-tag{color:var(--text-ghost);font-size:10px;letter-spacing:1px;padding:4px 8px;border:1px solid var(--border)}
.concept-card-desc{color:var(--text-secondary);font-size:13px;line-height:1.8}
.concept-card-formula{color:var(--success);font-size:12px;padding:12px;background:var(--bg-primary);border-left:2px solid var(--success);font-family:var(--font-mono)}"
}

fn tech_styles() -> String {
  ".tech-grid{display:flex;gap:32px;justify-content:center;flex-wrap:wrap}
.tech-card{background:var(--bg-card);padding:32px 24px;display:flex;flex-direction:column;align-items:center;gap:12px;border:1px solid var(--border);border-bottom:3px solid var(--success);min-width:160px;transition:all .4s;position:relative;overflow:hidden}
.tech-card::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle,var(--success-glow) 0%,transparent 70%);opacity:0;transition:opacity .4s}
.tech-card:hover::before{opacity:.1}
.tech-card:hover{transform:translateY(-8px);box-shadow:0 10px 40px var(--success-glow);border-color:var(--success)}
.tech-card-name{color:var(--success);font-size:20px;font-weight:700;transition:text-shadow .3s}
.tech-card:hover .tech-card-name{text-shadow:0 0 20px var(--success-glow)}
.tech-card-desc{color:var(--text-muted);font-size:11px;letter-spacing:1px}
.tech-card-label{color:var(--success);font-size:10px;opacity:.3;transition:opacity .3s}
.tech-card:hover .tech-card-label{opacity:1}"
}

fn cta_styles() -> String {
  ".cta{display:flex;flex-direction:column;align-items:center;gap:32px;padding:120px 80px;position:relative;overflow:hidden}
.cta::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--danger),transparent)}
.cta::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--danger),transparent)}
.cta-title{font-size:48px;font-weight:700;background:linear-gradient(135deg,var(--text-primary),var(--danger));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.cta-sub{color:var(--text-secondary);font-size:18px;text-align:center}
.cta-btns{display:flex;gap:16px}
.cta-glow{width:300px;height:2px;background:linear-gradient(90deg,transparent 0%,var(--danger) 50%,transparent 100%);animation:pulse 2s infinite}
.cta-hash{color:var(--text-ghost);font-size:11px;letter-spacing:2px}
.cta-deco{color:var(--text-ghost);font-size:10px;letter-spacing:3px}
.btn-outline-red{padding:16px 32px;font-size:14px;color:var(--danger);border:1px solid var(--danger);background:transparent;transition:all .3s;position:relative;overflow:hidden}
.btn-outline-red::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:var(--danger);transition:left .3s;z-index:-1}
.btn-outline-red:hover::before{left:0}
.btn-outline-red:hover{color:var(--bg-primary)}"
}

fn footer_styles() -> String {
  ".footer{display:flex;justify-content:space-between;align-items:center;padding:48px 80px;border-top:1px solid var(--border);background:var(--bg-secondary)}
.footer-left{display:flex;flex-direction:column;gap:8px}
.footer-logo{display:flex;gap:4px;font-size:20px;font-weight:700}
.footer-tagline{color:var(--text-muted);font-size:11px;letter-spacing:1px}
.footer-links{display:flex;gap:32px}
.footer-links a{color:var(--text-secondary);font-size:12px;transition:all .3s;position:relative}
.footer-links a:hover{color:var(--success)}
.footer-copy{color:var(--text-ghost);font-size:11px}
.footer-glyph{color:var(--success);font-size:11px;animation:pulse 3s infinite}
.footer-bottom{display:flex;justify-content:center;padding:24px 80px;background:#050505}
.footer-ascii{color:var(--text-ghost);font-size:12px;letter-spacing:6px;animation:pulse 4s infinite}"
}

fn responsive() -> String {
  "@media(max-width:1024px){.nav{padding:0 24px}section{padding:60px 24px}.hero{padding:80px 24px}.philo-grid{grid-template-columns:1fr}.concepts-grid{grid-template-columns:1fr}.tech-grid{flex-wrap:wrap}.hero-h2{font-size:64px}.hero-h1{font-size:48px}.cta{padding:80px 24px}.footer{padding:32px 24px}}@media(max-width:768px){.nav-links{display:none}.nav-deco{display:none}.hero-stats{flex-wrap:wrap;justify-content:center;gap:16px}.stat{flex:1;min-width:120px}.footer{flex-direction:column;gap:24px;text-align:center}.footer-links{justify-content:center}.hero-glitch{display:none}.hero-line{display:none}.cta-title{font-size:32px}.section-title{font-size:28px}}"
}

fn accessibility() -> String {
  "@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}.scanlines,.scan-line,.grid-bg{display:none}}a:focus,button:focus{outline:2px solid var(--success);outline-offset:3px}@media(prefers-contrast:high){:root{--danger:#ff0000;--success:#00ff00;--text-primary:#ffffff;--text-secondary:#cccccc}.scanlines,.scan-line,.grid-bg{display:none}}"
}
