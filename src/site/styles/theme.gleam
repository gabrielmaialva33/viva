//// VIVA Terminal Theme - CSS-in-Gleam
//// Paleta: Danger (#ff4d4d) + Success (#00ff66) + Void (#0a0a0a)
//// WCAG AA Compliant

/// All CSS styles concatenated
pub fn all_styles() -> String {
  css_variables()
  <> reset()
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
  ":root{--danger:#ff4d4d;--success:#00ff66;--warning:#ffaa00;--info:#00aaff;--bg-primary:#0a0a0a;--bg-secondary:#0f0f0f;--text-primary:#ffffff;--text-secondary:#a3a3a3;--text-muted:#888888;--text-ghost:#525252;--border:#262626;--font-mono:'JetBrains Mono',monospace}"
}

fn reset() -> String {
  "*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}html{scroll-behavior:smooth}body{font-family:var(--font-mono);background:var(--bg-primary);color:var(--text-primary);line-height:1.6;min-height:100vh}a{color:inherit;text-decoration:none}.skip-link{position:absolute;top:-100px;left:0;background:var(--success);color:var(--bg-primary);padding:1rem 2rem;z-index:100000;font-weight:600}.skip-link:focus{top:0}"
}

fn nav_styles() -> String {
  ".nav{display:flex;justify-content:space-between;align-items:center;padding:0 80px;height:72px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--bg-primary);z-index:1000}.nav-logo{display:flex;align-items:center;gap:4px;font-size:20px;font-weight:700}.nav-logo .red{color:var(--danger)}.nav-logo .green{color:var(--success)}.nav-links{display:flex;align-items:center;gap:32px;list-style:none}.nav-links a{color:var(--text-secondary);font-size:12px;transition:color .2s}.nav-links a:hover{color:var(--text-primary)}.nav-cta{background:var(--danger);color:var(--bg-primary);padding:10px 20px;font-size:12px;font-weight:600;border-bottom:2px solid #cc2222}.nav-cta:hover{filter:brightness(1.1)}.nav-status{display:flex;align-items:center;gap:8px}.nav-status-dot{width:6px;height:6px;background:var(--success);border-radius:50%;animation:pulse 2s infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}.nav-status-text{color:var(--success);font-size:11px;letter-spacing:1px}.nav-deco{color:var(--danger);font-size:12px;opacity:.5}"
}

fn hero_styles() -> String {
  ".hero{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:32px;padding:120px 80px;min-height:700px;position:relative;overflow:hidden}.hero-tag{padding:8px 16px;border:1px solid var(--danger);opacity:.9}.hero-tag-text{color:var(--danger);font-size:11px;letter-spacing:2px}.hero-headline{display:flex;flex-direction:column;align-items:center;gap:8px}.hero-h1{color:var(--text-secondary);font-size:72px;font-weight:700;letter-spacing:8px}.hero-h2{color:var(--danger);font-size:96px;font-weight:700;letter-spacing:12px;text-shadow:0 0 60px rgba(255,77,77,.3)}.hero-sub{color:var(--text-secondary);font-size:18px;text-align:center;line-height:1.6;max-width:600px}.hero-ctas{display:flex;gap:16px;padding-top:24px}.btn-primary{background:var(--success);color:var(--bg-primary);padding:14px 28px;font-size:13px;font-weight:600;border-bottom:3px solid #00aa44;transition:transform .2s}.btn-primary:hover{transform:translateY(-2px)}.btn-outline{padding:14px 28px;font-size:13px;color:var(--text-muted);border:1px solid var(--text-muted);transition:all .2s}.btn-outline:hover{color:var(--text-primary);border-color:var(--text-primary)}.hero-stats{display:flex;gap:48px;padding-top:48px}.stat{display:flex;flex-direction:column;align-items:center;gap:4px}.stat-value{color:var(--success);font-size:24px;font-weight:700}.stat-label{color:var(--text-muted);font-size:11px}.stat-pulse{width:6px;height:6px;background:var(--success);border-radius:50%;opacity:.5}.hero-glitch{position:absolute;font-size:96px;font-weight:700;letter-spacing:12px;opacity:.1;pointer-events:none}.hero-glitch-1{color:var(--danger);top:30%;left:10%}.hero-glitch-2{color:var(--success);top:32%;left:12%}.hero-line{position:absolute;height:1px;opacity:.5}.hero-line-1{background:var(--danger);width:120px;bottom:20%;left:5%}.hero-line-2{background:var(--success);width:80px;bottom:18%;right:8%}.hero-version{position:absolute;bottom:40px;right:80px;color:var(--text-muted);font-size:11px}"
}

fn section_styles() -> String {
  "section{padding:100px 80px}.section-alt{background:var(--bg-secondary)}.section-header{display:flex;flex-direction:column;align-items:center;gap:16px;margin-bottom:64px}.section-tag{font-size:11px;letter-spacing:2px}.section-tag.green{color:var(--success)}.section-tag.red{color:var(--danger)}.section-title{font-size:36px;font-weight:700;text-align:center}.section-sub{color:var(--text-secondary);font-size:14px;text-align:center;line-height:1.6;max-width:700px}.section-deco{color:var(--border);font-size:11px;letter-spacing:2px;margin-top:32px}.separator{display:flex;align-items:center;justify-content:center;gap:16px;padding-top:32px}.separator-line{width:200px;height:1px;background:var(--border)}.separator-dot{font-size:8px;opacity:.5}.separator-dot.red{color:var(--danger)}.separator-dot.green{color:var(--success)}.ascii-box{color:#222;font-size:11px}"
}

fn philo_styles() -> String {
  ".philo-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}.philo-card{background:var(--bg-primary);padding:32px;display:flex;flex-direction:column;gap:16px}.philo-card.red{border-top:2px solid var(--danger);border-left:1px solid var(--danger)}.philo-card.green{border-top:2px solid var(--success);border-right:1px solid var(--success)}.philo-card-title{font-size:14px;font-weight:700;letter-spacing:2px}.philo-card.red .philo-card-title{color:var(--danger)}.philo-card.green .philo-card-title{color:var(--success)}.philo-card-desc{color:var(--text-secondary);font-size:12px;line-height:1.7}.philo-card-dots{display:flex;gap:4px;padding-top:12px}.philo-dot{width:4px;height:4px;border-radius:50%;opacity:.3}.philo-dot.red{background:var(--danger)}.philo-dot.green{background:var(--success)}"
}

fn concepts_styles() -> String {
  ".concepts-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}.concepts-col{display:flex;flex-direction:column;gap:24px}.concept-card{background:var(--bg-secondary);padding:32px;border:1px solid var(--border);display:flex;flex-direction:column;gap:16px}.concept-card.highlight{border-color:var(--danger);border-left-width:2px;border-top-width:1px}.concept-card-header{display:flex;justify-content:space-between;align-items:center}.concept-card-title{font-size:18px;font-weight:700}.concept-card-title.red{color:var(--danger)}.concept-card-title.green{color:var(--success)}.concept-card-tag{color:var(--text-ghost);font-size:11px}.concept-card-desc{color:var(--text-secondary);font-size:12px;line-height:1.7}.concept-card-formula{color:var(--success);font-size:11px;opacity:.7;padding-top:8px}"
}

fn tech_styles() -> String {
  ".tech-grid{display:flex;gap:32px;justify-content:center}.tech-card{background:var(--bg-primary);padding:24px;display:flex;flex-direction:column;align-items:center;gap:8px;border-bottom:2px solid var(--success);min-width:140px}.tech-card-name{color:var(--success);font-size:18px;font-weight:700}.tech-card-desc{color:var(--text-muted);font-size:11px}.tech-card-label{color:var(--success);font-size:11px;opacity:.3}"
}

fn cta_styles() -> String {
  ".cta{display:flex;flex-direction:column;align-items:center;gap:32px;padding:120px 80px;border-top:1px solid var(--danger);border-bottom:1px solid var(--danger)}.cta-title{font-size:48px;font-weight:700}.cta-sub{color:var(--text-secondary);font-size:18px;text-align:center}.cta-btns{display:flex;gap:16px}.cta-glow{width:300px;height:2px;background:linear-gradient(90deg,transparent 0%,var(--danger) 50%,transparent 100%);opacity:.3}.cta-hash{color:var(--text-muted);font-size:11px}.cta-deco{color:var(--border);font-size:11px;letter-spacing:2px}.btn-outline-red{padding:16px 32px;font-size:14px;color:var(--danger);border:1px solid var(--danger);background:var(--bg-secondary);transition:all .2s}.btn-outline-red:hover{background:var(--danger);color:var(--bg-primary)}"
}

fn footer_styles() -> String {
  ".footer{display:flex;justify-content:space-between;align-items:center;padding:48px 80px;border-top:1px solid var(--border)}.footer-left{display:flex;flex-direction:column;gap:8px}.footer-logo{display:flex;gap:4px;font-size:18px;font-weight:700}.footer-tagline{color:var(--text-muted);font-size:11px}.footer-links{display:flex;gap:32px}.footer-links a{color:var(--text-secondary);font-size:12px;transition:color .2s}.footer-links a:hover{color:var(--text-primary)}.footer-copy{color:var(--text-muted);font-size:11px}.footer-glyph{color:var(--text-ghost);font-size:11px}.footer-bottom{display:flex;justify-content:center;padding:24px 80px;background:#050505}.footer-ascii{color:#444;font-size:11px;letter-spacing:4px}"
}

fn responsive() -> String {
  "@media(max-width:1024px){.nav{padding:0 24px}section{padding:60px 24px}.hero{padding:80px 24px}.philo-grid{grid-template-columns:1fr}.concepts-grid{grid-template-columns:1fr}.tech-grid{flex-wrap:wrap}.hero-h2{font-size:64px}.hero-h1{font-size:48px}}@media(max-width:768px){.nav-links{display:none}.nav-deco{display:none}.hero-stats{flex-wrap:wrap;justify-content:center}.footer{flex-direction:column;gap:24px;text-align:center}.footer-links{justify-content:center}.hero-glitch{display:none}.hero-line{display:none}}"
}

fn accessibility() -> String {
  "@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}a:focus,button:focus{outline:2px solid var(--success);outline-offset:3px}@media(prefers-contrast:high){:root{--danger:#f00;--success:#0f0;--text-primary:#fff;--text-secondary:#ccc}}"
}
