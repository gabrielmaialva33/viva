#!/usr/bin/env python3
"""Inject Mermaid.js into Gleam generated docs."""

import sys
from pathlib import Path

MERMAID_SCRIPT = '''<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "neutral" });
  document.querySelectorAll("pre code.language-mermaid, pre.mermaid").forEach((el) => {
    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = el.textContent;
    el.parentElement.replaceWith(container);
  });
  mermaid.run();
</script>
</body>'''

def inject_mermaid(docs_dir: str) -> int:
    """Inject Mermaid.js into all HTML files in docs_dir."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"❌ Directory not found: {docs_dir}", file=sys.stderr)
        return 1

    count = 0
    for html_file in docs_path.rglob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        if "mermaid.esm" not in content:
            content = content.replace("</body>", MERMAID_SCRIPT)
            html_file.write_text(content, encoding="utf-8")
            count += 1

    print(f"✅ Mermaid injected into {count} HTML files")
    return 0

if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "build/dev/docs/viva"
    sys.exit(inject_mermaid(docs_dir))
