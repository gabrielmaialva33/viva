#!/bin/bash
# Inject Mermaid.js into Gleam docs

DOCS_DIR="build/dev/docs/viva"

MERMAID_SCRIPT='
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "neutral" });

  // Find code blocks with mermaid class and render them
  document.querySelectorAll("pre code.language-mermaid, pre.mermaid").forEach((el) => {
    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = el.textContent;
    el.parentElement.replaceWith(container);
  });

  // Re-run mermaid after DOM changes
  mermaid.run();
</script>
</body>'

# Find all HTML files and inject Mermaid before </body>
find "$DOCS_DIR" -name "*.html" -type f | while read file; do
  if ! grep -q "mermaid" "$file"; then
    sed -i "s|</body>|$MERMAID_SCRIPT|g" "$file"
  fi
done

echo "✅ Mermaid injected into $(find "$DOCS_DIR" -name "*.html" | wc -l) HTML files"
