"""
Render the LangGraph workflow as a PNG.

Usage:
    python scripts/draw_graph.py

Writes docs/graph.png. Uses LangGraph's built-in mermaid renderer,
which posts the graph structure to https://mermaid.ink (free, no auth)
and gets back a PNG. One-time internet call — after this you have the
image and the rest of the pipeline still runs offline.

If the network call fails (offline, firewall), the raw Mermaid source
is written to docs/graph.mmd instead — paste it into
https://mermaid.live to render manually.
"""
from pathlib import Path
import sys

# Make the project root importable so `from src.graph import build_graph`
# works whether you run this from the project root or from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import build_graph


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "docs"
    out_dir.mkdir(exist_ok=True)

    app = build_graph()
    graph = app.get_graph()

    png_path = out_dir / "graph.png"
    try:
        png_bytes = graph.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"Wrote {png_path}")
        return 0
    except Exception as e:
        # Fallback: write the Mermaid source so the user can render it
        # manually at https://mermaid.live
        mmd_path = out_dir / "graph.mmd"
        mmd_path.write_text(graph.draw_mermaid(), encoding="utf-8")
        print(f"PNG render failed ({type(e).__name__}: {e})")
        print(f"Wrote Mermaid source to {mmd_path} — paste it into https://mermaid.live to render.")
        return 1


if __name__ == "__main__":
    sys.exit(main())