"""Embed map_data.json (or fixture_data.json) into template.html -> oslo_demand_rebalancing_map.html."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "__DATA__"
# Repo layout: src/build_html.py reads ../data/map_data.json and writes ../docs/index.html.
# Flat layout (everything in one folder): reads and writes next to the script.
REPO = HERE.name == "src" and (HERE.parent / "docs").is_dir()
DATA_DIR = HERE.parent / "data" if REPO else HERE
OUT_PATH = HERE.parent / "docs" / "index.html" if REPO else HERE / "oslo_demand_rebalancing_map.html"


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        src = Path(arg)
    elif (DATA_DIR / "map_data.json").exists():
        src = DATA_DIR / "map_data.json"
    else:
        src = DATA_DIR / "fixture_data.json"
        print("WARNING: map_data.json not found, using fixture_data.json (random values)")
    if not src.exists():
        sys.exit("data file not found: %s" % src)

    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    if tpl.count(PLACEHOLDER) != 1:
        sys.exit("template.html must contain the %s placeholder exactly once" % PLACEHOLDER)

    raw = src.read_text(encoding="utf-8")
    data = json.loads(raw)  # validate
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")

    out = OUT_PATH
    out.write_text(tpl.replace(PLACEHOLDER, text), encoding="utf-8")
    print("wrote %s (%.2f MB) from %s" % (out, out.stat().st_size / 1e6, src.name))


if __name__ == "__main__":
    main()
