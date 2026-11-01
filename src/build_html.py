"""Embed map_data.json (or fixture_data.json) into template.html -> oslo_demand_rebalancing_map.html."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
PLACEHOLDER = "__DATA__"


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        src = Path(arg)
    elif (HERE / "map_data.json").exists():
        src = HERE / "map_data.json"
    else:
        src = HERE / "fixture_data.json"
        print("WARNING: map_data.json not found, using fixture_data.json (random values)")
    if not src.exists():
        sys.exit("data file not found: %s" % src)

    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    if tpl.count(PLACEHOLDER) != 1:
        sys.exit("template.html must contain the %s placeholder exactly once" % PLACEHOLDER)

    raw = src.read_text(encoding="utf-8")
    data = json.loads(raw)  # validate
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")

    out = HERE / "oslo_demand_rebalancing_map.html"
    out.write_text(tpl.replace(PLACEHOLDER, text), encoding="utf-8")
    print("wrote %s (%.2f MB) from %s" % (out, out.stat().st_size / 1e6, src.name))


if __name__ == "__main__":
    main()
