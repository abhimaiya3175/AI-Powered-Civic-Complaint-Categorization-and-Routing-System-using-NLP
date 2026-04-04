# pipeline/map_export.py
"""
Generate a self-contained Leaflet.js HTML map from pipeline results.
Each complaint with detected lat/lon becomes a marker.
Complaints without coordinates show in a sidebar table.
"""

import json
import pathlib
from datetime import datetime


# Default centre for Bengaluru (used when no geocoordinates are available)
DEFAULT_LAT = 12.9716
DEFAULT_LON = 77.5946

CATEGORY_COLORS = {
    "Road":         "#E85D24",
    "Water":        "#3B8BD4",
    "Garbage":      "#639922",
    "Drainage":     "#533AB7",
    "Electricity":  "#BA7517",
    "Building":     "#993556",
    "Other":        "#5F5E5A",
}

_COLOR_DEFAULT = "#888780"


def _color_for(category: str) -> str:
    for key, color in CATEGORY_COLORS.items():
        if key.lower() in category.lower():
            return color
    return _COLOR_DEFAULT


def generate_map(results: list[dict], out_dir: str = "outputs") -> str:
    """
    Write a self-contained HTML Leaflet map.

    Parameters
    ----------
    results : list[dict]
        Each dict must have keys:
          sample_id, kannada_text, english_text,
          category, confidence, location (dict with display, lat, lon)

    Returns
    -------
    Path to the generated HTML file.
    """
    base = pathlib.Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base / f"map_{ts}.html"

    markers_js = _build_markers_js(results)
    table_rows = _build_table_rows(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BBMP Civic Complaints Map</title>
<link rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; display: flex; height: 100vh; }}
  #map {{ flex: 1; }}
  #sidebar {{
    width: 360px; overflow-y: auto;
    background: #fafafa; border-left: 1px solid #e0e0e0;
    padding: 16px;
  }}
  h1 {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #2c2c2a; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: #f0efe8; text-align: left; padding: 6px 8px; color: #3d3d3a; }}
  td {{ padding: 6px 8px; border-top: 1px solid #e8e8e0; color: #444; vertical-align: top; }}
  .badge {{
    display: inline-block; padding: 2px 6px;
    border-radius: 4px; font-size: 11px; font-weight: 500;
    background: #eeedfe; color: #3c3489;
  }}
  .conf {{ color: #888; font-size: 11px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="sidebar">
  <h1>🗺 BBMP Complaints ({len(results)} samples)</h1>
  <table>
    <thead><tr><th>#</th><th>Category</th><th>Location</th><th>Confidence</th></tr></thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{DEFAULT_LAT}, {DEFAULT_LON}], 12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors'
}}).addTo(map);

function colorIcon(color) {{
  return L.divIcon({{
    className: '',
    html: `<div style="
      width:14px;height:14px;border-radius:50%;
      background:${{color}};border:2px solid #fff;
      box-shadow:0 1px 4px rgba(0,0,0,.4)"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  }});
}}

{markers_js}
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"🗺  Leaflet map saved → {out_path}")
    return str(out_path)


def _build_markers_js(results: list[dict]) -> str:
    lines = []
    for r in results:
        loc = r.get("location", {})
        lat = loc.get("lat")
        lon = loc.get("lon")
        if lat is None or lon is None:
            continue

        color    = _color_for(r.get("category", ""))
        category = _js_str(r.get("category", ""))
        eng      = _js_str(r.get("english_text", "")[:120])
        conf     = r.get("confidence", 0)
        sid      = r.get("sample_id", "")
        display  = _js_str(loc.get("display", ""))

        popup = (
            f"<b>#{sid} – {category}</b><br>"
            f"<em>{display}</em><br>"
            f"Confidence: {conf:.0%}<br>"
            f"{eng}…"
        )
        lines.append(
            f'L.marker([{lat},{lon}], {{icon: colorIcon("{color}")}}).addTo(map)'
            f'.bindPopup({json.dumps(popup)});'
        )
    return "\n".join(lines)


def _build_table_rows(results: list[dict]) -> str:
    rows = []
    for r in results:
        color   = _color_for(r.get("category", ""))
        cat     = r.get("category", "Unknown")
        display = r.get("location", {}).get("display", "Not detected")
        conf    = r.get("confidence", 0)
        sid     = r.get("sample_id", "")
        rows.append(
            f'<tr>'
            f'<td>{sid}</td>'
            f'<td><span class="badge" style="background:{color}20;color:{color}">{cat}</span></td>'
            f'<td>{display}</td>'
            f'<td class="conf">{conf:.0%}</td>'
            f'</tr>'
        )
    return "\n      ".join(rows)


def _js_str(s: str) -> str:
    return s.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")