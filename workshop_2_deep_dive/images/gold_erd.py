# Regenerate: python3 gold_erd.py gold_erd.svg, then render to PNG, e.g.
#   "Google Chrome" --headless=new --hide-scrollbars --window-size=1880,1040 --screenshot=gold_erd.png file://$PWD/gold_erd.svg
"""Generate the workshop 2 gold-layer ERD as SVG (no dependencies)."""
import sys
OUT = sys.argv[1]
W, H = 1880, 1040
ROW, HEAD, BOX_W = 22, 30, 300
Q_COL = {"Q1": "#1b7f5b", "Q2": "#2a6fdb", "Q3": "#c2410c", "Q4": "#8b3fd1", "Q5": "#b8860b", "Q6": "#0e7490", "dash": "#6b7280"}

# name -> (x, y, columns[(name, tag)], queries, style)
T = {
 "stats_aggregator_dimension": (20, 120, [("visual_id", "PK"), ("name", "'regen_stats', …"), ("agg_type", ""), ("statistics", ""), ("signal_expressions", "")], ["Q1", "Q2", "Q3", "Q4"], "dim"),
 "stats_aggregator_fact": (400, 120, [("container_id", "FK"), ("visual_id", "FK"), ("event_id", "FK"), ("event_instance_id", "FK"), ("channel_name", ""), ("aggregation_label", "'energy_kwh', …"), ("statistic_value", "")], ["Q1", "Q2", "Q3", "Q4"], "fact"),
 "measurement_dimension": (790, 350, [("container_id", "PK"), ("start_ts / stop_ts", ""), ("trip_start_dt, duration_s", ""), ("vehicle_key, manufacturer", ""), ("powertrain_type, scenario", ""), ("driver, tires, traffic, …", ""), ("config_hash", ""), ("_created_at", "incremental state")], ["Q1", "Q2", "Q3", "Q4", "Q6"], "dim"),
 "event_instance_fact": (1180, 120, [("container_id", "FK"), ("event_id", "FK"), ("event_instance_id", ""), ("start_ts", ""), ("end_ts", "")], ["Q3", "Q5"], "fact"),
 "event_dimension": (1560, 120, [("event_id", "PK"), ("event_name", ""), ("event_type", ""), ("event_expression", ""), ("required_channels", "")], ["Q5"], "dim"),
 "histogram_dimension": (20, 590, [("visual_id", "PK"), ("name", "'speed_distance', …"), ("agg_type", "duration / distance"), ("bins, bins_unit", ""), ("signal_expression", "")], ["Q2"], "dim"),
 "histogram_fact": (400, 590, [("container_id", "FK"), ("visual_id", "FK"), ("event_id", "FK"), ("bin_id, bin_name", ""), ("lower_bound / upper_bound", ""), ("hist_value", "s or km")], ["Q2"], "fact"),
 "calculated_channel_fact": (1180, 590, [("container_id", "FK"), ("channel_id", "FK"), ("tstart / tend", ""), ("value", "")], ["Q6"], "fact"),
 "calculated_channel_dimension": (1560, 590, [("channel_id", "PK"), ("channel_description", ""), ("channel_expression", "")], [], "dim"),
}
GREY = [("histogram2d_dimension / _fact", "visual_id · x_bin_id · y_bin_id", "dashboard (section 8)"),
        ("calculated_channel_metrics", "container_id · channel_id", "silver-shaped channel catalogue"),
        ("channel_mapping_resolution_dimension", "container_id · channel_id · channel_alias", "which physical channel each alias used")]

BW = {}
def bw(t): return BW.get(t, BOX_W)
def box_h(t): return HEAD + ROW * len(T[t][2]) + 34
def row_y(t, col):
    x, y, cols, *_ = T[t]
    i = [c for c, _ in cols].index(col)
    return y + HEAD + ROW * i + ROW / 2
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<rect width="100%" height="100%" fill="#ffffff"/>',
     f'<text x="30" y="40" font-size="24" font-weight="bold" fill="#111827">Workshop 2 gold layer: tables, keys and the section 7 queries</text>',
     f'<text x="30" y="66" font-size="14" fill="#4b5563">All tables are &lt;your schema&gt;.impulse_w2_&lt;name&gt;. Facts (orange header) join their dimension on visual_id / event_id / channel_id, and every fact joins measurement_dimension on container_id.</text>']

def line(x1, y1, x2, y2, label="", color="#9ca3af", via=None, lx=None, ly=None, width=1.8, dash=""):
    pts = [(x1, y1)] + (via or [((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2)]) + [(x2, y2)]
    d = "M " + " L ".join(f"{a:.0f} {b:.0f}" for a, b in pts)
    o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" {dash}/>')
    o.append(f'<circle cx="{x1:.0f}" cy="{y1:.0f}" r="4" fill="{color}"/><circle cx="{x2:.0f}" cy="{y2:.0f}" r="4" fill="{color}"/>')
    if label:
        lx = lx if lx is not None else (x1 + x2) / 2
        ly = ly if ly is not None else (y1 + y2) / 2
        wlab = 7.4 * len(label) + 10
        o.append(f'<rect x="{lx - wlab / 2:.0f}" y="{ly - 11:.0f}" width="{wlab:.0f}" height="20" rx="4" fill="#ffffff" stroke="{color}"/>')
        o.append(f'<text x="{lx:.0f}" y="{ly + 4:.0f}" font-size="12" text-anchor="middle" fill="#374151" font-family="Menlo, monospace">{esc(label)}</text>')

R = lambda t: T[t][0] + bw(t)
L = lambda t: T[t][0]
# --- relationships (drawn first, boxes on top)
def gap_label(x, y, text, color="#6b7280"):
    o.append(f'<text x="{x:.0f}" y="{y - 7:.0f}" font-size="11" text-anchor="middle" fill="{color}" font-family="Menlo, monospace">{esc(text)}</text>')
def short(t1, c1, t2, c2):          # fact (right) -> dimension (left neighbour) or the other way, same row band
    x1, y1 = (L(t1), row_y(t1, c1)) if T[t1][0] > T[t2][0] else (R(t1), row_y(t1, c1))
    x2, y2 = (R(t2), row_y(t2, c2)) if T[t1][0] > T[t2][0] else (L(t2), row_y(t2, c2))
    line(x1, y1, x2, y2)
    gap_label((x1 + x2) / 2, min(y1, y2), c2)
short("stats_aggregator_fact", "visual_id", "stats_aggregator_dimension", "visual_id")
short("histogram_fact", "visual_id", "histogram_dimension", "visual_id")
short("event_instance_fact", "event_id", "event_dimension", "event_id")
short("calculated_channel_fact", "channel_id", "calculated_channel_dimension", "channel_id")
# container_id trunks into measurement_dimension
ym = row_y("measurement_dimension", "container_id")
for fact, tx in (("stats_aggregator_fact", 745), ("histogram_fact", 745)):
    yf = row_y(fact, "container_id")
    line(R(fact), yf, L("measurement_dimension"), ym, via=[(tx, yf), (tx, ym)])
for fact, tx in (("event_instance_fact", 1135), ("calculated_channel_fact", 1135)):
    yf = row_y(fact, "container_id")
    line(L(fact), yf, R("measurement_dimension"), ym, via=[(tx, yf), (tx, ym)])
# Q3: stats_aggregator_fact -> event_instance_fact, routed above the boxes
yi = row_y("stats_aggregator_fact", "event_instance_id"); ye = row_y("event_instance_fact", "event_instance_id")
line(R("stats_aggregator_fact"), yi, L("event_instance_fact"), ye, color=Q_COL["Q3"], width=2.2,
     via=[(722, yi), (722, 100), (1162, 100), (1162, ye)])
lab = "Q3 · container_id, event_id, event_instance_id"
o.append(f'<rect x="{942 - 175}" y="90" width="350" height="20" rx="4" fill="#ffffff" stroke="{Q_COL["Q3"]}"/>')
o.append(f'<text x="942" y="104" font-size="12" text-anchor="middle" fill="{Q_COL["Q3"]}" font-family="Menlo, monospace">{esc(lab)}</text>')

# --- boxes
for name, (x, y, cols, qs, style) in T.items():
    h = box_h(name); BOXW = bw(name)
    head = "#f97316" if style == "fact" else "#1f2937"
    o.append(f'<rect x="{x}" y="{y}" width="{BOXW}" height="{h}" rx="8" fill="#ffffff" stroke="{head}" stroke-width="2"/>')
    o.append(f'<path d="M {x} {y + 8} a 8 8 0 0 1 8 -8 h {BOXW - 16} a 8 8 0 0 1 8 8 v {HEAD - 8} h -{BOXW} z" fill="{head}"/>')
    o.append(f'<text x="{x + 12}" y="{y + 20}" font-size="15" font-weight="bold" fill="#ffffff">{name}</text>')
    o.append(f'<text x="{x + BOXW - 10}" y="{y + 20}" font-size="11" fill="#ffffff" text-anchor="end">{"FACT" if style == "fact" else "DIM"}</text>')
    for i, (c, tag) in enumerate(cols):
        cy = y + HEAD + ROW * i
        if i % 2: o.append(f'<rect x="{x + 2}" y="{cy}" width="{BOXW - 4}" height="{ROW}" fill="#f9fafb"/>')
        key = tag if tag in ("PK", "FK") else ""
        if key:
            o.append(f'<text x="{x + 10}" y="{cy + 15}" font-size="11" font-weight="bold" fill="{"#b45309" if key == "PK" else "#6b7280"}">{key}</text>')
        o.append(f'<text x="{x + 38}" y="{cy + 15}" font-size="13" fill="#111827" font-family="Menlo, monospace">{esc(c)}</text>')
        if tag and not key:
            o.append(f'<text x="{x + BOXW - 10}" y="{cy + 15}" font-size="11" fill="#6b7280" text-anchor="end" font-style="italic">{esc(tag)}</text>')
    by = y + HEAD + ROW * len(cols) + 8
    bx = x + 10
    if not qs:
        o.append(f'<text x="{bx}" y="{by + 14}" font-size="11" fill="#6b7280" font-style="italic">not queried in section 7</text>')
    for q in qs:
        o.append(f'<rect x="{bx}" y="{by}" width="34" height="20" rx="10" fill="{Q_COL[q]}"/>')
        o.append(f'<text x="{bx + 17}" y="{by + 14}" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">{q}</text>')
        bx += 40

for tx in (745, 1135):
    o.append(f'<rect x="{tx - 41}" y="{ym + 70}" width="82" height="20" rx="4" fill="#ffffff" stroke="#9ca3af"/>')
    o.append(f'<text x="{tx}" y="{ym + 84}" font-size="11" text-anchor="middle" fill="#374151" font-family="Menlo, monospace">container_id</text>')

# --- grey strip: other gold tables
gy = 830
o.append(f'<text x="30" y="{gy - 12}" font-size="14" font-weight="bold" fill="#374151">Also in gold (not used by Q1–Q6)</text>')
for i, (n, keys, note) in enumerate(GREY):
    x = 30 + i * 610
    o.append(f'<rect x="{x}" y="{gy}" width="590" height="62" rx="8" fill="#f3f4f6" stroke="#d1d5db" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{x + 14}" y="{gy + 24}" font-size="14" font-weight="bold" fill="#374151">{n}</text>')
    o.append(f'<text x="{x + 14}" y="{gy + 46}" font-size="12" fill="#6b7280" font-family="Menlo, monospace">{esc(keys)}</text>')
    o.append(f'<text x="{x + 576}" y="{gy + 46}" font-size="12" fill="#6b7280" text-anchor="end" font-style="italic">{esc(note)}</text>')

# --- legend: queries
ly = 930
o.append(f'<text x="30" y="{ly}" font-size="14" font-weight="bold" fill="#374151">Queries (section 7)</text>')
legend = [("Q1", "regen energy per vehicle"), ("Q2", "kWh / 100 km (stats + distance histogram)"), ("Q3", "charging SOC rate per session"),
          ("Q4", "speed at HARSH_BRAKE alerts"), ("Q5", "event inventory"), ("Q6", "calculated channel: peak battery power")]
for i, (q, txt) in enumerate(legend):
    x = 30 + (i % 3) * 610; y = ly + 16 + (i // 3) * 30
    o.append(f'<rect x="{x}" y="{y}" width="34" height="20" rx="10" fill="{Q_COL[q]}"/>')
    o.append(f'<text x="{x + 17}" y="{y + 14}" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">{q}</text>')
    o.append(f'<text x="{x + 44}" y="{y + 15}" font-size="13" fill="#111827">{esc(txt)}</text>')
o.append("</svg>")
open(OUT, "w").write("\n".join(o))
print("wrote", OUT)
