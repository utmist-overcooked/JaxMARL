"""Utilities for writing FSQ checkpoint rollout viewer artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


def index_to_coord(index: int, levels: list[int] | tuple[int, ...]) -> list[int]:
    coords = []
    remainder = int(index)
    for level in levels:
        coords.append(remainder % int(level))
        remainder //= int(level)
    return coords


def inventory_label(value: int) -> str:
    if value == 0:
        return "empty"
    labels = []
    if value & 1:
        labels.append("plate")
    if value & 2:
        labels.append("cooked")
    if value & 64:
        labels.append("burning")
    if value & 128:
        labels.append("burned")
    ingredient_bits = value >> 2
    if ingredient_bits:
        labels.append(f"ingredients_bits={ingredient_bits}")
    return "+".join(labels) if labels else str(value)


def state_summary(state: Any, agent_idx: int, action: int) -> dict[str, Any]:
    agent = jax.tree_util.tree_map(lambda x: np.asarray(x), state.agents)
    pot_timers = np.asarray(state.pot_cooking_timer)
    pot_mask = np.asarray(state.pot_active_mask).astype(bool)
    active_pots = pot_timers[pot_mask].astype(int).tolist()
    return {
        "agent": int(agent_idx),
        "action": int(action),
        "pos": [
            int(agent.pos.x[agent_idx]),
            int(agent.pos.y[agent_idx]),
        ],
        "dir": int(agent.dir[agent_idx]),
        "inventory": inventory_label(int(agent.inventory[agent_idx])),
        "time": int(np.asarray(state.time)),
        "active_pot_timers": active_pots,
    }


def build_viewer_data(
    *,
    layout: str,
    levels: list[int] | tuple[int, ...],
    codebook: np.ndarray,
    counts: np.ndarray,
    examples: dict[int, list[dict[str, Any]]],
    dim_counts: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    levels = [int(level) for level in levels]
    counts = np.asarray(counts, dtype=np.int64)
    data = {
        "layout": layout,
        "levels": levels,
        "total_samples": int(counts.sum()),
        "metadata": metadata or {},
        "codes": [
            {
                "index": int(i),
                "coord": index_to_coord(int(i), levels),
                "normalized": np.asarray(codebook[i]).astype(float).tolist(),
                "count": int(counts[i]),
                "examples": examples.get(int(i), []),
            }
            for i in range(len(counts))
        ],
    }
    if dim_counts is not None:
        data["dim_counts"] = np.asarray(dim_counts, dtype=np.int64).tolist()
    return data


def write_viewer_artifacts(
    out_dir: Path | str, data: dict[str, Any], *, write_index: bool = False
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    usage_path = out_dir / "fsq_usage.json"
    index_path = out_dir / "index.html"
    usage_path.write_text(json.dumps(data, indent=2))
    if write_index:
        index_path.write_text(make_viewer_html(data))
    return {
        "viewer_dir": str(out_dir),
        "usage_json": str(usage_path),
        "index_html": str(index_path) if write_index else "",
    }


def make_viewer_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FSQ Code Viewer</title>
  <style>
    body {{
      margin: 0;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #202124;
    }}
    header {{
      padding: 16px 24px;
      border-bottom: 1px solid #d8d8d2;
      background: #ffffff;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(460px, 1fr) 420px;
      gap: 20px;
      padding: 20px 24px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    a {{
      color: #194f90;
    }}
    #plot {{
      width: 100%;
      height: 680px;
      background: #ffffff;
      border: 1px solid #d8d8d2;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }}
    #plot.dragging {{
      cursor: grabbing;
    }}
    aside {{
      background: #ffffff;
      border: 1px solid #d8d8d2;
      padding: 16px;
      overflow: auto;
      max-height: 680px;
    }}
    .muted {{ color: #656565; }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }}
    .stat {{
      border: 1px solid #e0e0dc;
      padding: 8px;
      background: #fbfbf9;
    }}
    .example {{
      margin: 14px 0;
      border-top: 1px solid #e0e0dc;
      padding-top: 12px;
    }}
    .example img, .rollout img {{
      width: 100%;
      image-rendering: pixelated;
      border: 1px solid #d8d8d2;
      background: #111;
    }}
    .rollout {{
      margin: 0 0 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid #e0e0dc;
    }}
    .coords {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    svg text {{
      font-size: 11px;
      fill: #606060;
    }}
    circle {{
      cursor: pointer;
      stroke: #1d1d1b;
      stroke-width: 0.8;
    }}
    circle.selected {{
      stroke-width: 3;
      stroke: #000000;
    }}
  </style>
</head>
<body>
  <header>
    <h1>FSQ Code Viewer</h1>
    <div class="muted" id="subtitle"></div>
  </header>
  <main>
    <svg id="plot" role="img" aria-label="FSQ code grid"></svg>
    <aside>
      <div id="rollout"></div>
      <h2 id="code-title">Select a code</h2>
      <div id="details" class="muted">Hover or click a point in the grid.</div>
      <div id="examples"></div>
    </aside>
  </main>
  <script>
    const DATA = {payload};
    const svg = document.getElementById("plot");
    const details = document.getElementById("details");
    const examples = document.getElementById("examples");
    const title = document.getElementById("code-title");
    const subtitle = document.getElementById("subtitle");
    const rollout = document.getElementById("rollout");
    const selected = {{value: null}};
    const meta = DATA.metadata || {{}};

    const bits = [
      DATA.layout,
      `levels=${{DATA.levels.join("x")}}`,
      `samples=${{DATA.total_samples}}`
    ];
    if (meta.run_name) bits.push(`run=${{meta.run_name}}`);
    if (meta.checkpoint_update !== undefined) bits.push(`update=${{meta.checkpoint_update}}`);
    if (meta.recipe) bits.push(`recipe=${{meta.recipe}}`);
    subtitle.textContent = bits.join(" | ");

    if (meta.gif) {{
      rollout.className = "rollout";
      rollout.innerHTML = `
        <div class="muted">Rollout GIF</div>
        <a href="${{meta.gif}}"><img src="${{meta.gif}}" alt="checkpoint rollout gif"></a>
      `;
    }}

    const rotation = {{x: -0.55, y: 0.72}};
    const drag = {{active: false, x: 0, y: 0}};
    const cubeEdges = [
      [[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]],
      [[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]],
      [[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]],
      [[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]],
      [[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]],
      [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]],
    ];

    function edgeCoord(coord) {{
      return coord.map((value, dim) => value * Math.max(DATA.levels[dim] - 1, 0));
    }}

    function plotSize() {{
      const rect = svg.getBoundingClientRect();
      return {{
        width: rect.width || 800,
        height: rect.height || 680,
        margin: 54
      }};
    }}

    function toUnit(coord) {{
      return coord.map((value, dim) => {{
        const denom = Math.max(DATA.levels[dim] - 1, 1);
        return value / denom - 0.5;
      }});
    }}

    function rotatePoint(coord) {{
      let [x, y, z] = toUnit(coord);
      const cosY = Math.cos(rotation.y);
      const sinY = Math.sin(rotation.y);
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;

      const cosX = Math.cos(rotation.x);
      const sinX = Math.sin(rotation.x);
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      return [x1, y2, z2];
    }}

    function projectCoord(coord) {{
      const size = plotSize();
      const [x, y, z] = rotatePoint(coord);
      const perspective = 1 / (1.85 - z);
      const scale = Math.min(size.width, size.height) * 1.18;
      return {{
        x: size.width / 2 + x * scale * perspective,
        y: size.height / 2 + y * scale * perspective,
        z,
        perspective,
      }};
    }}

    function color(count, maxCount) {{
      if (count <= 0) return "#d5d5cf";
      const t = Math.sqrt(count / Math.max(maxCount, 1));
      const r = Math.round(230 - 170 * t);
      const g = Math.round(230 - 95 * t);
      const b = Math.round(230 - 200 * t);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function showCode(code) {{
      selected.value = code.index;
      document.querySelectorAll("circle").forEach(c => c.classList.toggle("selected", Number(c.dataset.index) === code.index));
      title.textContent = `Code ${{code.index}}`;
      details.innerHTML = `
        <div class="coords">coord=(${{code.coord.join(", ")}}), normalized=(${{code.normalized.map(v => v.toFixed(2)).join(", ")}})</div>
        <div class="stat-grid">
          <div class="stat"><b>count</b><br>${{code.count}}</div>
          <div class="stat"><b>examples</b><br>${{code.examples.length}}</div>
        </div>
      `;
      examples.innerHTML = "";
      for (const ex of code.examples) {{
        const div = document.createElement("div");
        div.className = "example";
        div.innerHTML = `
          <div><b>episode</b> ${{ex.episode}}, <b>step</b> ${{ex.step}}, <b>agent</b> ${{ex.agent}}</div>
          <div class="muted">pos=(${{ex.summary.pos.join(",")}}), dir=${{ex.summary.dir}}, inv=${{ex.summary.inventory}}, action=${{ex.summary.action}}, t=${{ex.summary.time}}</div>
          <img src="${{ex.image}}" alt="state example for FSQ code ${{code.index}}">
        `;
        examples.appendChild(div);
      }}
    }}

    function draw() {{
      svg.innerHTML = "";
      const maxCount = Math.max(...DATA.codes.map(c => c.count));
      for (const edge of cubeEdges) {{
        const a = projectCoord(edgeCoord(edge[0]));
        const b = projectCoord(edgeCoord(edge[1]));
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", a.x);
        line.setAttribute("y1", a.y);
        line.setAttribute("x2", b.x);
        line.setAttribute("y2", b.y);
        line.setAttribute("stroke", "#b8b8b0");
        line.setAttribute("stroke-width", "1");
        svg.appendChild(line);
      }}

      const sortedCodes = DATA.codes
        .map(code => ({{code, projected: projectCoord(code.coord)}}))
        .sort((a, b) => a.projected.z - b.projected.z);
      for (const item of sortedCodes) {{
        const code = item.code;
        const p = item.projected;
        const r = (5 + 15 * Math.sqrt(code.count / Math.max(maxCount, 1))) * (0.78 + 0.42 * p.perspective);
        const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("cx", p.x);
        c.setAttribute("cy", p.y);
        c.setAttribute("r", r);
        c.setAttribute("fill", color(code.count, maxCount));
        c.setAttribute("opacity", code.count > 0 ? "0.96" : "0.32");
        c.dataset.index = code.index;
        c.addEventListener("mouseenter", () => showCode(code));
        c.addEventListener("click", () => showCode(code));
        svg.appendChild(c);
      }}
      for (let i = 0; i < DATA.levels[0]; i++) {{
        const p = projectCoord([i, 0, 0]);
        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("x", p.x - 4);
        t.setAttribute("y", p.y + 30);
        t.textContent = i;
        svg.appendChild(t);
      }}
      if (selected.value !== null) {{
        document.querySelectorAll("circle").forEach(c => c.classList.toggle("selected", Number(c.dataset.index) === selected.value));
      }}
    }}

    svg.addEventListener("pointerdown", event => {{
      drag.active = true;
      drag.x = event.clientX;
      drag.y = event.clientY;
      svg.classList.add("dragging");
      svg.setPointerCapture(event.pointerId);
    }});
    svg.addEventListener("pointermove", event => {{
      if (!drag.active) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      drag.x = event.clientX;
      drag.y = event.clientY;
      rotation.y += dx * 0.01;
      rotation.x = Math.max(-1.45, Math.min(1.45, rotation.x - dy * 0.01));
      draw();
    }});
    svg.addEventListener("pointerup", event => {{
      drag.active = false;
      svg.classList.remove("dragging");
      svg.releasePointerCapture(event.pointerId);
    }});
    svg.addEventListener("pointercancel", () => {{
      drag.active = false;
      svg.classList.remove("dragging");
    }});
    window.addEventListener("resize", draw);

    draw();
    showCode(DATA.codes.reduce((a, b) => b.count > a.count ? b : a, DATA.codes[0]));
  </script>
</body>
</html>
"""
