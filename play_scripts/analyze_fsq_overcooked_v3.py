"""Collect FSQ communication examples and build an offline HTML viewer.

Example:
    source venv/bin/activate
    python play_scripts/analyze_fsq_overcooked_v3.py \
      --config baselines/MAPPO/config/mappo_rnn_overcooked_v3_fsq_distill.yaml \
      --actor-path /path/to/student_actor.safetensors \
      --out-dir outputs/fsq_code_viewer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.MAPPO.fsq import FSQ
from baselines.MAPPO.mappo_rnn_overcooked_v3_fsq_distill import (
    CommActorRNN,
    ScannedRNN,
)
from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from jaxmarl.wrappers.baselines import load_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="FSQ distillation config YAML.")
    parser.add_argument("--actor-path", required=True, help="Student actor safetensors.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--examples-per-code", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def index_to_coord(index: int, levels: list[int]) -> list[int]:
    coords = []
    remainder = int(index)
    for level in levels:
        coords.append(remainder % level)
        remainder //= level
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


def state_summary(state, agent_idx: int, action: int) -> dict:
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


def make_viewer_html(data: dict) -> str:
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
    #plot {{
      width: 100%;
      height: 680px;
      background: #ffffff;
      border: 1px solid #d8d8d2;
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
    .example img {{
      width: 100%;
      image-rendering: pixelated;
      border: 1px solid #d8d8d2;
      background: #111;
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
    <svg id="plot" role="img" aria-label="5 by 5 by 5 FSQ code grid"></svg>
    <aside>
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
    const selected = {{value: null}};

    subtitle.textContent = `${{DATA.layout}} | levels=${{DATA.levels.join("x")}} | samples=${{DATA.total_samples}}`;

    function project(x, y, z) {{
      const ox = 210, oy = 470;
      const sx = 78, sy = 44, sz = 58;
      return [
        ox + (x - y) * sx,
        oy + (x + y) * sy - z * sz
      ];
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
      for (const code of DATA.codes) {{
        const [x, y, z] = code.coord;
        const [px, py] = project(x, y, z);
        const r = 5 + 15 * Math.sqrt(code.count / Math.max(maxCount, 1));
        const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("cx", px);
        c.setAttribute("cy", py);
        c.setAttribute("r", r);
        c.setAttribute("fill", color(code.count, maxCount));
        c.dataset.index = code.index;
        c.addEventListener("mouseenter", () => showCode(code));
        c.addEventListener("click", () => showCode(code));
        svg.appendChild(c);
      }}
      for (let i = 0; i < DATA.levels[0]; i++) {{
        const [px, py] = project(i, 0, 0);
        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("x", px - 4);
        t.setAttribute("y", py + 38);
        t.textContent = i;
        svg.appendChild(t);
      }}
    }}

    draw();
    showCode(DATA.codes.reduce((a, b) => b.count > a.count ? b : a, DATA.codes[0]));
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    # Avoid requiring Hydra's `${now:...}` resolver for the unused hydra.run.dir.
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=False)
    config["NUM_ENVS"] = 1
    config["NUM_AGENTS"] = 2
    config["NUM_ACTORS"] = 2

    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    env = OvercookedV3(**config["ENV_KWARGS"])
    viz_env_kwargs = dict(config["ENV_KWARGS"])
    viz_env_kwargs["agent_view_size"] = None
    viz = OvercookedV3Visualizer(OvercookedV3(**viz_env_kwargs))
    actor = CommActorRNN(env.action_space(env.agents[0]).n, config=config)
    params = load_params(args.actor_path)
    fsq = FSQ(levels=tuple(config["FSQ_LEVELS"]))

    key = jax.random.PRNGKey(args.seed)
    counts = np.zeros((fsq.codebook_size,), dtype=np.int64)
    dim_counts = np.zeros((len(config["FSQ_LEVELS"]), max(config["FSQ_LEVELS"])), dtype=np.int64)
    examples = {i: [] for i in range(fsq.codebook_size)}

    for episode in range(args.episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        hstate = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        done_batch = jnp.zeros((env.num_agents,), dtype=bool)

        for step in range(args.max_steps):
            key, action_key, step_key = jax.random.split(key, 3)
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
                env.num_agents, *env.observation_space().shape
            )
            hstate, pi, comm_code, comm_index = actor.apply(
                params,
                hstate,
                (obs_batch[None, :], done_batch[None, :]),
            )
            action = pi.sample(seed=action_key).squeeze(axis=0)
            actions = {agent: int(action[i]) for i, agent in enumerate(env.agents)}

            codes_np = np.asarray(comm_code.squeeze(axis=0))
            indices_np = np.asarray(comm_index.squeeze(axis=0)).astype(int)
            rendered = None
            for agent_idx, code_index in enumerate(indices_np.tolist()):
                counts[code_index] += 1
                coord = index_to_coord(code_index, list(config["FSQ_LEVELS"]))
                for dim, value in enumerate(coord):
                    dim_counts[dim, value] += 1

                if len(examples[code_index]) < args.examples_per_code:
                    if rendered is None:
                        rendered = np.asarray(viz.render_state(state))
                    image_name = (
                        f"code_{code_index:03d}_ep{episode:03d}_"
                        f"step{step:04d}_agent{agent_idx}.png"
                    )
                    image_path = examples_dir / image_name
                    Image.fromarray(rendered).save(image_path)
                    examples[code_index].append(
                        {
                            "episode": episode,
                            "step": step,
                            "agent": agent_idx,
                            "image": f"examples/{image_name}",
                            "summary": state_summary(
                                state, agent_idx, int(action[agent_idx])
                            ),
                            "raw_code": codes_np[agent_idx].astype(float).tolist(),
                        }
                    )

            obs, state, reward, done, info = env.step(step_key, state, actions)
            done_batch = jnp.full((env.num_agents,), done["__all__"], dtype=bool)
            if bool(done["__all__"]):
                break

    codebook = np.asarray(fsq.codebook)
    data = {
        "layout": config["ENV_KWARGS"]["layout"],
        "levels": list(config["FSQ_LEVELS"]),
        "total_samples": int(counts.sum()),
        "dim_counts": dim_counts.tolist(),
        "codes": [
            {
                "index": i,
                "coord": index_to_coord(i, list(config["FSQ_LEVELS"])),
                "normalized": codebook[i].astype(float).tolist(),
                "count": int(counts[i]),
                "examples": examples[i],
            }
            for i in range(fsq.codebook_size)
        ],
    }

    (out_dir / "fsq_usage.json").write_text(json.dumps(data, indent=2))
    (out_dir / "index.html").write_text(make_viewer_html(data))
    print(f"Wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
