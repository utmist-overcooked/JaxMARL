const runSelect = document.getElementById("run");
const checkpointSelect = document.getElementById("checkpoint");
const recipeSelect = document.getElementById("recipe");
const meta = document.getElementById("meta");
const svg = document.getElementById("plot");
const rollout = document.getElementById("rollout");
const title = document.getElementById("code-title");
const details = document.getElementById("details");
const examples = document.getElementById("examples");

let artifacts = [];
let currentArtifact = null;
let data = null;

const selected = { value: null };
const rotation = { x: -0.55, y: 0.72 };
const zoom = { value: 1.0 };
const drag = { active: false, x: 0, y: 0, moved: false };
const cubeEdges = [
  [[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]],
  [[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]],
  [[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]],
  [[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]],
  [[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]],
  [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]],
];

function unique(items, key) {
  const map = new Map();
  for (const item of items) {
    map.set(key(item), item);
  }
  return [...map.values()];
}

function checkpointLabel(value) {
  return value === "unknown" ? "unknown" : `update ${value}`;
}

function recipeLabel(item) {
  if (item.recipe_index === null) {
    return item.recipe;
  }
  return `recipe ${item.recipe_index}: ${item.recipe}`;
}

function assetUrl(path) {
  return `${currentArtifact.asset_url}${path}`;
}

function setOptions(select, items, label, value) {
  const previous = select.value;
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = value(item);
    option.textContent = label(item);
    select.appendChild(option);
  }
  if ([...select.options].some(option => option.value === previous)) {
    select.value = previous;
  }
}

function selectedRunArtifacts() {
  return artifacts.filter(item => item.run === runSelect.value);
}

function selectedCheckpointArtifacts() {
  return selectedRunArtifacts().filter(
    item => String(item.checkpoint) === checkpointSelect.value,
  );
}

function renderSelectors() {
  const runs = unique(artifacts, item => item.run);
  setOptions(runSelect, runs, item => item.run, item => item.run);

  const checkpoints = unique(selectedRunArtifacts(), item => String(item.checkpoint));
  setOptions(
    checkpointSelect,
    checkpoints,
    item => checkpointLabel(item.checkpoint),
    item => String(item.checkpoint),
  );

  const recipes = selectedCheckpointArtifacts();
  setOptions(recipeSelect, recipes, recipeLabel, item => item.id);
  loadSelectedArtifact();
}

async function loadSelectedArtifact() {
  const item = artifacts.find(artifact => artifact.id === recipeSelect.value);
  if (!item) {
    currentArtifact = null;
    data = null;
    meta.textContent = "No FSQ artifact selected.";
    svg.innerHTML = "";
    rollout.innerHTML = "";
    title.textContent = "Select a code";
    details.textContent = "No FSQ artifact selected.";
    examples.innerHTML = "";
    return;
  }

  currentArtifact = item;
  meta.textContent = [
    item.layout,
    `levels=${item.levels.join("x")}`,
    `samples=${item.total_samples}`,
    `nonzero=${item.nonzero_codes}`,
    item.path,
  ].join(" | ");

  const response = await fetch(item.usage_url);
  data = await response.json();
  selected.value = null;
  renderRollout();
  draw();
  clearDetails();
}

function renderRollout() {
  rollout.innerHTML = "";
  rollout.className = "";
  const gif = data.metadata && data.metadata.gif;
  if (!gif) {
    return;
  }
  rollout.className = "rollout";
  rollout.innerHTML = `
    <div class="muted">Rollout GIF</div>
    <a href="${assetUrl(gif)}"><img src="${assetUrl(gif)}" alt="checkpoint rollout gif"></a>
  `;
}

function plotSize() {
  const rect = svg.getBoundingClientRect();
  return {
    width: rect.width || 800,
    height: rect.height || 680,
  };
}

function edgeCoord(coord) {
  return coord.map((value, dim) => value * Math.max(data.levels[dim] - 1, 0));
}

function toUnit(coord) {
  return coord.map((value, dim) => {
    const denom = Math.max(data.levels[dim] - 1, 1);
    return value / denom - 0.5;
  });
}

function rotatePoint(coord) {
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
}

function projectCoord(coord) {
  const size = plotSize();
  const [x, y, z] = rotatePoint(coord);
  const perspective = 1 / (1.85 - z);
  const scale = Math.min(size.width, size.height) * 1.18 * zoom.value;
  return {
    x: size.width / 2 + x * scale * perspective,
    y: size.height / 2 + y * scale * perspective,
    z,
    perspective,
  };
}

function color(count, maxCount) {
  if (count <= 0) {
    return "#d5d5cf";
  }
  const t = Math.sqrt(count / Math.max(maxCount, 1));
  const r = Math.round(230 - 170 * t);
  const g = Math.round(230 - 95 * t);
  const b = Math.round(230 - 200 * t);
  return `rgb(${r},${g},${b})`;
}

function updateSelectedMarker() {
  document.querySelectorAll("circle").forEach(circle => {
    circle.classList.toggle(
      "selected",
      Number(circle.dataset.index) === selected.value,
    );
  });
}

function clearDetails() {
  title.textContent = "Select a code";
  details.textContent = data
    ? "Hover or click a point in the grid."
    : "Select a run, checkpoint, and recipe.";
  details.classList.add("muted");
  examples.innerHTML = "";
}

function renderCode(code) {
  details.classList.remove("muted");
  title.textContent = `Code ${code.index}`;
  details.innerHTML = `
    <div class="coords">coord=(${code.coord.join(", ")}), normalized=(${code.normalized.map(v => v.toFixed(2)).join(", ")})</div>
    <div class="stat-grid">
      <div class="stat"><b>count</b><br>${code.count}</div>
      <div class="stat"><b>examples</b><br>${code.examples.length}</div>
    </div>
  `;
  examples.innerHTML = "";
  for (const ex of code.examples) {
    const div = document.createElement("div");
    div.className = "example";
    div.innerHTML = `
      <div><b>episode</b> ${ex.episode}, <b>step</b> ${ex.step}, <b>agent</b> ${ex.agent}</div>
      <div class="muted">pos=(${ex.summary.pos.join(",")}), dir=${ex.summary.dir}, inv=${ex.summary.inventory}, action=${ex.summary.action}, t=${ex.summary.time}</div>
      <img src="${assetUrl(ex.image)}" alt="state example for FSQ code ${code.index}">
    `;
    examples.appendChild(div);
  }
}

function selectCode(code) {
  selected.value = code.index;
  updateSelectedMarker();
  renderCode(code);
}

function previewCode(code) {
  renderCode(code);
}

function restoreSelectedCode() {
  if (selected.value === null) {
    clearDetails();
    return;
  }
  const code = data && data.codes.find(item => item.index === selected.value);
  if (code) {
    renderCode(code);
  }
}

function clearSelection() {
  selected.value = null;
  updateSelectedMarker();
  clearDetails();
}

function draw() {
  svg.innerHTML = "";
  if (!data) {
    return;
  }
  const maxCount = Math.max(...data.codes.map(code => code.count));
  for (const edge of cubeEdges) {
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
  }

  const sortedCodes = data.codes
    .map(code => ({ code, projected: projectCoord(code.coord) }))
    .sort((a, b) => a.projected.z - b.projected.z);
  for (const item of sortedCodes) {
    const code = item.code;
    const p = item.projected;
    const radius =
      (5 + 15 * Math.sqrt(code.count / Math.max(maxCount, 1))) *
      (0.78 + 0.42 * p.perspective);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", radius);
    circle.setAttribute("fill", color(code.count, maxCount));
    circle.setAttribute("opacity", code.count > 0 ? "0.96" : "0.32");
    circle.dataset.index = code.index;
    circle.addEventListener("pointerdown", event => {
      event.stopPropagation();
      drag.moved = false;
    });
    circle.addEventListener("mouseenter", () => previewCode(code));
    circle.addEventListener("mouseleave", restoreSelectedCode);
    circle.addEventListener("click", event => {
      event.stopPropagation();
      if (drag.moved) {
        return;
      }
      selectCode(code);
    });
    svg.appendChild(circle);
  }
  for (let i = 0; i < data.levels[0]; i++) {
    const p = projectCoord([i, 0, 0]);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", p.x - 4);
    text.setAttribute("y", p.y + 30);
    text.textContent = i;
    svg.appendChild(text);
  }
  updateSelectedMarker();
}

runSelect.addEventListener("change", renderSelectors);
checkpointSelect.addEventListener("change", renderSelectors);
recipeSelect.addEventListener("change", loadSelectedArtifact);
svg.addEventListener("pointerdown", event => {
  drag.active = true;
  drag.x = event.clientX;
  drag.y = event.clientY;
  drag.moved = false;
  svg.classList.add("dragging");
  svg.setPointerCapture(event.pointerId);
});
svg.addEventListener("pointermove", event => {
  if (!drag.active) {
    return;
  }
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
    drag.moved = true;
  }
  drag.x = event.clientX;
  drag.y = event.clientY;
  rotation.y += dx * 0.01;
  rotation.x = Math.max(-1.45, Math.min(1.45, rotation.x - dy * 0.01));
  draw();
});
svg.addEventListener("pointerup", event => {
  drag.active = false;
  svg.classList.remove("dragging");
  svg.releasePointerCapture(event.pointerId);
});
svg.addEventListener("pointercancel", () => {
  drag.active = false;
  svg.classList.remove("dragging");
});
svg.addEventListener("click", event => {
  if (drag.moved) {
    return;
  }
  if (event.target.tagName.toLowerCase() !== "circle") {
    clearSelection();
  }
});
svg.addEventListener(
  "wheel",
  event => {
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.001);
    zoom.value = Math.max(0.35, Math.min(3.0, zoom.value * factor));
    draw();
  },
  { passive: false },
);
window.addEventListener("resize", draw);

fetch("/api/artifacts")
  .then(response => response.json())
  .then(payload => {
    artifacts = payload.artifacts;
    meta.textContent = `Found ${artifacts.length} FSQ artifacts.`;
    renderSelectors();
  });
