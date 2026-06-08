'use strict';

const TL = (() => {
  const NODE_W = 100;
  const NODE_H = 36;
  const LANE_H = 64;
  const CHAIN_OFFSET_PX = 160;
  const AXIS_H = 44;
  const PAD_LEFT = 20;
  const PAD_RIGHT = 60;
  const TIP_W = 340;
  const TIP_H = 220;

  const TYPE_COLORS = {
    cron: '#4d8ec9',
    interval: '#3aaa8e',
    once: '#e6a817',
    after: '#7b8fa1',
  };

  let _ppm = 30;

  // ── SVG helper ───────────────────────────────────────────────────────────────

  function makeSvg(tag, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  function funcName(pluginId) {
    const s = String(pluginId).split('.').pop();
    return s.length > 14 ? s.slice(0, 13) + '…' : s;
  }

  // ── Data fetch ───────────────────────────────────────────────────────────────

  async function fetchSchedules() {
    const res = await fetch('/api/schedules');
    const data = await res.json();
    const flat = [];
    for (const [pluginId, arr] of Object.entries(data)) {
      for (const s of arr) flat.push({ ...s, plugin_id: pluginId });
    }
    return flat;
  }

  // ── Graph ────────────────────────────────────────────────────────────────────

  function buildGraph(schedules) {
    const byId = {};
    const children = {};
    const parents = {};

    for (const s of schedules) byId[s.id] = s;

    for (const s of schedules) {
      if (s.type === 'after' && s.config != null && s.config.trigger_id != null) {
        const tid = Number(s.config.trigger_id);
        if (!children[tid]) children[tid] = [];
        children[tid].push(s.id);
        parents[s.id] = tid;
      }
    }

    const childSet = new Set(Object.keys(parents).map(Number));
    const roots = schedules
      .filter(s => !childSet.has(s.id))
      .sort((a, b) => {
        const ta = a.next_run ? new Date(a.next_run).getTime() : Date.now();
        const tb = b.next_run ? new Date(b.next_run).getTime() : Date.now();
        return ta - tb;
      })
      .map(s => s.id);

    return { byId, children, parents, roots };
  }

  // ── Y lane assignment (DFS, minimizes crossings for trees) ───────────────────

  function assignLanes(graph) {
    const lanes = {};
    let counter = 0;

    function dfs(id, startLane) {
      const cs = graph.children[id] || [];
      if (!cs.length) {
        lanes[id] = startLane;
        return startLane + 1;
      }
      let cur = startLane;
      for (const c of cs) cur = dfs(c, cur);
      lanes[id] = (startLane + cur - 1) / 2;
      return cur;
    }

    for (const rid of graph.roots) {
      counter = dfs(rid, counter) + 0.5;
    }
    return lanes;
  }

  // ── X + Y pixel positions ────────────────────────────────────────────────────

  function computePositions(graph, lanes, ppm) {
    let refTime = Infinity;
    for (const s of Object.values(graph.byId)) {
      if (s.type !== 'after' && s.next_run) {
        const t = new Date(s.next_run).getTime();
        if (t < refTime) refTime = t;
      }
    }
    if (!isFinite(refTime)) refTime = Date.now();
    refTime = Math.floor(refTime / 3_600_000) * 3_600_000;

    const positions = {};
    const queue = [...graph.roots];
    const visited = new Set();

    while (queue.length) {
      const id = queue.shift();
      if (visited.has(id)) continue;
      visited.add(id);

      const s = graph.byId[id];
      if (!s) continue;

      let x;
      if (s.type === 'after') {
        const parentX = positions[graph.parents[id]]?.x;
        x = (parentX != null ? parentX : PAD_LEFT) + CHAIN_OFFSET_PX;
      } else {
        const t = s.next_run ? new Date(s.next_run).getTime() : Date.now();
        x = PAD_LEFT + ((t - refTime) / 60_000) * ppm;
      }

      positions[id] = { x, y: AXIS_H + (lanes[id] ?? 0) * LANE_H + LANE_H / 2 };

      for (const c of graph.children[id] || []) queue.push(c);
    }

    return { positions, refTime };
  }

  // ── Draw time axis ───────────────────────────────────────────────────────────

  function drawAxis(container, refTime, totalW, ppm) {
    container.appendChild(makeSvg('line', {
      x1: 0, y1: AXIS_H - 1, x2: totalW, y2: AXIS_H - 1, class: 'tl-axis-line',
    }));

    const minPerTick = ppm >= 10 ? 15 : ppm >= 3 ? 60 : 120;
    const msPerTick = minPerTick * 60_000;
    let tick = Math.ceil(refTime / msPerTick) * msPerTick;

    while (PAD_LEFT + ((tick - refTime) / 60_000) * ppm < totalW) {
      const x = PAD_LEFT + ((tick - refTime) / 60_000) * ppm;
      container.appendChild(makeSvg('line', { x1: x, y1: AXIS_H - 8, x2: x, y2: AXIS_H, class: 'tl-axis-tick' }));
      const d = new Date(tick);
      const lbl = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
      const t = makeSvg('text', { x, y: AXIS_H - 12, class: 'tl-axis-label', 'text-anchor': 'middle' });
      t.textContent = lbl;
      container.appendChild(t);
      tick += msPerTick;
    }

    const nowX = PAD_LEFT + ((Date.now() - refTime) / 60_000) * ppm;
    if (nowX >= 0 && nowX <= totalW) {
      container.appendChild(makeSvg('line', { x1: nowX, y1: AXIS_H, x2: nowX, y2: 9_999, class: 'tl-now-line' }));
      const nl = makeSvg('text', { x: nowX + 4, y: AXIS_H + 14, class: 'tl-now-label' });
      nl.textContent = 'now';
      container.appendChild(nl);
    }
  }

  // ── Draw edges ───────────────────────────────────────────────────────────────

  function drawEdges(container, graph, positions) {
    for (const [tidStr, cids] of Object.entries(graph.children)) {
      const src = positions[Number(tidStr)];
      if (!src) continue;
      for (const cid of cids) {
        const dst = positions[cid];
        if (!dst) continue;
        const x1 = src.x + NODE_W / 2, y1 = src.y;
        const x2 = dst.x - NODE_W / 2, y2 = dst.y;
        const mx = (x1 + x2) / 2;
        container.appendChild(makeSvg('path', {
          d: `M ${x1},${y1} C ${mx},${y1} ${mx},${y2} ${x2},${y2}`,
          class: 'tl-edge',
        }));
        container.appendChild(makeSvg('polygon', {
          points: `${x2},${y2} ${x2 - 7},${y2 - 4} ${x2 - 7},${y2 + 4}`,
          class: 'tl-arrow',
        }));
      }
    }
  }

  // ── Draw nodes ───────────────────────────────────────────────────────────────

  function drawNodes(container, graph, positions) {
    const nodeMap = {};
    for (const [idStr, s] of Object.entries(graph.byId)) {
      const pos = positions[Number(idStr)];
      if (!pos) continue;
      const color = TYPE_COLORS[s.type] || '#888';

      const g = makeSvg('g', {
        class: 'tl-node',
        transform: `translate(${pos.x - NODE_W / 2},${pos.y - NODE_H / 2})`,
      });
      g.appendChild(makeSvg('rect', { width: NODE_W, height: NODE_H, rx: 6, ry: 6, fill: color, class: 'tl-node-rect' }));
      const t = makeSvg('text', { x: NODE_W / 2, y: NODE_H / 2 + 4, class: 'tl-node-label', 'text-anchor': 'middle' });
      t.textContent = funcName(s.plugin_id);
      g.appendChild(t);
      container.appendChild(g);
      nodeMap[idStr] = { el: g, s };
    }
    return nodeMap;
  }

  // ── Tooltip ──────────────────────────────────────────────────────────────────

  function attachTooltips(nodeMap, tip) {
    function esc(v) { return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

    for (const { el, s } of Object.values(nodeMap)) {
      el.addEventListener('mouseenter', () => {
        tip.innerHTML = `
          <div class="tl-tip-row"><span class="tl-tip-key">plugin</span><span class="tl-tip-val">${esc(s.plugin_id)}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">id</span><span class="tl-tip-val">${s.id}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">type</span><span class="tl-tip-val">${esc(s.type)}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">next run</span><span class="tl-tip-val">${esc(s.next_run || '—')}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">args</span><pre class="tl-tip-pre">${esc(JSON.stringify(s.args || {}, null, 2))}</pre></div>
          <div class="tl-tip-row"><span class="tl-tip-key">config</span><pre class="tl-tip-pre">${esc(JSON.stringify(s.config || {}, null, 2))}</pre></div>
        `;
        tip.style.display = 'block';
      });
      el.addEventListener('mousemove', e => {
        let lx = e.clientX + 14;
        let ly = e.clientY + 14;
        if (lx + TIP_W > window.innerWidth) lx = e.clientX - TIP_W - 14;
        if (ly + TIP_H > window.innerHeight) ly = e.clientY - TIP_H - 14;
        tip.style.left = lx + 'px';
        tip.style.top = ly + 'px';
      });
      el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    }
  }

  // ── Main render ──────────────────────────────────────────────────────────────

  async function render() {
    const svgEl = document.getElementById('timeline-svg');
    const tip = document.getElementById('timeline-tooltip');
    svgEl.innerHTML = '';
    if (tip) tip.style.display = 'none';

    let schedules;
    try {
      schedules = await fetchSchedules();
    } catch (e) {
      svgEl.setAttribute('width', 400);
      svgEl.setAttribute('height', 60);
      const t = makeSvg('text', { x: 16, y: 36, class: 'tl-axis-label' });
      t.textContent = 'Failed to load schedules.';
      svgEl.appendChild(t);
      return;
    }

    if (!schedules.length) {
      svgEl.setAttribute('width', 400);
      svgEl.setAttribute('height', 60);
      const t = makeSvg('text', { x: 16, y: 36, class: 'tl-axis-label' });
      t.textContent = 'No active schedules.';
      svgEl.appendChild(t);
      return;
    }

    const graph = buildGraph(schedules);
    const lanes = assignLanes(graph);
    const { positions, refTime } = computePositions(graph, lanes, _ppm);

    const posArr = Object.values(positions);
    if (!posArr.length) return;

    const maxX = Math.max(...posArr.map(p => p.x)) + NODE_W / 2 + PAD_RIGHT;
    const maxY = Math.max(...posArr.map(p => p.y)) + NODE_H / 2 + 20;
    const totalW = Math.max(maxX, 800);
    const totalH = Math.max(maxY, AXIS_H + LANE_H);

    svgEl.setAttribute('width', totalW);
    svgEl.setAttribute('height', totalH);
    svgEl.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);

    drawAxis(svgEl, refTime, totalW, _ppm);
    drawEdges(svgEl, graph, positions);
    const nodeMap = drawNodes(svgEl, graph, positions);
    if (tip) attachTooltips(nodeMap, tip);
  }

  // ── Init ─────────────────────────────────────────────────────────────────────

  function init() {
    const toggleBtn = document.getElementById('timeline-toggle');
    if (!toggleBtn) return;

    const section = document.getElementById('timeline-section');
    const zoom = document.getElementById('timeline-zoom');
    const zoomDisplay = document.getElementById('timeline-zoom-display');
    const refreshBtn = document.getElementById('timeline-refresh');

    let loaded = false;

    toggleBtn.addEventListener('click', () => {
      const open = section.style.display !== 'none';
      section.style.display = open ? 'none' : 'block';
      toggleBtn.textContent = open ? 'Timeline ▾' : 'Timeline ▴';
      if (!open && !loaded) { loaded = true; render(); }
    });

    zoom.addEventListener('input', () => {
      _ppm = parseInt(zoom.value, 10);
      zoomDisplay.textContent = _ppm;
      if (section.style.display !== 'none') render();
    });

    refreshBtn.addEventListener('click', render);
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', TL.init);
