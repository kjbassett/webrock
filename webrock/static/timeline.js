'use strict';

const TL = (() => {
  const MIN_NODE_W = 8;
  const NODE_GAP   = 2;
  const NODE_H     = 36;
  const LANE_H     = 64;
  const AXIS_H     = 62;
  const PAD_LEFT   = 20;
  const PAD_RIGHT  = 60;
  const WIN_DAYS   = 90;
  const MAX_OCC    = 500;

  const DAYS_SHORT   = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  const TYPE_COLORS = {
    cron:     '#4d8ec9',
    interval: '#3aaa8e',
    once:     '#e6a817',
    after:    '#7b8fa1',
  };

  let _ppm = 0.2;

  // ── Zoom helpers ─────────────────────────────────────────────────────────────

  function sliderToPpm(v) { return Math.pow(10, (v - 300) / 100); }

  function formatPpm(ppm) {
    if (ppm >= 1) {
      const v = ppm >= 10 ? Math.round(ppm) : +ppm.toFixed(1);
      return `${v} px/min`;
    }
    const mpp = 1 / ppm;
    const v = mpp >= 10 ? Math.round(mpp) : +mpp.toFixed(1);
    return `${v} min/px`;
  }

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
    const res  = await fetch('/api/schedules');
    const data = await res.json();
    const flat = [];
    for (const [pluginId, arr] of Object.entries(data))
      for (const s of arr) flat.push({ ...s, plugin_id: pluginId });
    return flat;
  }

  async function fetchAvgDurations(schedules) {
    const pluginIds = [...new Set(schedules.map(s => s.plugin_id))];
    const avgDurMs  = {};
    await Promise.all(pluginIds.map(async pid => {
      try {
        const res = await fetch(`/api/runs/${pid}`);
        if (!res.ok) return;
        const runs   = await res.json();
        const groups = {};
        for (const run of runs) {
          if (run.started_at != null && run.finished_at != null && run.schedule_id != null) {
            const dur = (run.finished_at - run.started_at) * 1000;
            if (dur > 0) {
              if (!groups[run.schedule_id]) groups[run.schedule_id] = [];
              groups[run.schedule_id].push(dur);
            }
          }
        }
        for (const [sid, durs] of Object.entries(groups))
          avgDurMs[Number(sid)] = durs.reduce((a, b) => a + b, 0) / durs.length;
      } catch (_) {}
    }));
    return avgDurMs;
  }

  // ── Graph ────────────────────────────────────────────────────────────────────

  function buildGraph(schedules) {
    const byId     = {};
    const children = {};
    const parents  = {};

    for (const s of schedules) byId[s.id] = s;

    for (const s of schedules) {
      if (s.type === 'after' && s.config?.trigger_id != null) {
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

  // ── Y lane assignment ────────────────────────────────────────────────────────

  function assignLanes(graph) {
    const lanes = {};
    let   counter = 0;

    function dfs(id, start) {
      const cs = graph.children[id] || [];
      if (!cs.length) { lanes[id] = start; return start + 1; }
      let cur = start;
      for (const c of cs) cur = dfs(c, cur);
      lanes[id] = (start + cur - 1) / 2;
      return cur;
    }

    for (const rid of graph.roots) counter = dfs(rid, counter) + 0.5;
    return lanes;
  }

  // ── Occurrence generators ────────────────────────────────────────────────────

  function parseCronField(field) {
    if (field === '*') return null;
    const vals = new Set();
    for (const part of String(field).split(',')) {
      if (part.includes('-')) {
        const [lo, hi] = part.split('-').map(Number);
        for (let i = lo; i <= hi; i++) vals.add(i);
      } else {
        vals.add(Number(part));
      }
    }
    return vals;
  }

  function cronMatches(config, ms) {
    const d  = new Date(ms);
    const fm = parseCronField(config.minutes);
    const fh = parseCronField(config.hours);
    const fd = parseCronField(config.days_of_month);
    const fM = parseCronField(config.months);
    const fw = parseCronField(config.days_of_week);
    if (fm && !fm.has(d.getMinutes())) return false;
    if (fh && !fh.has(d.getHours()))   return false;
    if (fd && !fd.has(d.getDate()))     return false;
    if (fM && !fM.has(d.getMonth() + 1)) return false;
    if (fw && !fw.has(d.getDay()))      return false;
    return true;
  }

  function getCronOccurrences(config, winStart, winEnd) {
    const occs = [];
    let t = Math.ceil(winStart / 60_000) * 60_000;
    while (t <= winEnd && occs.length < MAX_OCC) {
      if (cronMatches(config, t)) occs.push(t);
      t += 60_000;
    }
    return occs;
  }

  function getIntervalOccurrences(schedule, winStart, winEnd) {
    const sec = Number(schedule.config?.seconds ?? 0);
    if (sec <= 0) {
      const t = schedule.next_run ? new Date(schedule.next_run).getTime() : Date.now();
      return (t >= winStart && t <= winEnd) ? [t] : (t < winStart ? [winStart] : []);
    }
    const ms    = sec * 1_000;
    const base  = schedule.next_run ? new Date(schedule.next_run).getTime() : Date.now();
    const first = base + Math.ceil((winStart - base) / ms) * ms;
    const occs  = [];
    for (let t = first; t <= winEnd && occs.length < MAX_OCC; t += ms) occs.push(t);
    return occs;
  }

  function getScheduleOccurrences(schedule, winStart, winEnd) {
    if (schedule.type === 'cron')     return getCronOccurrences(schedule.config || {}, winStart, winEnd);
    if (schedule.type === 'interval') return getIntervalOccurrences(schedule, winStart, winEnd);
    if (schedule.type === 'once') {
      const t = schedule.next_run ? new Date(schedule.next_run).getTime() : null;
      return (t && t >= winStart && t <= winEnd) ? [t] : [];
    }
    return [];
  }

  function generateOccurrences(graph, winStart, winEnd) {
    const occMap  = {};
    const queue   = [...graph.roots];
    const visited = new Set();

    while (queue.length) {
      const id = queue.shift();
      if (visited.has(id)) continue;
      visited.add(id);

      const s = graph.byId[id];
      if (!s) continue;

      occMap[id] = s.type === 'after'
        ? (occMap[graph.parents[id]] || []).slice()
        : getScheduleOccurrences(s, winStart, winEnd);

      for (const c of graph.children[id] || []) queue.push(c);
    }

    return occMap;
  }

  // ── Compute pixel positions ───────────────────────────────────────────────────
  // x is the LEFT edge of each node (not centre).

  function computePositions(graph, lanes, occMap, avgDurMs, refTime, ppm) {
    const nodeW = id => Math.max(MIN_NODE_W, ((avgDurMs[id] ?? 0) / 60_000) * ppm);

    const posArr         = [];
    const parentPositions = {};
    const occupied       = new Map();   // pixel-y → [{x, x2}]

    function isOccupied(y, x, x2) {
      return (occupied.get(y) || []).some(iv => x < iv.x2 && x2 > iv.x);
    }
    function markOccupied(y, x, x2) {
      if (!occupied.has(y)) occupied.set(y, []);
      occupied.get(y).push({ x, x2 });
    }

    const OFFSETS = [0, LANE_H / 2, -LANE_H / 2, LANE_H, -LANE_H,
                     LANE_H * 1.5, -LANE_H * 1.5, LANE_H * 2, -LANE_H * 2];

    const queue   = [...graph.roots];
    const visited = new Set();

    while (queue.length) {
      const id = queue.shift();
      if (visited.has(id)) continue;
      visited.add(id);

      const s = graph.byId[id];
      if (!s) continue;

      const w     = nodeW(id);
      const baseY = AXIS_H + (lanes[id] ?? 0) * LANE_H + LANE_H / 2;
      const myPos = [];

      if (s.type !== 'after') {
        for (let i = 0; i < (occMap[id] || []).length; i++) {
          const t = occMap[id][i];
          const x = PAD_LEFT + ((t - refTime) / 60_000) * ppm;
          posArr.push({ id, occIdx: i, x, y: baseY, w, schedule: s, occTime: t });
          myPos.push({ x, y: baseY, w, occTime: t });
          markOccupied(baseY, x, x + w);
        }
      } else {
        const parentId = graph.parents[id];
        const pp       = parentPositions[parentId] || [];

        for (let i = 0; i < pp.length; i++) {
          const x       = pp[i].x + pp[i].w + NODE_GAP;
          // Expected start = when parent finishes
          const occTime = pp[i].occTime + (avgDurMs[parentId] ?? 0);

          let y = baseY;
          for (const off of OFFSETS) {
            const ty = baseY + off;
            if (!isOccupied(ty, x, x + w)) { y = ty; break; }
          }

          posArr.push({ id, occIdx: i, x, y, w, schedule: s, occTime });
          myPos.push({ x, y, w, occTime });
          markOccupied(y, x, x + w);
        }
      }

      parentPositions[id] = myPos;
      for (const c of graph.children[id] || []) queue.push(c);
    }

    return { posArr, parentPositions };
  }

  // ── Draw axis ────────────────────────────────────────────────────────────────

  function drawAxis(container, refTime, totalW, ppm) {
    const MS_PER_DAY = 86_400_000;

    let day = Math.ceil(refTime / MS_PER_DAY) * MS_PER_DAY;
    while (day <= refTime + (totalW / ppm) * 60_000 + MS_PER_DAY) {
      const x = PAD_LEFT + ((day - refTime) / 60_000) * ppm;
      if (x >= 0 && x <= totalW + 1) {
        container.appendChild(makeSvg('line', {
          x1: x, y1: 18, x2: x, y2: 9_999, class: 'tl-day-line',
        }));
        const d   = new Date(day);
        const lbl = `${DAYS_SHORT[d.getDay()]} ${d.getDate()} ${MONTHS_SHORT[d.getMonth()]}`;
        const t   = makeSvg('text', { x: x + 4, y: 14, class: 'tl-date-label' });
        t.textContent = lbl;
        container.appendChild(t);
      }
      day += MS_PER_DAY;
    }

    container.appendChild(makeSvg('line', {
      x1: 0, y1: AXIS_H - 1, x2: totalW, y2: AXIS_H - 1, class: 'tl-axis-line',
    }));

    const minPerTick = ppm >= 10 ? 15 : ppm >= 1 ? 60 : ppm >= 0.1 ? 360 : 1440;
    const msPerTick  = minPerTick * 60_000;
    let   tick       = Math.ceil(refTime / msPerTick) * msPerTick;

    while (PAD_LEFT + ((tick - refTime) / 60_000) * ppm < totalW) {
      const x = PAD_LEFT + ((tick - refTime) / 60_000) * ppm;
      if (x >= 0) {
        container.appendChild(makeSvg('line', {
          x1: x, y1: AXIS_H - 8, x2: x, y2: AXIS_H, class: 'tl-axis-tick',
        }));
        const d   = new Date(tick);
        const lbl = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
        const t   = makeSvg('text', { x, y: AXIS_H - 10, class: 'tl-axis-label', 'text-anchor': 'middle' });
        t.textContent = lbl;
        container.appendChild(t);
      }
      tick += msPerTick;
    }

    const nowX = PAD_LEFT + ((Date.now() - refTime) / 60_000) * ppm;
    if (nowX >= 0 && nowX <= totalW) {
      container.appendChild(makeSvg('line', {
        x1: nowX, y1: AXIS_H, x2: nowX, y2: 9_999, class: 'tl-now-line',
      }));
      const nl = makeSvg('text', { x: nowX + 4, y: AXIS_H + 14, class: 'tl-now-label' });
      nl.textContent = 'now';
      container.appendChild(nl);
    }
  }

  // ── Draw edges ───────────────────────────────────────────────────────────────

  function drawEdges(container, graph, parentPositions) {
    for (const [tidStr, cids] of Object.entries(graph.children)) {
      const pp = parentPositions[Number(tidStr)];
      if (!pp) continue;
      for (const cid of cids) {
        const cp = parentPositions[cid];
        if (!cp) continue;
        const n = Math.min(pp.length, cp.length);
        for (let i = 0; i < n; i++) {
          const x1 = pp[i].x + pp[i].w, y1 = pp[i].y;
          const x2 = cp[i].x,            y2 = cp[i].y;

          if (Math.abs(y1 - y2) < 2) continue;   // same lane — adjacency is self-evident

          // Step connector: drop/rise from parent right to child y, then across
          container.appendChild(makeSvg('path', {
            d: `M ${x1},${y1} V ${y2} H ${x2}`,
            class: 'tl-edge',
          }));
          container.appendChild(makeSvg('polygon', {
            points: `${x2},${y2} ${x2 - 6},${y2 - 3} ${x2 - 6},${y2 + 3}`,
            class: 'tl-arrow',
          }));
        }
      }
    }
  }

  // ── Draw nodes ───────────────────────────────────────────────────────────────

  function drawNodes(container, posArr) {
    const nodeMap = {};
    for (const pos of posArr) {
      const { id, occIdx, x, y, w, schedule: s, occTime } = pos;
      const color = TYPE_COLORS[s.type] || '#888';

      const g = makeSvg('g', {
        class: 'tl-node',
        'data-schedule-id': id,
        transform: `translate(${x},${y - NODE_H / 2})`,
      });
      g.appendChild(makeSvg('rect', {
        width: w, height: NODE_H,
        rx: Math.min(6, w / 2), ry: 6,
        fill: color, class: 'tl-node-rect' + (s.paused ? ' tl-paused' : ''),
      }));
      if (w >= 24) {
        const t = makeSvg('text', {
          x: w / 2, y: NODE_H / 2 + 4, class: 'tl-node-label', 'text-anchor': 'middle',
        });
        t.textContent = funcName(s.plugin_id);
        g.appendChild(t);
      }
      container.appendChild(g);
      nodeMap[`${id}_${occIdx}`] = { el: g, s, occTime };
    }
    return nodeMap;
  }

  // ── Node behaviour: tooltip + hover dimming + click-to-scroll ────────────────

  function attachNodeBehavior(nodeMap, avgDurMs, tip, svgEl) {
    function esc(v) { return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

    const allNodes = Object.values(nodeMap);

    // Semi-transparent overlay drawn behind nodes; pointer-events:none keeps hover working
    const overlay = makeSvg('rect', {
      x: 0, y: 0, width: '100%', height: '100%',
      class: 'tl-overlay', 'pointer-events': 'none',
    });
    svgEl.insertBefore(overlay, svgEl.firstChild);

    function setDim(activeScheduleId) {
      overlay.classList.add('tl-overlay-active');
      for (const nd of allNodes) {
        const rect = nd.el.querySelector('.tl-node-rect');
        if (!rect) continue;
        const same = Number(nd.el.getAttribute('data-schedule-id')) === activeScheduleId;
        rect.classList.toggle('tl-bright', same);
        rect.classList.toggle('tl-dimmed', !same);
      }
    }

    function clearDim() {
      overlay.classList.remove('tl-overlay-active');
      for (const nd of allNodes) {
        const rect = nd.el.querySelector('.tl-node-rect');
        if (rect) rect.classList.remove('tl-dimmed', 'tl-bright');
      }
    }

    for (const { el, s, occTime } of allNodes) {
      const dur = avgDurMs[s.id] ?? 0;

      el.addEventListener('mouseenter', () => {
        const startStr = occTime ? new Date(occTime).toLocaleString() : '—';
        const endStr   = (occTime && dur > 0) ? new Date(occTime + dur).toLocaleString() : null;

        tip.innerHTML = `
          <div class="tl-tip-row"><span class="tl-tip-key">plugin</span><span class="tl-tip-val">${esc(s.plugin_id)}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">id</span><span class="tl-tip-val">${s.id}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">type</span><span class="tl-tip-val">${esc(s.type)}</span></div>
          <div class="tl-tip-row"><span class="tl-tip-key">starts at</span><span class="tl-tip-val">${esc(startStr)}</span></div>
          ${endStr ? `<div class="tl-tip-row"><span class="tl-tip-key">ends at</span><span class="tl-tip-val">${esc(endStr)}</span></div>` : ''}
          <div class="tl-tip-row"><span class="tl-tip-key">args</span><pre class="tl-tip-pre">${esc(JSON.stringify(s.args || {}, null, 2))}</pre></div>
          <div class="tl-tip-row"><span class="tl-tip-key">config</span><pre class="tl-tip-pre">${esc(JSON.stringify(s.config || {}, null, 2))}</pre></div>
          <div class="tl-tip-row"><button class="btn btn-sm btn-outline-light tl-edit-btn" onclick="editSchedulePlugin('${esc(s.plugin_id)}')">Edit</button></div>
        `;
        tip.style.display = 'block';
        setDim(s.id);
      });

      el.addEventListener('mousemove', e => {
        // Measure actual tooltip size now that content is rendered
        const tipH = tip.offsetHeight || 260;
        const tipW = tip.offsetWidth  || 340;
        let lx = e.clientX + 14, ly = e.clientY + 14;
        if (lx + tipW > window.innerWidth)  lx = e.clientX - tipW - 14;
        if (ly + tipH > window.innerHeight) ly = e.clientY - tipH - 14;
        // Keep on-screen
        if (lx < 4) lx = 4;
        if (ly < 4) ly = 4;
        tip.style.left = lx + 'px';
        tip.style.top  = ly + 'px';
      });

      el.addEventListener('mouseleave', () => {
        tip.style.display = 'none';
        clearDim();
      });

      el.addEventListener('click', () => {
        if (typeof window.tlSelectSchedule === 'function') window.tlSelectSchedule(s.id);
      });
    }
  }

  // ── Main render ──────────────────────────────────────────────────────────────

  async function render() {
    const svgEl = document.getElementById('timeline-svg');
    const tip   = document.getElementById('timeline-tooltip');
    svgEl.innerHTML = '';
    if (tip) tip.style.display = 'none';

    let schedules;
    try {
      schedules = await fetchSchedules();
    } catch (_) {
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

    const avgDurMs = await fetchAvgDurations(schedules);

    const winStart = Date.now() - 60 * 60_000;
    const winEnd   = winStart + WIN_DAYS * 24 * 60 * 60_000;
    const refTime  = Math.floor(winStart / 3_600_000) * 3_600_000;

    const graph  = buildGraph(schedules);
    const lanes  = assignLanes(graph);
    const occMap = generateOccurrences(graph, winStart, winEnd);
    const { posArr, parentPositions } = computePositions(graph, lanes, occMap, avgDurMs, refTime, _ppm);

    if (!posArr.length) {
      svgEl.setAttribute('width', 800);
      svgEl.setAttribute('height', AXIS_H + LANE_H);
      svgEl.setAttribute('viewBox', `0 0 800 ${AXIS_H + LANE_H}`);
      drawAxis(svgEl, refTime, 800, _ppm);
      return;
    }

    const maxX   = Math.max(...posArr.map(p => p.x + p.w)) + PAD_RIGHT;
    const maxY   = Math.max(...posArr.map(p => p.y)) + NODE_H / 2 + 20;
    const totalW = Math.max(maxX, 800);
    const totalH = Math.max(maxY, AXIS_H + LANE_H);

    svgEl.setAttribute('width',   totalW);
    svgEl.setAttribute('height',  totalH);
    svgEl.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);

    drawAxis(svgEl, refTime, totalW, _ppm);
    drawEdges(svgEl, graph, parentPositions);
    const nodeMap = drawNodes(svgEl, posArr);
    if (tip) attachNodeBehavior(nodeMap, avgDurMs, tip, svgEl);
  }

  // ── Init ─────────────────────────────────────────────────────────────────────

  function init() {
    const toggleBtn = document.getElementById('timeline-toggle');
    if (!toggleBtn) return;

    const section     = document.getElementById('timeline-section');
    const zoom        = document.getElementById('timeline-zoom');
    const zoomDisplay = document.getElementById('timeline-zoom-display');
    const refreshBtn  = document.getElementById('timeline-refresh');

    zoomDisplay.textContent = formatPpm(_ppm);

    let loaded = false;

    toggleBtn.addEventListener('click', () => {
      const open = section.style.display !== 'none';
      section.style.display = open ? 'none' : 'block';
      toggleBtn.textContent = open ? 'Timeline ▾' : 'Timeline ▴';
      if (!open && !loaded) { loaded = true; render(); }
    });

    zoom.addEventListener('input', () => {
      _ppm = sliderToPpm(parseInt(zoom.value, 10));
      zoomDisplay.textContent = formatPpm(_ppm);
      if (section.style.display !== 'none') render();
    });

    refreshBtn.addEventListener('click', render);
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', TL.init);
