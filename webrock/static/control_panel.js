// ---- Legacy helper kept for plugin accordion stop button ----
function stop(plugin) {
    fetch(`/api/plugins/${plugin}/stop`, { method: 'POST' })
        .then(r => r.json())
        .then(r => { console.log(r); checkStatusAndUpdateLight(plugin); })
        .catch(e => console.error('Error:', e));
}

function checkStatusAndUpdateLight(plugin) {
    fetch(`/api/plugins/${plugin}/status`)
        .then(r => r.json())
        .then(r => {
            const light = document.getElementById(plugin + '_statusLight');
            if (!light) return;
            const running = r.task_status === 'running';
            light.classList.toggle('blue', running);
            light.classList.toggle('red', !running);
            const stamp = document.getElementById(`${plugin}_lastRunTimestamp`);
            if (stamp) stamp.textContent = `Last Run Time: ${r.lastRunTimestamp ?? ''}`;
        })
        .catch(e => console.error('Error:', e));
}

// ---- Schedule type form helpers ----
function updateScheduleFields(pluginId) {
    const type = document.getElementById(`schedule-type-${pluginId}`).value;
    const sections = {
        interval: `schedule-interval-wrap-${pluginId}`,
        cron: `schedule-cron-wrap-${pluginId}`,
        once: `schedule-once-wrap-${pluginId}`,
        after: `schedule-after-wrap-${pluginId}`
    };
    for (const [key, id] of Object.entries(sections)) {
        document.getElementById(id).style.display = key === type ? "block" : "none";
    }
}

function formatJob(job) {
    const c = job.config;
    if (job.type === "interval") return `Every ${c.seconds}s`;
    if (job.type === "cron") return `Recurring: ${c.minutes}m ${c.hours}h`;
    if (job.type === "once") return `Specific Time @ ${c.timestamp === "now" ? "now" : c.timestamp}`;
    if (job.type === "after") return `After schedule #${c.trigger_id}`;
    return "Unknown";
}

function editSchedulePlugin(pluginId) {
    const pluginEl = document.getElementById(`plugin-${pluginId}`);
    if (!pluginEl) return;
    pluginEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    const collapseEl = document.getElementById(`collapse-${pluginId}`);
    if (collapseEl && !collapseEl.classList.contains('show') && typeof bootstrap !== 'undefined')
        new bootstrap.Collapse(collapseEl, { toggle: true });
}

// ---- Plugin-accordion schedule rows ----
function renderScheduleRow(job, pluginId) {
    const tr = document.createElement("tr");
    [job.id, formatJob(job), JSON.stringify(job.args), job.next_run ?? "n/a", job.last_run ?? "never"].forEach(val => {
        const td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
    });
    const btnTd = document.createElement("td");
    const editBtn = document.createElement("button");
    editBtn.className = "btn btn-sm btn-secondary me-1";
    editBtn.textContent = "Edit";
    editBtn.onclick = () => openEditForm(job, pluginId);
    const removeBtn = document.createElement("button");
    removeBtn.className = "btn btn-sm btn-danger";
    removeBtn.textContent = "Remove";
    removeBtn.onclick = async () => {
        await fetch(`/api/schedules/${job.id}`, { method: "DELETE" });
        tr.remove();
    };
    btnTd.append(editBtn, removeBtn);
    tr.appendChild(btnTd);
    return tr;
}

// ---- Run-history rows ----
function renderRunRow(run) {
    const fmt = ts => ts ? new Date(ts * 1000).toLocaleString() : "—";
    const tr = document.createElement("tr");
    [run.id, run.schedule_id, fmt(run.started_at), fmt(run.finished_at), run.status].forEach(val => {
        const td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
    });
    const td = document.createElement("td");
    const text = String(run.error ?? (run.result !== null ? run.result : ""));
    if (run.error && text.length > 30) {
        const textSpan = document.createElement("span");
        textSpan.textContent = text.slice(0, 30) + "...";
        const chevron = document.createElement("button");
        chevron.textContent = "▶";
        chevron.className = "run-error-chevron";
        let expanded = false;
        chevron.addEventListener("click", () => {
            expanded = !expanded;
            textSpan.textContent = expanded ? text : text.slice(0, 30) + "...";
            textSpan.style.whiteSpace = expanded ? "pre-wrap" : "";
            chevron.textContent = expanded ? "▼" : "▶";
        });
        td.append(textSpan, chevron);
    } else {
        td.textContent = text;
    }
    tr.appendChild(td);
    return tr;
}

// ---- Edit form ----
function openEditForm(job, pluginId) {
    const form = document.getElementById(`plugin-form-${pluginId}`);
    form.dataset.editId = job.id;
    const typeSelect = document.getElementById(`schedule-type-${pluginId}`);
    typeSelect.value = job.type;
    updateScheduleFields(pluginId);
    const setVal = (name, val) => {
        const el = form.querySelector(`[name="${name}"]`);
        if (el) el.value = val ?? "";
    };
    const c = job.config || {};
    if (job.type === "once")          setVal("_timestamp",          c.timestamp === "now" ? "" : (c.timestamp ?? ""));
    else if (job.type === "interval") setVal("_seconds",            c.seconds ?? 60);
    else if (job.type === "after")    setVal("_trigger_schedule_id",c.trigger_id ?? "");
    else if (job.type === "cron") {
        setVal("_minutes",       c.minutes       ?? "*");
        setVal("_hours",         c.hours         ?? "*");
        setVal("_days_of_week",  c.days_of_week  ?? "*");
        setVal("_days_of_month", c.days_of_month ?? "*");
        setVal("_months",        c.months        ?? "*");
    }
    for (const [key, val] of Object.entries(job.args || {})) setVal(key, val);
    document.getElementById(`schedule-submit-${pluginId}`).textContent = "Update Schedule";
    document.getElementById(`schedule-cancel-${pluginId}`).style.display = "";
    form.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function cancelEditForm(pluginId) {
    const form = document.getElementById(`plugin-form-${pluginId}`);
    delete form.dataset.editId;
    form.reset();
    document.getElementById(`schedule-type-${pluginId}`).value = "once";
    updateScheduleFields(pluginId);
    document.getElementById(`schedule-submit-${pluginId}`).textContent = "Schedule Job";
    document.getElementById(`schedule-cancel-${pluginId}`).style.display = "none";
}

async function submitScheduleForm(event, pluginId) {
    event.preventDefault();
    const form = event.target;
    const editId = form.dataset.editId ? parseInt(form.dataset.editId, 10) : null;
    const data = Object.fromEntries(new FormData(form));
    const stype = data["_schedule_type"] || "once";
    let config = {};
    if (stype === "once")          config = { timestamp: data["_timestamp"] || "now" };
    else if (stype === "interval") config = { seconds: parseInt(data["_seconds"] || "60", 10) };
    else if (stype === "after")    config = { trigger_id: parseInt(data["_trigger_schedule_id"], 10) };
    else if (stype === "cron") {
        config = {
            minutes:       data["_minutes"]       || "*",
            hours:         data["_hours"]         || "*",
            days_of_week:  data["_days_of_week"]  || "*",
            days_of_month: data["_days_of_month"] || "*",
            months:        data["_months"]        || "*",
        };
    }
    const args = {};
    for (const [key, value] of Object.entries(data)) {
        if (!key.startsWith("_")) args[key] = value;
    }
    const url    = editId ? `/api/schedules/${editId}` : "/api/schedules";
    const method = editId ? "PATCH" : "POST";
    const body   = editId ? { type: stype, args, config }
                          : { plugin_id: pluginId, type: stype, args, config };
    const submitBtn = document.getElementById(`schedule-submit-${pluginId}`);
    let res, result;
    try {
        res = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        result = await res.json();
    } catch (err) {
        console.error("Schedule request failed:", err);
        if (submitBtn) {
            const prev = submitBtn.textContent;
            submitBtn.textContent = "Request failed";
            submitBtn.classList.add("btn-danger"); submitBtn.classList.remove("btn-primary");
            setTimeout(() => { submitBtn.textContent = prev; submitBtn.classList.remove("btn-danger"); submitBtn.classList.add("btn-primary"); }, 3000);
        }
        return;
    }
    if (!res.ok) {
        console.error(editId ? "Update failed:" : "Schedule failed:", result);
        if (submitBtn) {
            const prev = submitBtn.textContent;
            const msg = result?.error ? `Error: ${result.error}` : `Server error ${res.status}`;
            submitBtn.textContent = msg;
            submitBtn.classList.add("btn-danger"); submitBtn.classList.remove("btn-primary");
            setTimeout(() => { submitBtn.textContent = prev; submitBtn.classList.remove("btn-danger"); submitBtn.classList.add("btn-primary"); }, 4000);
        }
        return;
    }
    console.log(editId ? "Updated:" : "Scheduled:", result);
    if (editId) cancelEditForm(pluginId);
    const tbody = document.getElementById(`schedule-list-${pluginId}`);
    if (tbody) {
        const schedulesRes = await fetch(`/api/schedules?plugin_id=${encodeURIComponent(pluginId)}`);
        if (schedulesRes.ok) {
            const schedulesByPlugin = await schedulesRes.json();
            tbody.innerHTML = "";
            for (const job of (schedulesByPlugin[pluginId] || [])) {
                tbody.appendChild(renderScheduleRow(job, pluginId));
            }
        }
    }
}

// ---- Timeline selection sync ----
function syncTimelineSelection() {
    const checkedIds = new Set(
        [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")].map(cb => cb.dataset.id)
    );
    document.querySelectorAll(".tl-node[data-schedule-id]").forEach(g => {
        const rect = g.querySelector(".tl-node-rect");
        if (rect) rect.classList.toggle("tl-selected", checkedIds.has(String(g.getAttribute("data-schedule-id"))));
    });
}

// ---- Schedule Manager: dependent-task confirmation modal ----
function showDependentModal(dependents, onConfirm) {
    let modal = document.getElementById("dep-modal");
    if (!modal) {
        modal = document.createElement("div");
        modal.id = "dep-modal";
        modal.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:9999";
        document.body.appendChild(modal);
    }
    const names = dependents.map(d => `• ${d.plugin_id} (schedule #${d.id})`).join("<br>");
    modal.innerHTML = `
        <div style="background:#1e1e2e;border:1px solid #555;border-radius:8px;padding:24px;max-width:480px;color:#ddd">
            <h5 style="margin-bottom:12px">Start dependent tasks?</h5>
            <p style="font-size:0.9em;margin-bottom:12px">
                The following tasks were waiting for the stopped job to finish:
            </p>
            <div style="font-size:0.85em;margin-bottom:16px;line-height:1.8">${names}</div>
            <div style="display:flex;gap:8px;justify-content:flex-end">
                <button id="dep-no"  class="btn btn-sm btn-outline-secondary">Skip</button>
                <button id="dep-yes" class="btn btn-sm btn-primary">Start them</button>
            </div>
        </div>`;
    modal.querySelector("#dep-no").onclick = () => { modal.remove(); };
    modal.querySelector("#dep-yes").onclick = async () => {
        modal.remove();
        await onConfirm();
    };
}

// ---- Schedule Manager: row rendering ----
function renderSchedMgrRow(job) {
    const tr = document.createElement("tr");
    tr.dataset.schedId = job.id;
    tr.dataset.pluginId = job.plugin_id;
    tr.dataset.disabled = job.disabled ? "1" : "0";

    // Checkbox
    const cbTd = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.className = "schedule-cb"; cb.dataset.id = job.id;
    cb.addEventListener("change", () => {
        syncTimelineSelection();
        const allCbs = document.querySelectorAll("#schedmgr-body .schedule-cb");
        const selectAll = document.getElementById("schedmgr-select-all");
        if (selectAll) selectAll.checked = allCbs.length > 0 && [...allCbs].every(c => c.checked);
    });
    cbTd.appendChild(cb);

    const idTd = document.createElement("td");
    idTd.textContent = job.id;

    const pluginTd = document.createElement("td");
    pluginTd.textContent = job.plugin_id;

    const typeTd = document.createElement("td");
    typeTd.textContent = formatJob(job);

    // Next Run cell with inline set picker
    const nextTd = document.createElement("td");
    nextTd.style.whiteSpace = "nowrap";
    const nextRunSpan = document.createElement("span");
    nextRunSpan.textContent = job.next_run ?? "—";
    const setNRBtn = document.createElement("button");
    setNRBtn.className = "btn btn-sm btn-link p-0 ms-1 text-light";
    setNRBtn.title = "Set next run time";
    setNRBtn.textContent = "✎";
    setNRBtn.style.cssText = "font-size:0.75em;opacity:0.5;vertical-align:middle";
    const nrPickerWrap = document.createElement("span");
    nrPickerWrap.style.display = "none";
    const nrInput = document.createElement("input");
    nrInput.type = "datetime-local";
    nrInput.className = "form-control form-control-sm d-inline-block";
    nrInput.style.cssText = "width:auto;font-size:11px";
    const nrConfirmBtn = document.createElement("button");
    nrConfirmBtn.className = "btn btn-sm btn-primary ms-1";
    nrConfirmBtn.textContent = "✓";
    const nrCancelBtn = document.createElement("button");
    nrCancelBtn.className = "btn btn-sm btn-link text-light ms-1";
    nrCancelBtn.textContent = "✗";
    nrPickerWrap.append(nrInput, nrConfirmBtn, nrCancelBtn);
    nextTd.append(nextRunSpan, setNRBtn, nrPickerWrap);
    setNRBtn.onclick = () => {
        const now = new Date();
        now.setSeconds(0, 0);
        nrInput.value = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
        nextRunSpan.style.display = "none"; setNRBtn.style.display = "none"; nrPickerWrap.style.display = "";
    };
    nrCancelBtn.onclick = () => { nrPickerWrap.style.display = "none"; nextRunSpan.style.display = ""; setNRBtn.style.display = ""; };
    nrConfirmBtn.onclick = async () => {
        const val = nrInput.value;
        if (!val) return;
        await fetch(`/api/schedules/${job.id}/next-run`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ next_run: val }),
        });
        nrPickerWrap.style.display = "none"; nextRunSpan.style.display = ""; setNRBtn.style.display = "";
        loadSchedMgr();
    };

    const lastTd = document.createElement("td");
    lastTd.textContent = job.last_run ?? "never";

    // Schedule state column
    const schedStateTd = document.createElement("td");
    const schedBadge = document.createElement("span");
    if (job.disabled) {
        schedBadge.className = "badge text-bg-secondary";
        schedBadge.textContent = "Disabled";
    } else {
        schedBadge.className = "badge text-bg-success";
        schedBadge.textContent = "Enabled";
    }
    schedStateTd.appendChild(schedBadge);

    // Task state column
    const taskStateTd = document.createElement("td");
    const taskBadge = document.createElement("span");
    if (job.task_status === "running") {
        taskBadge.className = "badge text-bg-primary";
        taskBadge.textContent = "Running";
    } else if (job.task_status === "paused") {
        taskBadge.className = "badge text-bg-warning";
        taskBadge.textContent = "Paused";
    } else {
        taskBadge.className = "badge schedmgr-idle";
        taskBadge.textContent = "Idle";
    }
    taskStateTd.appendChild(taskBadge);

    // Actions column
    const btnTd = document.createElement("td");
    btnTd.className = "d-flex gap-1 flex-nowrap";

    // Schedule enable/disable toggle
    const toggleSchedBtn = document.createElement("button");
    toggleSchedBtn.className = "schedmgr-state-btn";
    toggleSchedBtn.textContent = job.disabled ? "▶" : "⏸";
    toggleSchedBtn.title = job.disabled ? "Enable schedule" : "Disable schedule";
    toggleSchedBtn.onclick = async () => {
        await fetch(`/api/schedules/${job.id}/disabled`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ disabled: !job.disabled }),
        });
        loadSchedMgr();
    };

    const resetBtn = document.createElement("button");
    resetBtn.className = "btn btn-sm btn-outline-secondary";
    resetBtn.textContent = "Reset Next Run";
    resetBtn.title = "Clear next_run so the scheduler recalculates from now";
    resetBtn.onclick = async () => {
        await fetch(`/api/schedules/${job.id}/reset`, { method: "POST" });
        loadSchedMgr();
    };

    const editBtn = document.createElement("button");
    editBtn.className = "btn btn-sm btn-outline-info";
    editBtn.textContent = "Edit";
    editBtn.onclick = () => editSchedulePlugin(job.plugin_id);

    btnTd.append(toggleSchedBtn, editBtn, resetBtn);

    // Task controls (pause, resume, stop) — only when task is active
    const pid = job.plugin_id.replaceAll(".", "__");

    if (job.task_status === "running") {
        const pauseBtn = document.createElement("button");
        pauseBtn.className = "btn btn-sm btn-outline-warning";
        pauseBtn.textContent = "Pause";
        pauseBtn.onclick = async () => {
            await fetch(`/api/plugins/${pid}/pause`, { method: "POST" });
            loadSchedMgr();
        };

        const stopBtn = document.createElement("button");
        stopBtn.className = "btn btn-sm btn-outline-danger";
        stopBtn.textContent = "Stop";
        stopBtn.onclick = async () => {
            const res = await fetch(`/api/plugins/${pid}/stop`, { method: "POST" });
            const data = await res.json();
            loadSchedMgr();
            if (data.dependent_schedules && data.dependent_schedules.length > 0) {
                showDependentModal(data.dependent_schedules, async () => {
                    await Promise.all(
                        data.dependent_schedules.map(d =>
                            fetch(`/api/schedules/${d.id}/run-now`, { method: "POST" })
                        )
                    );
                    loadSchedMgr();
                });
            }
        };

        btnTd.append(pauseBtn, stopBtn);
    } else if (job.task_status === "paused") {
        const resumeBtn = document.createElement("button");
        resumeBtn.className = "btn btn-sm btn-outline-success";
        resumeBtn.textContent = "Resume";
        resumeBtn.onclick = async () => {
            await fetch(`/api/plugins/${pid}/resume`, { method: "POST" });
            loadSchedMgr();
        };

        const stopBtn = document.createElement("button");
        stopBtn.className = "btn btn-sm btn-outline-danger";
        stopBtn.textContent = "Stop";
        stopBtn.onclick = async () => {
            await fetch(`/api/plugins/${pid}/stop`, { method: "POST" });
            loadSchedMgr();
        };

        btnTd.append(resumeBtn, stopBtn);
    }

    tr.append(cbTd, idTd, pluginTd, typeTd, nextTd, lastTd, schedStateTd, taskStateTd, btnTd);
    return tr;
}

async function loadSchedMgr() {
    const tbody = document.getElementById("schedmgr-body");
    if (!tbody) return;

    const prevChecked = new Set(
        [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")].map(cb => cb.dataset.id)
    );

    const res = await fetch("/api/schedules");
    if (!res.ok) return;
    const byPlugin = await res.json();
    const jobs = Object.entries(byPlugin).flatMap(([pluginId, scheds]) =>
        scheds.map(s => ({ ...s, plugin_id: pluginId }))
    );
    tbody.innerHTML = "";
    for (const job of jobs) tbody.appendChild(renderSchedMgrRow(job));

    document.querySelectorAll("#schedmgr-body .schedule-cb").forEach(cb => {
        if (prevChecked.has(cb.dataset.id)) cb.checked = true;
    });
    const allCbs = document.querySelectorAll("#schedmgr-body .schedule-cb");
    const selectAll = document.getElementById("schedmgr-select-all");
    if (selectAll) selectAll.checked = allCbs.length > 0 && [...allCbs].every(c => c.checked);
    syncTimelineSelection();

    // Refresh system pause button state
    const sysRes = await fetch("/api/system/status");
    if (sysRes.ok) {
        const sys = await sysRes.json();
        updateSystemPauseBtn(sys.paused, sys.pending_count);
    }
}

function updateSystemPauseBtn(paused, pendingCount) {
    const btn = document.getElementById("schedmgr-system-pause-btn");
    if (!btn) return;
    if (paused) {
        btn.textContent = pendingCount > 0
            ? `▶ Resume System (${pendingCount} pending)`
            : "▶ Resume System";
        btn.className = "btn btn-sm btn-warning";
        btn.title = "Resume system — pending tasks will start";
    } else {
        btn.textContent = "⏸ Pause System";
        btn.className = "btn btn-sm btn-outline-warning";
        btn.title = "Pause system — new tasks will queue instead of starting";
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    // Build TOC
    const tocList = document.getElementById("toc-list");
    const tocPopup = document.getElementById("toc-popup");
    const tocFab = document.getElementById("toc-fab");
    document.querySelectorAll("[data-plugin-name]").forEach(el => {
        const pluginId = el.id.replace("plugin-", "");
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#";
        a.textContent = el.getAttribute("data-plugin-name");
        a.addEventListener("click", e => {
            e.preventDefault();
            el.scrollIntoView({ behavior: "smooth", block: "start" });
            const collapseEl = document.getElementById(`collapse-${pluginId}`);
            if (collapseEl && !collapseEl.classList.contains("show"))
                new bootstrap.Collapse(collapseEl, { toggle: true });
            tocPopup.style.display = "none";
        });
        li.appendChild(a);
        tocList.appendChild(li);
    });
    document.getElementById("toc-search").addEventListener("input", e => {
        const q = e.target.value.toLowerCase();
        tocList.querySelectorAll("li").forEach(li => {
            li.style.display = li.textContent.toLowerCase().includes(q) ? "" : "none";
        });
    });
    tocFab.addEventListener("click", e => {
        e.stopPropagation();
        const showing = tocPopup.style.display === "none";
        tocPopup.style.display = showing ? "block" : "none";
        if (showing) {
            document.getElementById("toc-search").value = "";
            document.getElementById("toc-search").dispatchEvent(new Event("input"));
            document.getElementById("toc-search").focus();
        }
    });
    document.addEventListener("click", e => {
        if (!tocPopup.contains(e.target) && e.target !== tocFab) tocPopup.style.display = "none";
    });

    // Schedule Manager panel
    const schedmgrToggle  = document.getElementById("schedmgr-toggle");
    const schedmgrSection = document.getElementById("schedmgr-section");
    let schedmgrLoaded = false;

    schedmgrToggle.addEventListener("click", () => {
        const open = schedmgrSection.style.display === "none";
        schedmgrSection.style.display = open ? "block" : "none";
        schedmgrToggle.textContent = open ? "Schedules ▴" : "Schedules ▾";
        if (open && !schedmgrLoaded) { schedmgrLoaded = true; loadSchedMgr(); }
    });
    document.getElementById("schedmgr-refresh").addEventListener("click", loadSchedMgr);

    // Select-all
    document.getElementById("schedmgr-select-all").addEventListener("change", e => {
        const cbs = document.querySelectorAll("#schedmgr-body .schedule-cb");
        const allChecked = [...cbs].every(cb => cb.checked);
        cbs.forEach(cb => { cb.checked = !allChecked; });
        e.target.checked = !allChecked;
        syncTimelineSelection();
    });

    // Bulk enable/disable
    async function bulkSetDisabled(disabled) {
        const ids = [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => parseInt(cb.dataset.id, 10));
        if (!ids.length) return;
        await fetch("/api/schedules/bulk-set-disabled", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids, disabled }),
        });
        loadSchedMgr();
    }
    document.getElementById("schedmgr-disable-btn").addEventListener("click", () => bulkSetDisabled(true));
    document.getElementById("schedmgr-enable-btn").addEventListener("click",  () => bulkSetDisabled(false));

    // Bulk reset next-run
    async function bulkResetNextRun() {
        const ids = [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => parseInt(cb.dataset.id, 10));
        if (!ids.length) return;
        await Promise.all(ids.map(id => fetch(`/api/schedules/${id}/reset`, { method: "POST" })));
        loadSchedMgr();
    }
    document.getElementById("schedmgr-reset-btn").addEventListener("click", bulkResetNextRun);

    // System pause/resume
    const sysPauseBtn = document.getElementById("schedmgr-system-pause-btn");
    sysPauseBtn.addEventListener("click", async () => {
        const sysRes = await fetch("/api/system/status");
        const sys = await sysRes.json();
        if (sys.paused) {
            await fetch("/api/system/resume", { method: "POST" });
        } else {
            await fetch("/api/system/pause", { method: "POST" });
        }
        loadSchedMgr();
    });

    // Enable-at...
    const RESUME_PLUGIN_ID = "webrock.resume_schedules";
    document.getElementById("schedmgr-enable-at-btn").addEventListener("click", () => {
        const wrap = document.getElementById("schedmgr-enable-at-wrap");
        wrap.style.display = wrap.style.display === "none" ? "inline-flex" : "none";
    });
    document.getElementById("schedmgr-enable-at-cancel").addEventListener("click", () => {
        document.getElementById("schedmgr-enable-at-wrap").style.display = "none";
    });
    document.getElementById("schedmgr-enable-at-confirm").addEventListener("click", async () => {
        const ids = [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => cb.dataset.id);
        if (!ids.length) { alert("Select at least one schedule to enable."); return; }
        const dtVal = document.getElementById("schedmgr-enable-at-dt").value;
        if (!dtVal) { alert("Pick a date and time."); return; }
        const res = await fetch("/api/schedules", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                plugin_id: RESUME_PLUGIN_ID, type: "once",
                args: { schedule_ids: ids.join(",") },
                config: { timestamp: dtVal },
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(`Failed to schedule enable: ${err.error || res.status}`);
            return;
        }
        document.getElementById("schedmgr-enable-at-wrap").style.display = "none";
        loadSchedMgr();
    });

    // Timeline → schedule manager selection bridge
    window.tlSelectSchedule = async function(scheduleId) {
        if (schedmgrSection.style.display === "none") {
            schedmgrSection.style.display = "block";
            schedmgrToggle.textContent = "Schedules ▴";
        }
        if (!schedmgrLoaded) { schedmgrLoaded = true; await loadSchedMgr(); }
        const cb = document.querySelector(`#schedmgr-body .schedule-cb[data-id="${scheduleId}"]`);
        if (cb) {
            cb.checked = !cb.checked;
            syncTimelineSelection();
            const allCbs = document.querySelectorAll("#schedmgr-body .schedule-cb");
            const selectAll = document.getElementById("schedmgr-select-all");
            if (selectAll) selectAll.checked = allCbs.length > 0 && [...allCbs].every(c => c.checked);
        }
    };

    // Load schedule rows in plugin accordions
    const schedulesRes = await fetch("/api/schedules");
    if (schedulesRes.ok) {
        const schedules = await schedulesRes.json();
        for (const [pluginId, jobs] of Object.entries(schedules)) {
            const tbody = document.getElementById(`schedule-list-${pluginId}`);
            if (!tbody) continue;
            for (const job of jobs) tbody.appendChild(renderScheduleRow(job, pluginId));
        }
    }

    // Load run histories
    for (const tbody of document.querySelectorAll("[id^='run-history-']")) {
        const pluginId = tbody.id.replace("run-history-", "");
        const runsRes = await fetch(`/api/runs/${pluginId}`);
        if (!runsRes.ok) continue;
        const runs = await runsRes.json();
        for (const run of runs) tbody.appendChild(renderRunRow(run));
    }

    // Neon parallax
    (function () {
        const RATES = [0.07, 0.13, 0.05, 0.11, 0.09];
        const orbs = document.querySelectorAll(".neon-orb");
        let ticking = false;
        window.addEventListener("scroll", () => {
            if (ticking) return;
            ticking = true;
            requestAnimationFrame(() => {
                const y = window.scrollY;
                orbs.forEach((orb, i) => { orb.style.transform = `translateY(${y * RATES[i]}px)`; });
                ticking = false;
            });
        }, { passive: true });
    }());
});
