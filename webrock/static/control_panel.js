function stop(plugin) {
    fetch(`/api/plugins/${plugin}/stop`, { method: 'POST' })
        .then(response => response.json())
        .then(response => {
            console.log(response);
            checkStatusAndUpdateLight(plugin);
        })
        .catch(error => console.error('Error:', error));
}

function checkStatusAndUpdateLight(plugin) {
    fetch(`/api/plugins/${plugin}/status`, { method: 'GET' })
        .then(res => res.json())
        .then(res => {
            console.log(res);

            const statusLight = document.getElementById(plugin + '_statusLight');
            if (res.running) {
                statusLight.classList.remove('red');
                statusLight.classList.add('blue');
            } else {
                statusLight.classList.remove('blue');
                statusLight.classList.add('red');
            }

            const lastRunTimestamp = document.getElementById(`${plugin}_lastRunTimestamp`);
            lastRunTimestamp.textContent = `Last Run Time: ${res.lastRunTimestamp}`;
        })
        .catch(error => {
            console.error('Error:', error);
            statusLight.classList.remove('blue');
            statusLight.classList.add('red');
        });
}


function updateScheduleFields(pluginId) {
    const type = document.getElementById(`schedule-type-${pluginId}`).value;

    const sections = {
        interval: `schedule-interval-wrap-${pluginId}`,
        cron: `schedule-cron-wrap-${pluginId}`,
        once: `schedule-once-wrap-${pluginId}`,
        after: `schedule-after-wrap-${pluginId}`
    };

    for (const [key, id] of Object.entries(sections)) {
        document.getElementById(id).style.display =
            key === type ? "block" : "none";
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

function renderScheduleRow(job, pluginId) {
    const tr = document.createElement("tr");

    const idTd = document.createElement("td");
    idTd.textContent = job.id;

    const typeTd = document.createElement("td");
    typeTd.textContent = formatJob(job);

    const argsTd = document.createElement("td");
    argsTd.textContent = JSON.stringify(job.args);

    const nextRunTd = document.createElement("td");
    nextRunTd.textContent = job.next_run ?? "n/a";

    const lastRunTd = document.createElement("td");
    lastRunTd.textContent = job.last_run ?? "never";

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
    btnTd.appendChild(editBtn);
    btnTd.appendChild(removeBtn);

    tr.append(idTd, typeTd, argsTd, nextRunTd, lastRunTd, btnTd);
    return tr;
}

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

        td.appendChild(textSpan);
        td.appendChild(chevron);
    } else {
        td.textContent = text;
    }

    tr.appendChild(td);
    return tr;
}

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
    if (job.type === "once")         setVal("_timestamp", c.timestamp === "now" ? "" : (c.timestamp ?? ""));
    else if (job.type === "interval") setVal("_seconds", c.seconds ?? 60);
    else if (job.type === "after")    setVal("_trigger_schedule_id", c.trigger_id ?? "");
    else if (job.type === "cron") {
        setVal("_minutes",      c.minutes      ?? "*");
        setVal("_hours",        c.hours        ?? "*");
        setVal("_days_of_week", c.days_of_week ?? "*");
        setVal("_days_of_month",c.days_of_month?? "*");
        setVal("_months",       c.months       ?? "*");
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
    if (stype === "once") {
        config = { timestamp: data["_timestamp"] || "now" };
    } else if (stype === "interval") {
        config = { seconds: parseInt(data["_seconds"] || "60", 10) };
    } else if (stype === "after") {
        config = { trigger_id: parseInt(data["_trigger_schedule_id"], 10) };
    } else if (stype === "cron") {
        config = {
            minutes: data["_minutes"] || "*",
            hours: data["_hours"] || "*",
            days_of_week: data["_days_of_week"] || "*",
            days_of_month: data["_days_of_month"] || "*",
            months: data["_months"] || "*",
        };
    }

    const args = {};
    for (const [key, value] of Object.entries(data)) {
        if (!key.startsWith("_")) args[key] = value;
    }

    const url    = editId ? `/api/schedules/${editId}` : "/api/schedules";
    const method = editId ? "PATCH" : "POST";
    const body   = editId
        ? { type: stype, args, config }
        : { plugin_id: pluginId, type: stype, args, config };

    const submitBtn = document.getElementById(`schedule-submit-${pluginId}`);

    let res, result;
    try {
        res = await fetch(url, {
            method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        result = await res.json();
    } catch (err) {
        console.error("Schedule request failed:", err);
        if (submitBtn) {
            const prev = submitBtn.textContent;
            submitBtn.textContent = "Request failed";
            submitBtn.classList.add("btn-danger");
            submitBtn.classList.remove("btn-primary");
            setTimeout(() => {
                submitBtn.textContent = prev;
                submitBtn.classList.remove("btn-danger");
                submitBtn.classList.add("btn-primary");
            }, 3000);
        }
        return;
    }

    if (!res.ok) {
        console.error(editId ? "Update failed:" : "Schedule failed:", result);
        if (submitBtn) {
            const prev = submitBtn.textContent;
            const msg = result?.error ? `Error: ${result.error}` : `Server error ${res.status}`;
            submitBtn.textContent = msg;
            submitBtn.classList.add("btn-danger");
            submitBtn.classList.remove("btn-primary");
            setTimeout(() => {
                submitBtn.textContent = prev;
                submitBtn.classList.remove("btn-danger");
                submitBtn.classList.add("btn-primary");
            }, 4000);
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

function syncTimelineSelection() {
    const checkedIds = new Set(
        [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => cb.dataset.id)
    );
    document.querySelectorAll(".tl-node[data-schedule-id]").forEach(g => {
        const sid = String(g.getAttribute("data-schedule-id"));
        const rect = g.querySelector(".tl-node-rect");
        if (rect) rect.classList.toggle("tl-selected", checkedIds.has(sid));
    });
}

function renderSchedMgrRow(job) {
    const tr = document.createElement("tr");
    tr.dataset.schedId = job.id;
    tr.dataset.pluginId = job.plugin_id;
    tr.dataset.paused = job.paused ? "1" : "0";

    const cbTd = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "schedule-cb";
    cb.dataset.id = job.id;
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

    const nextTd = document.createElement("td");
    nextTd.textContent = job.next_run ?? "—";

    const lastTd = document.createElement("td");
    lastTd.textContent = job.last_run ?? "never";

    const stateTd = document.createElement("td");
    const badge = document.createElement("span");
    if (job.paused) {
        badge.className = "badge text-bg-warning";
        badge.textContent = "Paused";
    } else if (job.running) {
        badge.className = "badge text-bg-primary";
        badge.textContent = "Running";
    } else {
        badge.className = "badge schedmgr-idle";
        badge.textContent = "Idle";
    }
    stateTd.appendChild(badge);

    const btnTd = document.createElement("td");
    btnTd.className = "d-flex gap-1 flex-nowrap";

    const pauseBtn = document.createElement("button");
    pauseBtn.className = "schedmgr-state-btn";
    pauseBtn.textContent = job.paused ? "▶" : "⏸";
    pauseBtn.title = job.paused ? "Resume" : "Pause";
    pauseBtn.onclick = async () => {
        await fetch(`/api/schedules/${job.id}/paused`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paused: !job.paused }),
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

    const stopBtn = document.createElement("button");
    stopBtn.className = "btn btn-sm btn-outline-danger";
    stopBtn.textContent = "Stop";
    stopBtn.disabled = !job.running;
    stopBtn.onclick = async () => {
        const pid = job.plugin_id.replaceAll(".", "__");
        await fetch(`/api/plugins/${pid}/stop`, { method: "POST" });
        loadSchedMgr();
    };

    const editBtn = document.createElement("button");
    editBtn.className = "btn btn-sm btn-outline-info";
    editBtn.textContent = "Edit";
    editBtn.onclick = () => editSchedulePlugin(job.plugin_id);

    btnTd.append(pauseBtn, editBtn, resetBtn, stopBtn);
    tr.append(cbTd, idTd, pluginTd, typeTd, nextTd, lastTd, stateTd, btnTd);
    return tr;
}

async function loadSchedMgr() {
    const tbody = document.getElementById("schedmgr-body");
    if (!tbody) return;

    const prevChecked = new Set(
        [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => cb.dataset.id)
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
}

window.addEventListener("DOMContentLoaded", async () => {
    // Build TOC from plugin elements
    const tocList = document.getElementById("toc-list");
    const tocPopup = document.getElementById("toc-popup");
    const tocFab = document.getElementById("toc-fab");

    document.querySelectorAll("[data-plugin-name]").forEach(el => {
        const name = el.getAttribute("data-plugin-name");
        const pluginId = el.id.replace("plugin-", "");
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#";
        a.textContent = name;
        a.addEventListener("click", e => {
            e.preventDefault();
            // Scroll to plugin
            el.scrollIntoView({ behavior: "smooth", block: "start" });
            // Open the accordion
            const collapseEl = document.getElementById(`collapse-${pluginId}`);
            if (collapseEl && !collapseEl.classList.contains("show")) {
                new bootstrap.Collapse(collapseEl, { toggle: true });
            }
            // Hide popup
            tocPopup.style.display = "none";
        });
        li.appendChild(a);
        tocList.appendChild(li);
    });

    // Filter TOC on search input
    const tocSearch = document.getElementById("toc-search");
    tocSearch.addEventListener("input", () => {
        const query = tocSearch.value.toLowerCase();
        tocList.querySelectorAll("li").forEach(li => {
            li.style.display = li.textContent.toLowerCase().includes(query) ? "" : "none";
        });
    });

    // Toggle popup on FAB click
    tocFab.addEventListener("click", e => {
        e.stopPropagation();
        const showing = tocPopup.style.display === "none";
        tocPopup.style.display = showing ? "block" : "none";
        if (showing) {
            tocSearch.value = "";
            tocSearch.dispatchEvent(new Event("input"));
            tocSearch.focus();
        }
    });

    // Close popup when clicking outside
    document.addEventListener("click", e => {
        if (!tocPopup.contains(e.target) && e.target !== tocFab) {
            tocPopup.style.display = "none";
        }
    });

    // Schedule manager panel
    const schedmgrToggle = document.getElementById("schedmgr-toggle");
    const schedmgrSection = document.getElementById("schedmgr-section");
    let schedmgrLoaded = false;

    schedmgrToggle.addEventListener("click", () => {
        const open = schedmgrSection.style.display === "none";
        schedmgrSection.style.display = open ? "block" : "none";
        schedmgrToggle.textContent = open ? "Schedules ▴" : "Schedules ▾";
        if (open && !schedmgrLoaded) {
            schedmgrLoaded = true;
            loadSchedMgr();
        }
    });

    document.getElementById("schedmgr-refresh").addEventListener("click", loadSchedMgr);

    document.getElementById("schedmgr-select-all").addEventListener("change", e => {
        const cbs = document.querySelectorAll("#schedmgr-body .schedule-cb");
        const allChecked = [...cbs].every(cb => cb.checked);
        cbs.forEach(cb => { cb.checked = !allChecked; });
        e.target.checked = !allChecked;
        syncTimelineSelection();
    });

    async function bulkSetPaused(paused) {
        const ids = [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => parseInt(cb.dataset.id, 10));
        if (!ids.length) return;
        await fetch("/api/schedules/bulk-set-paused", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids, paused }),
        });
        loadSchedMgr();
    }

    document.getElementById("schedmgr-pause-btn").addEventListener("click", () => bulkSetPaused(true));
    document.getElementById("schedmgr-start-btn").addEventListener("click", () => bulkSetPaused(false));

    async function bulkResetNextRun() {
        const ids = [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => parseInt(cb.dataset.id, 10));
        if (!ids.length) return;
        await Promise.all(ids.map(id => fetch(`/api/schedules/${id}/reset`, { method: "POST" })));
        loadSchedMgr();
    }
    document.getElementById("schedmgr-reset-btn").addEventListener("click", bulkResetNextRun);

    const RESUME_PLUGIN_ID = "webrock.resume_schedules";

    document.getElementById("schedmgr-resume-at-btn").addEventListener("click", () => {
        const wrap = document.getElementById("schedmgr-resume-at-wrap");
        wrap.style.display = wrap.style.display === "none" ? "inline-flex" : "none";
    });

    document.getElementById("schedmgr-resume-at-cancel").addEventListener("click", () => {
        document.getElementById("schedmgr-resume-at-wrap").style.display = "none";
    });

    document.getElementById("schedmgr-resume-at-confirm").addEventListener("click", async () => {
        const ids = [...document.querySelectorAll("#schedmgr-body .schedule-cb:checked")]
            .map(cb => cb.dataset.id);
        if (!ids.length) { alert("Select at least one schedule to resume."); return; }
        const dtVal = document.getElementById("schedmgr-resume-at-dt").value;
        if (!dtVal) { alert("Pick a date and time."); return; }
        const ts = Math.floor(new Date(dtVal).getTime() / 1000);
        const res = await fetch("/api/schedules", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                plugin_id: RESUME_PLUGIN_ID,
                type: "once",
                args: { schedule_ids: ids.join(",") },
                config: { timestamp: ts },
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(`Failed to schedule resume: ${err.error || res.status}`);
            return;
        }
        document.getElementById("schedmgr-resume-at-wrap").style.display = "none";
        loadSchedMgr();
    });

    window.tlSelectSchedule = async function(scheduleId) {
        if (schedmgrSection.style.display === "none") {
            schedmgrSection.style.display = "block";
            schedmgrToggle.textContent = "Schedules ▴";
        }
        if (!schedmgrLoaded) {
            schedmgrLoaded = true;
            await loadSchedMgr();
        }
        const cb = document.querySelector(`#schedmgr-body .schedule-cb[data-id="${scheduleId}"]`);
        if (cb) {
            cb.checked = !cb.checked;
            syncTimelineSelection();
            const allCbs = document.querySelectorAll("#schedmgr-body .schedule-cb");
            const selectAll = document.getElementById("schedmgr-select-all");
            if (selectAll) selectAll.checked = allCbs.length > 0 && [...allCbs].every(c => c.checked);
        }
    };

    // Load schedules
    const res = await fetch("/api/schedules");
    if (!res.ok) return;
    const schedules = await res.json();

    for (const [pluginId, jobs] of Object.entries(schedules)) {
        const tbody = document.getElementById(`schedule-list-${pluginId}`);
        if (!tbody) continue;
        for (const job of jobs) {
            tbody.appendChild(renderScheduleRow(job, pluginId));
        }
    }

    // Load run histories for all plugins that have a run-history tbody
    const historyBodies = document.querySelectorAll("[id^='run-history-']");
    for (const tbody of historyBodies) {
        const pluginId = tbody.id.replace("run-history-", "");
        const runsRes = await fetch(`/api/runs/${pluginId}`);
        if (!runsRes.ok) continue;
        const runs = await runsRes.json();
        for (const run of runs) {
            tbody.appendChild(renderRunRow(run));
        }
    }

    // Neon parallax — each orb drifts at a different rate on scroll
    (function () {
        const RATES = [0.07, 0.13, 0.05, 0.11, 0.09];
        const orbs = document.querySelectorAll(".neon-orb");
        let ticking = false;
        window.addEventListener("scroll", () => {
            if (ticking) return;
            ticking = true;
            requestAnimationFrame(() => {
                const y = window.scrollY;
                orbs.forEach((orb, i) => {
                    orb.style.transform = `translateY(${y * RATES[i]}px)`;
                });
                ticking = false;
            });
        }, { passive: true });
    }());
});
