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
    if (job.type === "cron") return `Cron: ${c.minutes}m ${c.hours}h`;
    if (job.type === "once") return `Once @ ${c.timestamp === "now" ? "now" : c.timestamp}`;
    if (job.type === "after") return `After schedule #${c.trigger_id}`;
    return "Unknown";
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
    const removeBtn = document.createElement("button");
    removeBtn.className = "btn btn-sm btn-danger";
    removeBtn.textContent = "Remove";
    removeBtn.onclick = async () => {
        await fetch(`/api/schedules/${job.id}`, { method: "DELETE" });
        tr.remove();
    };
    btnTd.appendChild(removeBtn);

    tr.append(idTd, typeTd, argsTd, nextRunTd, lastRunTd, btnTd);
    return tr;
}

function renderRunRow(run) {
    const fmt = ts => ts ? new Date(ts * 1000).toLocaleString() : "—";
    const tr = document.createElement("tr");
    [
        run.id,
        run.schedule_id,
        fmt(run.started_at),
        fmt(run.finished_at),
        run.status,
        run.error ?? (run.result !== null ? run.result : "")
    ].forEach(val => {
        const td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
    });
    return tr;
}

async function submitScheduleForm(event, pluginId) {
    event.preventDefault();
    const form = event.target;
    const data = Object.fromEntries(new FormData(form));

    const stype = data["_schedule_type"] || "once";

    // Build schedule config from underscore-prefixed fields
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

    // All other fields are plugin args (no underscore prefix)
    const args = {};
    for (const [key, value] of Object.entries(data)) {
        if (!key.startsWith("_")) {
            args[key] = value;
        }
    }

    const res = await fetch("/api/schedules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plugin_id: pluginId, type: stype, args, config }),
    });
    const result = await res.json();
    console.log("Scheduled:", result);

    if (res.ok) {
        const tbody = document.getElementById(`schedule-list-${pluginId}`);
        if (tbody) {
            // Refresh schedule list for this plugin
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
}

window.addEventListener("DOMContentLoaded", async () => {
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
});
