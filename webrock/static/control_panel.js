function stop(plugin) {
    fetch(`/stop/${plugin}`, { method: 'GET' })
        .then(response => response.json())
        .then(response => {
            console.log(response);
            checkStatusAndUpdateLight(plugin);
        })
        .catch(error => console.error('Error:', error));
}

function checkStatusAndUpdateLight(plugin) {
    fetch(`/status/${plugin}`, { method: 'GET' })
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
        await fetch(`/remove_schedule/${pluginId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: job.id })
        });
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

window.addEventListener("DOMContentLoaded", async () => {
    // Load schedules
    const res = await fetch("/get_schedules");
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
        const runsRes = await fetch(`/get_runs/${pluginId}`);
        if (!runsRes.ok) continue;
        const runs = await runsRes.json();
        for (const run of runs) {
            tbody.appendChild(renderRunRow(run));
        }
    }
});
