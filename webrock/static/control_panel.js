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
    const s = job.schedule;
    if (s.type === "interval") return `Every ${s.seconds}s`;
    if (s.type === "cron") return `Cron: ${s.minutes}m ${s.hours}h`;
    if (s.type === "once") return `Once @ ${s.timestamp === "now" ? "now" : new Date(s.timestamp).toLocaleString()}`;
    if (s.type === "after") return `After ${s.trigger_plugin}[${s.trigger_schedule_index}]`;
    return "Unknown";
}

function computeNextRun(job) {
    const s = job.schedule;
    if (s.type === "after") return null;
    if (s.next_run) return new Date(s.next_run * 1000);
    return null;
}

function renderScheduleRow(job, pluginId, index) {
    const tr = document.createElement("tr");

    const descTd = document.createElement("td");
    descTd.textContent = formatJob(job);

    const nextRunTd = document.createElement("td");
    const next = computeNextRun(job);
    nextRunTd.dataset.timestamp = next ? next.toISOString() : "";
    nextRunTd.className = "next-run-cell";

    const btnTd = document.createElement("td");
    const removeBtn = document.createElement("button");
    removeBtn.className = "btn btn-sm btn-danger";
    removeBtn.textContent = "Remove";
    removeBtn.onclick = async () => {
        await fetch(`/remove_schedule/${pluginId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index })
        });
        tr.remove();
    };
    btnTd.appendChild(removeBtn);

    tr.appendChild(descTd);
    tr.appendChild(nextRunTd);
    tr.appendChild(btnTd);

    return tr;
}

async function submitScheduleForm(event, pluginId) {
    event.preventDefault();

    const type = document.getElementById(`schedule-type-${pluginId}`).value;

    let job = { type };

    if (type === "interval") {
        job.interval_seconds =
            parseInt(document.getElementById(`${pluginId}.interval_seconds`).value);
    }
    if (type === "cron") {
        job.time = document.getElementById(`${pluginId}.cron_time`).value;
    }
    if (type === "weekly") {
        job.weekday =
            parseInt(document.getElementById(`${pluginId}.weekday`).value);
        job.time =
            document.getElementById(`${pluginId}.weekly_time`).value;
    }
    if (type === "once") {
        job.when =
            document.getElementById(`${pluginId}.once_datetime`).value;
    }

    const res = await fetch(`/add_schedule/${pluginId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(job)
    });

    if (res.ok) {
        const tbody = document.getElementById(`schedule-list-${pluginId}`);
        const row = renderScheduleRow(job, pluginId, tbody.children.length);
        tbody.appendChild(row);
        event.target.reset();
        updateScheduleFields(pluginId);
    }
}

function updateCountdowns() {
    const cells = document.querySelectorAll(".next-run-cell");
    const now = new Date();
    for (const cell of cells) {
        const ts = cell.dataset.timestamp;
        if (!ts) {
            cell.textContent = "n/a";
            continue;
        }
        const next = new Date(ts);
        const diff = Math.max(0, next - now);
        const sec = Math.floor(diff / 1000) % 60;
        const min = Math.floor(diff / 60000) % 60;
        const hr = Math.floor(diff / 3600000);
        cell.textContent = `${hr}h ${min}m ${sec}s`;
    }
}

setInterval(updateCountdowns, 1000);

window.addEventListener("DOMContentLoaded", async () => {
    const res = await fetch("/get_schedules");
    if (!res.ok) return;

    const schedules = await res.json();

    for (const [pluginId, jobs] of Object.entries(schedules)) {
        const tbody = document.getElementById(`schedule-list-${pluginId}`);
        if (!tbody) continue;

        jobs.forEach((job, index) => {
            const row = renderScheduleRow(job, pluginId, index);
            tbody.appendChild(row);
        });
    }
});
