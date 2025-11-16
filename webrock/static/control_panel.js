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


function createScheduleListItem(job, pluginId, index) {
    const li = document.createElement("li");

    if (job.type === "interval") li.textContent = `Every ${job.interval_seconds}s`;
    else if (job.type === "cron") li.textContent = `At ${job.time}`;
    else if (job.type === "once") li.textContent = `One-time`;

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "Remove";
    removeBtn.className = "btn btn-sm btn-danger ms-2";
    removeBtn.onclick = async () => {
        await fetch(`/remove_schedule/${pluginId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index })
        });
        li.remove();
    };

    li.appendChild(removeBtn);
    return li;
}

async function addSchedule(pluginId) {
    const type = document.getElementById(`schedule-type-${pluginId}`).value;
    const value = document.getElementById(`schedule-value-${pluginId}`).value;

    let job = { type, args: {} };
    if (type === "interval") job.interval_seconds = parseInt(value);
    else if (type === "cron") job.time = value;

    const res = await fetch(`/update_schedule/${pluginId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(job)
    });

    if (res.ok) {
        const ul = document.getElementById(`schedule-list-${pluginId}`);
        const li = createScheduleListItem(job, pluginId, ul.children.length);
        ul.appendChild(li);
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    const res = await fetch("/get_schedules");
    if (!res.ok) return;
    const schedules = await res.json();

    for (const [pluginId, jobs] of Object.entries(schedules)) {
        const ul = document.getElementById(`schedule-list-${pluginId}`);
        if (!ul) continue;

        jobs.forEach((job, index) => {
            const li = createScheduleListItem(job, pluginId, index);
            ul.appendChild(li);
        });
    }
});

