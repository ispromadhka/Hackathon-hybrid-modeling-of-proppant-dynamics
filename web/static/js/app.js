// Proppant Transport Simulator - Frontend JS

// State
let simulationData = null;
let isPlaying = false;
let playInterval = null;
let currentFrame = 0;
let totalFrames = 0;

// DOM Elements
const themeToggle = document.getElementById('theme-toggle');
const runBtn = document.getElementById('run-btn');
const playBtn = document.getElementById('play-btn');
const timeSlider = document.getElementById('time-slider');
const timeCurrent = document.getElementById('time-current');
const timeMax = document.getElementById('time-max');
const playbackSpeed = document.getElementById('playback-speed');
const statusMessage = document.getElementById('status-message');

// Parameter inputs
const params = {
    c_in: document.getElementById('c_in'),
    w0: document.getElementById('w0'),
    mu0: document.getElementById('mu0'),
    Q: document.getElementById('Q'),
    chi: document.getElementById('chi'),
    c_in_times: document.getElementById('c_in_times')
};

// Metrics elements
const metrics = {
    nnTime: document.getElementById('nn-time'),
    nsTime: document.getElementById('ns-time'),
    speedup: document.getElementById('speedup'),
    l2Error: document.getElementById('l2-error')
};

// Theme handling
function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
    }
}

function toggleTheme() {
    const isDark = document.body.classList.contains('dark-theme');
    if (isDark) {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        localStorage.setItem('theme', 'light');
    } else {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    }
    // Re-render plots with new theme
    if (simulationData) {
        renderFrame(currentFrame);
    } else {
        initEmptyPlots();
    }
}

themeToggle.addEventListener('click', toggleTheme);

// Plotly configuration
function getPlotlyLayout(title) {
    const isDark = document.body.classList.contains('dark-theme');
    return {
        title: null,
        margin: { l: 50, r: 80, t: 10, b: 40 },
        xaxis: {
            title: 'x (m)',
            color: isDark ? '#95a5a6' : '#5d6d7e',
            gridcolor: isDark ? '#3d4450' : '#dce1e7',
            zerolinecolor: isDark ? '#3d4450' : '#dce1e7'
        },
        yaxis: {
            title: 'y (m)',
            color: isDark ? '#95a5a6' : '#5d6d7e',
            gridcolor: isDark ? '#3d4450' : '#dce1e7',
            zerolinecolor: isDark ? '#3d4450' : '#dce1e7',
            scaleanchor: 'x',
            scaleratio: 0.5
        },
        paper_bgcolor: isDark ? '#242830' : '#ffffff',
        plot_bgcolor: isDark ? '#2d323c' : '#f8f9fa',
        font: {
            color: isDark ? '#ecf0f1' : '#2c3e50'
        }
    };
}

function getColorscale() {
    return [
        [0, 'rgb(59, 76, 192)'],
        [0.25, 'rgb(98, 130, 234)'],
        [0.5, 'rgb(220, 220, 220)'],
        [0.75, 'rgb(241, 156, 81)'],
        [1, 'rgb(180, 4, 38)']
    ];
}

// Initialize empty plots
function initEmptyPlots() {
    const layout = getPlotlyLayout();
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    const emptyData = [{
        z: [[0]],
        type: 'heatmap',
        colorscale: getColorscale(),
        showscale: true,
        colorbar: {
            title: 'c',
            titleside: 'right',
            thickness: 15,
            len: 0.9
        }
    }];

    Plotly.newPlot('nn-plot', emptyData, layout, config);
    Plotly.newPlot('ns-plot', emptyData, layout, config);
}

// Render a specific frame
function renderFrame(frameIdx) {
    if (!simulationData) return;

    const { nn, ns, x_grid, y_grid, times, c_max } = simulationData;
    const layout = getPlotlyLayout();

    const nnFrame = nn[frameIdx];
    const nsFrame = ns[frameIdx];

    const nnData = [{
        z: nnFrame,
        x: x_grid,
        y: y_grid,
        type: 'heatmap',
        colorscale: getColorscale(),
        zmin: 0,
        zmax: c_max,
        showscale: true,
        colorbar: {
            title: 'c',
            titleside: 'right',
            thickness: 15,
            len: 0.9
        },
        hovertemplate: 'x: %{x:.2f} m<br>y: %{y:.2f} m<br>c: %{z:.4f}<extra></extra>'
    }];

    const nsData = [{
        z: nsFrame,
        x: x_grid,
        y: y_grid,
        type: 'heatmap',
        colorscale: getColorscale(),
        zmin: 0,
        zmax: c_max,
        showscale: true,
        colorbar: {
            title: 'c',
            titleside: 'right',
            thickness: 15,
            len: 0.9
        },
        hovertemplate: 'x: %{x:.2f} m<br>y: %{y:.2f} m<br>c: %{z:.4f}<extra></extra>'
    }];

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.react('nn-plot', nnData, layout, config);
    Plotly.react('ns-plot', nsData, layout, config);

    // Update time display
    const t = times[frameIdx];
    timeCurrent.textContent = `${t.toFixed(1)} s`;
    timeSlider.value = frameIdx;
}

// Run simulation
async function runSimulation() {
    runBtn.classList.add('loading');
    runBtn.disabled = true;
    statusMessage.className = 'status-message';
    statusMessage.style.display = 'none';

    // Collect parameters
    const paramValues = {
        c_in: parseFloat(params.c_in.value),
        w0: parseFloat(params.w0.value),
        mu0: parseFloat(params.mu0.value),
        Q: parseFloat(params.Q.value),  // Will be negated on backend
        chi: parseFloat(params.chi.value),
        c_in_times: parseFloat(params.c_in_times.value)
    };

    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(paramValues)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Simulation failed');
        }

        const data = await response.json();
        simulationData = data;

        // Update UI
        totalFrames = data.times.length;
        timeSlider.max = totalFrames - 1;
        timeSlider.value = 0;
        currentFrame = 0;
        timeMax.textContent = `${data.times[totalFrames - 1].toFixed(1)} s`;

        // Update metrics
        metrics.nnTime.textContent = `${data.nn_time.toFixed(3)} s`;
        metrics.nsTime.textContent = `${data.ns_time.toFixed(2)} s`;
        metrics.speedup.textContent = `${data.speedup.toFixed(1)}x`;
        metrics.l2Error.textContent = `${(data.l2_error * 100).toFixed(2)}%`;

        // Render first frame
        renderFrame(0);

        statusMessage.textContent = 'Simulation completed successfully!';
        statusMessage.className = 'status-message success';

    } catch (error) {
        console.error('Simulation error:', error);
        statusMessage.textContent = error.message;
        statusMessage.className = 'status-message error';
    } finally {
        runBtn.classList.remove('loading');
        runBtn.disabled = false;
    }
}

runBtn.addEventListener('click', runSimulation);

// Timelapse controls
function togglePlay() {
    if (isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!simulationData) return;

    isPlaying = true;
    playBtn.classList.add('playing');

    const speed = parseFloat(playbackSpeed.value);
    const interval = 100 / speed;  // ms per frame

    playInterval = setInterval(() => {
        currentFrame++;
        if (currentFrame >= totalFrames) {
            currentFrame = 0;  // Loop
        }
        renderFrame(currentFrame);
    }, interval);
}

function stopPlayback() {
    isPlaying = false;
    playBtn.classList.remove('playing');
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
}

playBtn.addEventListener('click', togglePlay);

timeSlider.addEventListener('input', (e) => {
    stopPlayback();
    currentFrame = parseInt(e.target.value);
    renderFrame(currentFrame);
});

playbackSpeed.addEventListener('change', () => {
    if (isPlaying) {
        stopPlayback();
        startPlayback();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    switch (e.key) {
        case ' ':  // Space - play/pause
            e.preventDefault();
            togglePlay();
            break;
        case 'ArrowLeft':  // Previous frame
            e.preventDefault();
            stopPlayback();
            if (currentFrame > 0) {
                currentFrame--;
                renderFrame(currentFrame);
            }
            break;
        case 'ArrowRight':  // Next frame
            e.preventDefault();
            stopPlayback();
            if (currentFrame < totalFrames - 1) {
                currentFrame++;
                renderFrame(currentFrame);
            }
            break;
    }
});

// Initialize
initTheme();
initEmptyPlots();
