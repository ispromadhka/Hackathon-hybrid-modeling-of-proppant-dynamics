"""
Web visualization for proppant dynamics simulation.

Two-phase proppant transport with:
- Variable viscosity μ(c)
- Gravity settling Vslip(c)
- Darcy flow in fracture
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.fno import create_model
from src.solver.proppant_transport import (
    SimulationParams,
    PhysicalParams,
    ProppantTransportSolver,
    create_initial_condition
)


# Global model instance
MODEL = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path: Path = None):
    """Load trained FNO model."""
    global MODEL

    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent.parent / 'checkpoints' / 'best.pt'

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return None

    MODEL = create_model(nx=64, ny=32, n_times=40, device=DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()
    print(f"Model loaded from {checkpoint_path}")
    return MODEL


# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Proppant Transport Simulator",
    update_title=None
)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Proppant Transport Simulator", style={'margin': '0'}),
        html.P("Two-phase flow with gravity settling | FNO Surrogate",
               style={'color': '#666', 'margin': '5px 0'})
    ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#f8f9fa'}),

    # Main content
    html.Div([
        # Left panel - Controls
        html.Div([
            html.H4("Initial Condition"),
            dcc.Dropdown(
                id='ic-type',
                options=[
                    {'label': 'Gaussian Blob', 'value': 'gaussian'},
                    {'label': 'Step Function', 'value': 'step'},
                    {'label': 'Horizontal Layer', 'value': 'layer'}
                ],
                value='gaussian'
            ),

            html.Label("X Position", style={'marginTop': '10px'}),
            dcc.Slider(id='x0-slider', min=0.1, max=0.8, value=0.3, step=0.05,
                       marks={0.1: '0.1', 0.5: '0.5', 0.8: '0.8'}),

            html.Label("Y Position"),
            dcc.Slider(id='y0-slider', min=0.2, max=0.8, value=0.5, step=0.05,
                       marks={0.2: '0.2', 0.5: '0.5', 0.8: '0.8'}),

            html.Label("Concentration"),
            dcc.Slider(id='amplitude-slider', min=0.1, max=0.5, value=0.3, step=0.05,
                       marks={0.1: '0.1', 0.3: '0.3', 0.5: '0.5'}),

            html.Hr(),
            html.H4("Physics"),

            html.Label("Gravity [m/s²]"),
            dcc.Slider(id='gravity-slider', min=0, max=15, value=9.81, step=0.5,
                       marks={0: '0', 5: '5', 10: '10', 15: '15'}),

            html.Label("Fluid Viscosity [mPa·s]"),
            dcc.Slider(id='viscosity-slider', min=1, max=100, value=1, step=1,
                       marks={1: '1', 50: '50', 100: '100'}),

            html.Label("Fracture Width [mm]"),
            dcc.Slider(id='width-slider', min=1, max=20, value=5, step=1,
                       marks={1: '1', 10: '10', 20: '20'}),

            html.Hr(),

            html.Button('Run Simulation', id='run-button', n_clicks=0,
                        style={'width': '100%', 'padding': '12px', 'fontSize': '16px',
                               'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                               'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),

            html.Br(), html.Br(),

            html.Div(id='timing-info', style={'padding': '10px', 'backgroundColor': '#e9ecef',
                                               'borderRadius': '5px', 'fontSize': '13px'})

        ], style={'width': '260px', 'padding': '15px', 'backgroundColor': '#ffffff',
                  'borderRight': '1px solid #ddd', 'height': '100vh', 'overflowY': 'auto'}),

        # Center - Visualization
        html.Div([
            # Animation controls
            html.Div([
                html.Button('<<', id='first-frame', n_clicks=0, style={'fontSize': '14px', 'padding': '5px 10px'}),
                html.Button('<', id='prev-frame', n_clicks=0, style={'fontSize': '14px', 'padding': '5px 10px'}),
                html.Button('Play', id='play-button', n_clicks=0,
                            style={'fontSize': '14px', 'width': '70px', 'padding': '5px',
                                   'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '3px'}),
                html.Button('>', id='next-frame', n_clicks=0, style={'fontSize': '14px', 'padding': '5px 10px'}),
                html.Button('>>', id='last-frame', n_clicks=0, style={'fontSize': '14px', 'padding': '5px 10px'}),
                html.Span(id='frame-info', style={'marginLeft': '20px', 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '10px'}),

            # Frame slider
            dcc.Slider(id='frame-slider', min=0, max=39, value=0, step=1,
                       marks={0: 't=0'}),

            # Main plot
            dcc.Graph(
                id='concentration-plot',
                config={'responsive': True},
                style={'height': 'calc(100vh - 220px)'}
            ),

            # Interval for animation
            dcc.Interval(id='animation-interval', interval=100, disabled=True),

            # Stores
            dcc.Store(id='simulation-data'),
            dcc.Store(id='is-playing', data=False)

        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#f8f9fa'})

    ], style={'display': 'flex', 'height': 'calc(100vh - 80px)'})

], style={'fontFamily': 'Arial, sans-serif'})


@callback(
    [Output('simulation-data', 'data'),
     Output('timing-info', 'children'),
     Output('frame-slider', 'max'),
     Output('frame-slider', 'marks')],
    Input('run-button', 'n_clicks'),
    [State('ic-type', 'value'),
     State('x0-slider', 'value'),
     State('y0-slider', 'value'),
     State('amplitude-slider', 'value'),
     State('gravity-slider', 'value'),
     State('viscosity-slider', 'value'),
     State('width-slider', 'value')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, ic_type, x0, y0, amplitude, gravity, viscosity_mPa, width_mm):
    """Run proppant transport simulation."""
    global MODEL

    # Physical parameters
    phys = PhysicalParams(
        mu_f=viscosity_mPa * 0.001,  # mPa·s to Pa·s
        g=gravity,
        w0=width_mm * 0.001  # mm to m
    )

    # Simulation parameters (higher resolution for smooth visuals)
    sim = SimulationParams(
        nx=128, ny=64,
        T=4.0, dt=0.002,
        save_every=50
    )

    solver = ProppantTransportSolver(sim, phys)

    # Create initial condition
    ic_params = {'ic_type': ic_type}
    if ic_type == 'gaussian':
        ic_params.update({'x0': x0, 'y0': y0, 'amplitude': amplitude,
                          'sigma_x': 0.15, 'sigma_y': 0.12})
    elif ic_type == 'step':
        ic_params.update({'x_thresh': x0, 'amplitude': amplitude})
    elif ic_type == 'layer':
        ic_params.update({'y_center': y0, 'thickness': 0.3, 'amplitude': amplitude})

    c0 = create_initial_condition(solver.X, solver.Y, **ic_params)

    timing_info = []

    # Run FNO if available
    fno_time = None
    if MODEL is not None:
        c0_t = torch.from_numpy(c0).float().unsqueeze(0).to(DEVICE)
        u_t = torch.zeros_like(c0_t)
        v_t = torch.zeros_like(c0_t)

        start = time.perf_counter()
        with torch.no_grad():
            _ = MODEL(c0_t, u_t, v_t)
        fno_time = time.perf_counter() - start
        timing_info.append(html.P(f"FNO: {fno_time*1000:.1f} ms", style={'margin': '3px 0'}))

    # Run numerical solver
    start = time.perf_counter()
    times, trajectory = solver.solve(c0)
    num_time = time.perf_counter() - start
    timing_info.append(html.P(f"Numerical: {num_time*1000:.1f} ms", style={'margin': '3px 0'}))

    if fno_time:
        speedup = num_time / fno_time
        timing_info.append(html.P(f"Speedup: {speedup:.1f}x",
                                  style={'margin': '3px 0', 'fontWeight': 'bold', 'color': '#28a745'}))

    # Physics info
    timing_info.append(html.Hr())
    timing_info.append(html.P(f"g = {gravity} m/s²", style={'margin': '2px 0', 'fontSize': '12px'}))
    timing_info.append(html.P(f"μ = {viscosity_mPa} mPa·s", style={'margin': '2px 0', 'fontSize': '12px'}))
    timing_info.append(html.P(f"w = {width_mm} mm", style={'margin': '2px 0', 'fontSize': '12px'}))

    # Prepare data
    data = {
        'times': times.tolist(),
        'trajectory': trajectory.tolist(),
        'x': solver.x.tolist(),
        'y': solver.y.tolist()
    }

    max_frame = len(times) - 1
    marks = {0: 't=0', max_frame: f't={times[-1]:.1f}s'}

    return data, timing_info, max_frame, marks


@callback(
    Output('concentration-plot', 'figure'),
    [Input('simulation-data', 'data'),
     Input('frame-slider', 'value')]
)
def update_plot(data, frame_idx):
    """Update concentration plot."""
    if data is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Click 'Run Simulation' to start",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500
        )
        return fig

    x = np.array(data['x'])
    y = np.array(data['y'])
    trajectory = np.array(data['trajectory'])
    times = np.array(data['times'])

    frame_idx = min(frame_idx, len(times) - 1)
    t = times[frame_idx]

    c = trajectory[frame_idx].T

    # Use contour for smooth visualization (like in reference video)
    fig = go.Figure(data=go.Contour(
        z=c,
        x=x,
        y=y,
        colorscale='Viridis',
        zmin=0,
        zmax=0.5,
        contours=dict(
            coloring='heatmap',
            showlabels=False,
        ),
        line=dict(width=0),
        colorbar=dict(title='c', titleside='right', tickformat='.2f')
    ))

    fig.update_layout(
        title=f"Proppant Concentration c(x,y) at t = {t:.2f}s",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=0.5),
        margin=dict(l=60, r=60, t=50, b=50)
    )

    return fig


@callback(
    Output('frame-info', 'children'),
    [Input('frame-slider', 'value'),
     Input('simulation-data', 'data')]
)
def update_frame_info(frame_idx, data):
    if data is None:
        return "No data"
    times = data['times']
    frame_idx = min(frame_idx, len(times) - 1)
    return f"Frame {frame_idx + 1}/{len(times)} | t = {times[frame_idx]:.2f}s"


@callback(
    [Output('frame-slider', 'value'),
     Output('animation-interval', 'disabled'),
     Output('play-button', 'children'),
     Output('is-playing', 'data')],
    [Input('play-button', 'n_clicks'),
     Input('animation-interval', 'n_intervals'),
     Input('first-frame', 'n_clicks'),
     Input('last-frame', 'n_clicks'),
     Input('prev-frame', 'n_clicks'),
     Input('next-frame', 'n_clicks')],
    [State('frame-slider', 'value'),
     State('is-playing', 'data'),
     State('simulation-data', 'data')],
    prevent_initial_call=True
)
def animation_controls(play_clicks, n_intervals, first_clicks, last_clicks,
                       prev_clicks, next_clicks, current_frame, is_playing, data):
    """Handle animation controls."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_frame, True, 'Play', False

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    max_frame = 39 if data is None else len(data['times']) - 1

    if trigger == 'play-button':
        is_playing = not is_playing
        return current_frame, not is_playing, 'Pause' if is_playing else 'Play', is_playing

    elif trigger == 'animation-interval' and is_playing:
        new_frame = (current_frame + 1) % (max_frame + 1)
        return new_frame, False, 'Pause', True

    elif trigger == 'first-frame':
        return 0, True, 'Play', False

    elif trigger == 'last-frame':
        return max_frame, True, 'Play', False

    elif trigger == 'prev-frame':
        return max(0, current_frame - 1), True, 'Play', False

    elif trigger == 'next-frame':
        return min(max_frame, current_frame + 1), True, 'Play', False

    return current_frame, not is_playing, 'Pause' if is_playing else 'Play', is_playing


def run_app(debug: bool = True, port: int = 8050):
    """Run the Dash application."""
    load_model()
    print(f"\nStarting server at http://localhost:{port}")
    print("Two-phase proppant transport solver")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_app()
