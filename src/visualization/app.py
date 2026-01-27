"""
Proppant Transport Simulator - NN vs NS Comparison UI.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import time
import warnings
import torch
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.solver_wrapper import ProppantSolver
from src.model.fno import create_model

app = dash.Dash(__name__, title="Proppant Simulator - NN vs NS")

# Domain size constants
DOMAIN_LX = 60.0
DOMAIN_LY = 30.0

def create_empty_figure(title="", show_message=None):
    """Create empty placeholder figure with correct axis range and 2:1 aspect ratio."""
    fig = go.Figure()
    if show_message:
        fig.add_annotation(
            text=show_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#7f8c8d')
        )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=11)),
        xaxis=dict(title='x [m]', range=[0, DOMAIN_LX], constrain='domain'),
        yaxis=dict(title='y [m]', range=[0, DOMAIN_LY], scaleanchor='x', scaleratio=1),
        margin=dict(l=50, r=80, t=30, b=50)
    )
    return fig

# Try to load trained model
MODEL = None
MODEL_METADATA = None

def load_model():
    """Load trained FNO model if available."""
    global MODEL, MODEL_METADATA
    checkpoint_path = Path(__file__).parent.parent.parent / 'checkpoints' / 'best.pt'
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'metadata.json'

    if checkpoint_path.exists():
        try:
            import json
            if data_path.exists():
                with open(data_path) as f:
                    MODEL_METADATA = json.load(f)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Detect model architecture from checkpoint
            state_dict = checkpoint['model_state_dict']

            # Get grid size from grid_x buffer
            if 'grid_x' in state_dict:
                grid_shape = state_dict['grid_x'].shape
                nx = grid_shape[2]
                ny = grid_shape[3]
            else:
                nx = MODEL_METADATA.get('nx', 64) if MODEL_METADATA else 64
                ny = MODEL_METADATA.get('ny', 32) if MODEL_METADATA else 32

            # Get n_params from lift layer (in_channels = n_params + 2)
            if 'lift.weight' in state_dict:
                in_channels = state_dict['lift.weight'].shape[1]
                n_params = in_channels - 2  # subtract 2 for grid coordinates
            else:
                n_params = 5

            # Get n_times from project layer
            if 'project.2.weight' in state_dict:
                n_times = state_dict['project.2.weight'].shape[0]
            else:
                n_times = 26

            print(f"Detected model: nx={nx}, ny={ny}, n_times={n_times}, n_params={n_params}")

            MODEL = create_model(nx=nx, ny=ny, n_times=n_times, n_params=n_params, device=device)
            # Load with strict=False to handle buffer issues, then verify key parameters
            MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
            MODEL.eval()
            print(f"Loaded FNO model from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            MODEL = None
    return False

# Try to load model on startup
load_model()

app.layout = html.Div([
    html.H1("Proppant Transport: Neural Network vs Numerical Solver",
            style={'textAlign': 'center', 'padding': '12px', 'backgroundColor': '#2c3e50',
                   'color': 'white', 'margin': '0', 'fontSize': '22px'}),

    html.Div([
        # Controls panel (left)
        html.Div([
            html.H4("Parameters", style={'color': '#2c3e50', 'marginTop': '0'}),

            html.Label("Inlet Concentration c₀", style={'fontWeight': 'bold'}),
            dcc.Slider(id='c_inlet', min=0.1, max=0.5, value=0.35, step=0.05,
                      marks={0.1: '0.1', 0.25: '0.25', 0.4: '0.4', 0.5: '0.5'}),

            html.Label("Flow Rate Q [m²/s]", style={'fontWeight': 'bold', 'marginTop': '8px'}),
            dcc.Slider(id='Q_inlet', min=0.02, max=0.1, value=0.05, step=0.01,
                      marks={0.02: '0.02', 0.05: '0.05', 0.1: '0.1'}),

            html.Label("Gravity [m/s²]", style={'fontWeight': 'bold', 'marginTop': '8px'}),
            dcc.Slider(id='gravity', min=0, max=15, value=9.81, step=1,
                      marks={0: '0', 5: '5', 10: '10', 15: '15'}),

            html.Label("Viscosity μ₀ [mPa·s]", style={'fontWeight': 'bold', 'marginTop': '8px'}),
            dcc.Slider(id='viscosity', min=1, max=50, value=1, step=1,
                      marks={1: '1', 10: '10', 25: '25', 50: '50'}),

            html.Label("Particle Radius [μm]", style={'fontWeight': 'bold', 'marginTop': '8px'}),
            dcc.Slider(id='r_particle', min=100, max=500, value=200, step=50,
                      marks={100: '100', 200: '200', 300: '300', 500: '500'}),

            html.Hr(),
            html.H4("Simulation", style={'color': '#2c3e50'}),

            html.Label("Injection Duration [s] (0=continuous)", style={'fontWeight': 'bold'}),
            dcc.Slider(id='injection_duration', min=0, max=50, value=0, step=5,
                      marks={0: 'cont.', 10: '10', 25: '25', 50: '50'}),

            html.Label("Total Time [s]", style={'fontWeight': 'bold', 'marginTop': '8px'}),
            dcc.Slider(id='sim_time', min=20, max=200, value=80, step=20,
                      marks={20: '20', 50: '50', 100: '100', 200: '200'}),

            html.Br(),
            html.Button('RUN COMPARISON', id='run-btn', n_clicks=0,
                       style={'width': '100%', 'padding': '15px', 'fontSize': '16px',
                              'backgroundColor': '#27ae60', 'color': 'white',
                              'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                              'fontWeight': 'bold', 'marginTop': '10px'}),

            # Loading indicator
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id='loading-output', style={'height': '20px'}),
                style={'marginTop': '10px'}
            ),

            # Timing info
            html.Div(id='timing-info', style={'marginTop': '10px', 'padding': '10px',
                                               'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),

        ], style={'width': '280px', 'padding': '15px', 'backgroundColor': '#f8f9fa',
                  'borderRight': '2px solid #ddd', 'overflowY': 'auto'}),

        # Main content area - vertical layout (NN top, NS bottom)
        html.Div([
            # NN Plot (top)
            html.Div([
                html.Div([
                    html.H3("Neural Network (FNO)",
                           style={'color': '#3498db', 'margin': '0', 'flex': '1'}),
                    html.Div(id='nn-time', style={'fontSize': '13px',
                                                   'color': '#3498db', 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                          'padding': '0 10px'}),
                dcc.Graph(id='nn-plot', style={'height': '42vh'},
                          figure=create_empty_figure("FNO Prediction", "Click RUN to start")),
            ], style={'padding': '2px'}),

            # NS Plot (bottom)
            html.Div([
                html.Div([
                    html.H3("Numerical Solver (NS)",
                           style={'color': '#e74c3c', 'margin': '0', 'flex': '1'}),
                    html.Div(id='ns-time', style={'fontSize': '13px',
                                                   'color': '#e74c3c', 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                          'padding': '0 10px'}),
                dcc.Graph(id='ns-plot', style={'height': '42vh'},
                          figure=create_empty_figure("Numerical Solver", "Click RUN to start")),
            ], style={'padding': '2px'}),

            # Colorbar for concentration scale
            html.Div([
                dcc.Graph(id='colorbar-plot', style={'height': '80px'},
                          figure=create_colorbar_figure(),
                          config={'staticPlot': True})
            ], style={'padding': '2px'})
        ], style={'flex': '1', 'padding': '3px', 'overflowY': 'auto'})

    ], style={'display': 'flex', 'height': 'calc(100vh - 50px)'})
], style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'margin': '0', 'padding': '0'})


def create_contour_figure(data, x, y, title, times, frame_idx=0, show_colorbar=False):
    """Create animated contour figure with fixed 0-65% color scale."""
    # Convert to percentage
    data_pct = data * 100

    # Fixed color scale 0-65% for all plots
    zmin, zmax = 0, 65

    frames = []
    for i in range(len(times)):
        frames.append(go.Frame(
            data=[go.Contour(
                z=np.clip(data_pct[i], 0, 65),
                x=x, y=y,
                colorscale='Turbo',
                zmin=zmin, zmax=zmax,
                contours=dict(coloring='heatmap', showlines=False),
                ncontours=50,
                showscale=False,  # No colorbar on individual plots
            )],
            name=str(i)
        ))

    fig = go.Figure(
        data=[go.Contour(
            z=np.clip(data_pct[frame_idx], 0, 65),
            x=x, y=y,
            colorscale='Turbo',
            zmin=zmin, zmax=zmax,
            contours=dict(coloring='heatmap', showlines=False),
            ncontours=50,
            showscale=False,
        )],
        frames=frames
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=11)),
        xaxis=dict(title='x [m]', constrain='domain'),
        yaxis=dict(title='y [m]', scaleanchor='x', scaleratio=1),  # 1:1 aspect for axes (domain is 2:1)
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'y': 1.02, 'x': 1.0, 'xanchor': 'right',
            'buttons': [
                {
                    'label': '▶',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': '⏸',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                }
            ]
        }],
        sliders=[{
            'active': frame_idx,
            'pad': {'t': 40, 'b': 10},
            'len': 0.9, 'x': 0.05, 'y': -0.15,
            'currentvalue': {
                'prefix': 't = ',
                'suffix': ' s',
                'visible': True,
                'xanchor': 'center',
                'font': {'size': 11}
            },
            'steps': [
                {
                    'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f'{times[i]:.0f}',
                    'method': 'animate'
                }
                for i in range(len(times))
            ]
        }],
        margin=dict(l=50, r=20, t=40, b=80)
    )

    return fig


def create_colorbar_figure():
    """Create horizontal colorbar showing concentration scale 0-100%."""
    # Create a dummy heatmap just for the colorbar
    fig = go.Figure()

    # Add invisible heatmap to get colorbar
    fig.add_trace(go.Heatmap(
        z=[[0, 65]],
        x=[0, 65],
        y=[0],
        colorscale='Turbo',
        zmin=0, zmax=65,
        showscale=True,
        colorbar=dict(
            title=dict(text='Concentration c [%]', side='top'),
            orientation='h',
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle',
            len=0.9,
            thickness=25,
            tickvals=[0, 10, 20, 30, 40, 50, 60],
            ticktext=['0%', '10%', '20%', '30%', '40%', '50%', '60%'],
        ),
        hoverinfo='none',
    ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=10),
        height=80,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Hide the actual heatmap data
    fig.update_traces(visible=True)

    return fig


@callback(
    [Output('nn-plot', 'figure'), Output('ns-plot', 'figure'),
     Output('nn-time', 'children'), Output('ns-time', 'children'), Output('timing-info', 'children'),
     Output('loading-output', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('c_inlet', 'value'), State('Q_inlet', 'value'),
     State('gravity', 'value'), State('viscosity', 'value'),
     State('r_particle', 'value'), State('injection_duration', 'value'),
     State('sim_time', 'value')],
    prevent_initial_call=True
)
def run_comparison(n, c_inlet, Q_inlet, gravity, viscosity, r_particle, injection_duration, sim_time):
    global MODEL

    # Common parameters
    nx, ny = 60, 30
    Lx, Ly = 60.0, 30.0
    n_output_steps = 21
    dT = sim_time / (n_output_steps - 1)

    # Convert units
    mu0 = viscosity * 0.001  # mPa·s to Pa·s
    r_p = r_particle * 1e-6  # μm to m

    # ===== Run Numerical Solver (NS) =====
    t0_ns = time.perf_counter()

    # injection_duration=0 means continuous injection
    inj_dur = None if injection_duration == 0 else injection_duration

    solver = ProppantSolver(
        nx=nx, ny=ny,
        Lx=Lx, Ly=Ly,
        T=sim_time, dT=dT,
        c_inlet=c_inlet,
        Q_inlet=Q_inlet,
        g=gravity,
        mu0=mu0,
        r_particle=r_p,
        injection_duration=inj_dur,
        inlet_fraction=0.5,      # Half-height inlet
        inlet_position=0.5,      # Centered vertically
    )

    times_ns, traj_ns = solver.solve()
    time_ns = (time.perf_counter() - t0_ns) * 1000  # ms

    # ===== Run Neural Network (NN) =====
    nn_available = MODEL is not None
    time_nn = 0
    traj_nn = None

    if nn_available:
        t0_nn = time.perf_counter()

        # Prepare input parameters based on model's expected n_params
        n_params = MODEL.n_params

        # Get device from model
        device = next(MODEL.parameters()).device

        if n_params == 5:
            # Old 5-param model (legacy, untrained properly)
            params = torch.tensor([[
                c_inlet,
                Q_inlet * 20,
                gravity,
                mu0 * 100,
                r_p * 1000,
            ]], dtype=torch.float32, device=device)
        else:
            # New 9-param model - NORMALIZED to [0, 1] (must match dataset.py!)
            # injection_mode: 0=continuous, 1=single_pulse, 2=multi_pulse
            inj_mode = 0 if injection_duration == 0 else 1  # continuous or single_pulse

            params = torch.tensor([[
                c_inlet / 0.5,                    # c_inlet normalized [0.15-0.5] -> [0.3-1]
                Q_inlet / 0.1,                    # Q_inlet normalized [0.02-0.1] -> [0.2-1]
                gravity / 12.0,                   # g normalized [0-12] -> [0-1]
                mu0 / 0.01,                       # mu0 normalized [0.0005-0.01] -> [0.05-1]
                r_p / 0.0005,                     # r_particle normalized [0.0001-0.0005] -> [0.2-1]
                0.5,                              # inlet_fraction=0.5 (already normalized)
                (2 - 2) / 1.0,                    # rk_stages=2 normalized -> 0
                0.0 / 2.0,                        # lim_type (koren=0) -> 0
                inj_mode / 2.0,                   # injection_mode normalized -> [0-1]
            ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = MODEL(params)
            # Output: (1, n_times, nx, ny) -> (n_times, ny, nx)
            traj_nn_raw = pred[0].cpu().numpy().transpose(0, 2, 1)

            # Interpolate to match NS grid if sizes differ
            from scipy.ndimage import zoom
            if traj_nn_raw.shape[1:] != (ny, nx):
                zoom_factors = (1, ny / traj_nn_raw.shape[1], nx / traj_nn_raw.shape[2])
                traj_nn = zoom(traj_nn_raw, zoom_factors, order=1)
            else:
                traj_nn = traj_nn_raw

        time_nn = (time.perf_counter() - t0_nn) * 1000  # ms

    # ===== Create Figures =====
    x = solver.x
    y = solver.y

    # NS figure with fixed color scale 0-65%
    fig_ns = create_contour_figure(traj_ns, x, y, f"NS: c₀={c_inlet}, g={gravity}", times_ns)

    # NN figure
    if nn_available and traj_nn is not None:
        # Match time steps
        n_common = min(len(times_ns), traj_nn.shape[0])
        times_common = times_ns[:n_common]
        traj_nn_matched = traj_nn[:n_common]

        fig_nn = create_contour_figure(traj_nn_matched, x, y, "FNO Prediction", times_common)

        # Compute error for display
        errors = []
        for i in range(n_common):
            diff = np.linalg.norm(traj_nn_matched[i] - traj_ns[i])
            norm = np.linalg.norm(traj_ns[i]) + 1e-8
            errors.append(diff / norm * 100)

        avg_error = np.mean(errors)
        final_error = errors[-1] if errors else 0
    else:
        # No NN model - show placeholder
        fig_nn = create_empty_figure("FNO Prediction", "No trained model.<br>Run: python app.py --train")
        avg_error = 0
        final_error = 0

    # Timing info
    nn_time_text = f"NN: {time_nn:.1f} ms" if nn_available else "NN: Model not loaded"
    ns_time_text = f"NS: {time_ns:.0f} ms"

    speedup = time_ns / time_nn if nn_available and time_nn > 0 else 0

    timing_content = [
        html.B("Computation Time:"),
        html.Br(),
        html.Span(f"Neural Network: ", style={'color': '#3498db'}),
        html.Span(f"{time_nn:.1f} ms" if nn_available else "N/A"),
        html.Br(),
        html.Span(f"Numerical Solver: ", style={'color': '#e74c3c'}),
        html.Span(f"{time_ns:.0f} ms"),
        html.Hr(style={'margin': '8px 0'}),
        html.Span(f"Speedup: ", style={'fontWeight': 'bold'}),
        html.Span(f"{speedup:.0f}x" if speedup > 0 else "N/A",
                  style={'color': '#27ae60', 'fontWeight': 'bold'}),
    ]

    if nn_available:
        # Format error nicely
        if avg_error > 100:
            error_color = '#e74c3c'
            error_note = " (model mismatch!)"
        elif avg_error > 20:
            error_color = '#f39c12'
            error_note = ""
        else:
            error_color = '#27ae60'
            error_note = ""

        timing_content.extend([
            html.Hr(style={'margin': '8px 0'}),
            html.B("Quality (Rel. L2):"),
            html.Br(),
            html.Span(f"Avg: {min(avg_error, 999.9):.1f}%{error_note}",
                     style={'color': error_color}),
            html.Br(),
            html.Span(f"Final: {min(final_error, 999.9):.1f}%",
                     style={'color': error_color}),
        ])

        if avg_error > 100:
            timing_content.extend([
                html.Hr(style={'margin': '8px 0'}),
                html.Span("Model trained on different data!",
                         style={'color': '#e74c3c', 'fontSize': '11px'}),
            ])

    return fig_nn, fig_ns, nn_time_text, ns_time_text, timing_content, ""


def run_app(debug=True, port=8050):
    print(f"Server: http://localhost:{port}")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_app()
