"""
Proppant Transport Simulator - NN vs NS Comparison UI.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
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
from src.model.fno_v2 import create_enhanced_model

app = dash.Dash(__name__, title="Proppant Simulator - NN vs NS")

# Domain size constants
DOMAIN_LX = 60.0
DOMAIN_LY = 30.0

# Try to load trained model
MODEL = None
MODEL_METADATA = None

def load_model():
    """Load trained FNO model if available. Tries v2 first, then v1."""
    global MODEL, MODEL_METADATA
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'metadata.json'

    # Try v2 first, then v1
    checkpoint_paths = [
        checkpoint_dir / 'best_v2.pt',
        checkpoint_dir / 'last_v2.pt',
        checkpoint_dir / 'best.pt',
        checkpoint_dir / 'last.pt',
    ]

    checkpoint_path = None
    for cp in checkpoint_paths:
        if cp.exists():
            checkpoint_path = cp
            break

    if checkpoint_path is None:
        print("No checkpoint found")
        return False

    try:
        import json
        if data_path.exists():
            with open(data_path) as f:
                MODEL_METADATA = json.load(f)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = checkpoint['model_state_dict']

        # Detect if it's v2 model (check first, needed for grid detection)
        # SpecBoost v2: has base_model.lift.0.weight
        # Plain v2: has lift.0.weight
        # v1: has lift.weight
        is_specboost = any('base_model.' in k for k in state_dict.keys())
        is_v2 = is_specboost or 'lift.0.weight' in state_dict
        prefix = 'base_model.' if is_specboost else ''

        # Detect model architecture from checkpoint weights
        grid_key = f'{prefix}grid_x'
        if grid_key in state_dict:
            grid_shape = state_dict[grid_key].shape
            nx = grid_shape[2]
            ny = grid_shape[3]
            print(f"Grid from checkpoint: nx={nx}, ny={ny}")
        else:
            nx = MODEL_METADATA.get('nx', 64) if MODEL_METADATA else 64
            ny = MODEL_METADATA.get('ny', 32) if MODEL_METADATA else 32
            print(f"Grid from metadata: nx={nx}, ny={ny}")

        if is_v2:
            # V2 model detection - handle both plain v2 and SpecBoost v2
            # (prefix already set above)

            # Detect dimensions
            lift_key = f'{prefix}lift.0.weight'
            if lift_key in state_dict:
                in_channels = state_dict[lift_key].shape[1]
                n_params = in_channels - 2
                width = state_dict[lift_key].shape[0]
            else:
                n_params = 9
                width = 192  # Default to xlarge

            # Detect n_times from project layer
            n_times = 26
            for key in state_dict:
                if 'project' in key and 'weight' in key and 'base_model' in key == is_specboost:
                    shape = state_dict[key].shape
                    if len(shape) >= 1 and shape[0] < 100:
                        n_times = shape[0]
                        break

            # Detect modes
            modes_key = f'{prefix}fno_blocks.0.spectral_conv.weights1'
            if modes_key in state_dict:
                modes1 = state_dict[modes_key].shape[2]
                modes2 = state_dict[modes_key].shape[3]
            else:
                modes1, modes2 = 32, 16

            # Detect model size from width
            if width >= 192:
                model_size = 'xlarge'
            elif width >= 128:
                model_size = 'large'
            elif width >= 96:
                model_size = 'medium'
            else:
                model_size = 'small'

            # Check for SpecBoost (has residual_modules)
            use_specboost = is_specboost

            print(f"Detected FNO v2: nx={nx}, ny={ny}, n_times={n_times}, n_params={n_params}")
            print(f"Architecture: {model_size}, width={width}, modes=({modes1},{modes2}), specboost={use_specboost}")
            print(f"Keys prefix: '{prefix}', sample: {[k for k in list(state_dict.keys())[:3]]}")

            MODEL = create_enhanced_model(
                nx=nx, ny=ny, n_times=n_times, n_params=n_params,
                device=device, model_size=model_size, use_specboost=use_specboost
            )
        else:
            # V1 model
            if 'lift.weight' in state_dict:
                in_channels = state_dict['lift.weight'].shape[1]
                n_params = in_channels - 2
                width = state_dict['lift.weight'].shape[0]
            else:
                n_params = 5
                width = 48

            if 'project.2.weight' in state_dict:
                n_times = state_dict['project.2.weight'].shape[0]
            else:
                n_times = 26

            if 'fno_blocks.0.spectral_conv.weights1' in state_dict:
                modes1 = state_dict['fno_blocks.0.spectral_conv.weights1'].shape[2]
                modes2 = state_dict['fno_blocks.0.spectral_conv.weights1'].shape[3]
            else:
                modes1, modes2 = 12, 8

            n_layers = sum(1 for k in state_dict if k.startswith('fno_blocks.') and k.endswith('.spectral_conv.weights1'))

            print(f"Detected FNO v1: nx={nx}, ny={ny}, n_times={n_times}, n_params={n_params}")
            print(f"Architecture: width={width}, modes=({modes1},{modes2}), layers={n_layers}")

            MODEL = create_model(
                nx=nx, ny=ny, n_times=n_times, n_params=n_params,
                modes1=modes1, modes2=modes2, width=width, n_layers=n_layers,
                device=device
            )

        # Load weights and check what was loaded
        model_keys = set(MODEL.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        matched = model_keys & ckpt_keys
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        if len(missing) > 0:
            print(f"WARNING: {len(missing)} missing keys (model has but checkpoint doesn't)")
            print(f"  Missing: {list(missing)[:3]}...")
        if len(unexpected) > 0:
            print(f"WARNING: {len(unexpected)} unexpected keys (checkpoint has but model doesn't)")
            print(f"  Unexpected: {list(unexpected)[:3]}...")

        if len(matched) < len(model_keys) * 0.5:
            print(f"ERROR: Only {len(matched)}/{len(model_keys)} keys matched! Model likely incompatible.")
            return False

        MODEL.load_state_dict(state_dict, strict=False)
        MODEL.eval()
        print(f"Loaded {len(matched)}/{len(model_keys)} weights from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        MODEL = None
    return False

load_model()

# ============== LAYOUT ==============
app.layout = html.Div([
    html.H1("Proppant Transport: Neural Network vs Numerical Solver",
            style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2c3e50',
                   'color': 'white', 'margin': '0', 'fontSize': '20px'}),

    html.Div([
        # Controls panel (left)
        html.Div([
            html.H4("Parameters", style={'color': '#2c3e50', 'marginTop': '0'}),

            html.Label("Inlet Concentration c₀", style={'fontWeight': 'bold'}),
            dcc.Input(id='c_inlet', type='number', min=0.15, max=0.40, value=0.30, step=0.01,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Label("Flow Rate Q [m²/s]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='Q_inlet', type='number', min=0.02, max=0.10, value=0.05, step=0.01,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Label("Gravity [m/s²]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='gravity', type='number', min=0, max=9.81, value=9.81, step=0.1,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Label("Viscosity μ₀ [mPa·s]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='viscosity', type='number', min=1, max=10, value=1, step=0.5,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Label("Particle Radius [μm]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='r_particle', type='number', min=100, max=300, value=200, step=10,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Hr(),
            html.H4("Simulation", style={'color': '#2c3e50'}),

            html.Label("Injection Duration [s]", style={'fontWeight': 'bold'}),
            dcc.Input(id='injection_duration', type='number', min=1, max=60, value=5, step=1,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Label("Total Time [s]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='sim_time', type='number', min=10, max=400, value=80, step=10,
                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),

            html.Br(),
            html.Button('RUN SIMULATION', id='run-btn', n_clicks=0,
                       style={'width': '100%', 'padding': '15px', 'fontSize': '16px',
                              'backgroundColor': '#27ae60', 'color': 'white',
                              'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                              'fontWeight': 'bold', 'marginTop': '10px'}),

            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id='loading-output', style={'height': '20px'}),
                style={'marginTop': '10px'}
            ),

            html.Div(id='timing-info', style={'marginTop': '10px', 'padding': '10px',
                                               'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),

        ], style={'width': '280px', 'padding': '15px', 'backgroundColor': '#f8f9fa',
                  'borderRight': '2px solid #ddd', 'overflowY': 'auto', 'flexShrink': '0'}),

        # Main content area
        html.Div([
            # NN Plot - Fixed 1200x600 px (2:1 aspect ratio)
            html.Div([
                html.Div([
                    html.H3("Neural Network (FNO)", style={'color': '#3498db', 'margin': '0', 'flex': '1'}),
                    html.Div(id='nn-time', style={'fontSize': '14px', 'color': '#3498db', 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 10px'}),
                dcc.Graph(id='nn-plot', config={'staticPlot': False}, style={'height': '600px', 'width': '1200px'})
            ]),

            # NS Plot - Fixed 1200x600 px (2:1 aspect ratio)
            html.Div([
                html.Div([
                    html.H3("Numerical Solver (NS)", style={'color': '#e74c3c', 'margin': '0', 'flex': '1'}),
                    html.Div(id='ns-time', style={'fontSize': '14px', 'color': '#e74c3c', 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 10px'}),
                dcc.Graph(id='ns-plot', config={'staticPlot': False}, style={'height': '600px', 'width': '1200px'})
            ]),

            # Horizontal Colorbar at the bottom
            html.Div([
                html.Div("Concentration c [%]", style={'textAlign': 'center', 'fontSize': '14px',
                                                        'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div(style={
                    'height': '25px',
                    'background': 'linear-gradient(to right, #30123b, #4662d7, #35aac8, #79d151, #faba39, #f66b19, #ca2a04)',
                    'borderRadius': '4px',
                    'margin': '0 50px',
                }),
                html.Div([
                    html.Span('0', style={'position': 'absolute', 'left': '50px'}),
                    html.Span('10', style={'position': 'absolute', 'left': 'calc(16.67% + 40px)'}),
                    html.Span('20', style={'position': 'absolute', 'left': 'calc(33.33% + 30px)'}),
                    html.Span('30', style={'position': 'absolute', 'left': 'calc(50% + 20px)'}),
                    html.Span('40', style={'position': 'absolute', 'left': 'calc(66.67% + 10px)'}),
                    html.Span('50', style={'position': 'absolute', 'left': 'calc(83.33%)'}),
                    html.Span('65', style={'position': 'absolute', 'right': '50px'}),
                ], style={'position': 'relative', 'height': '20px', 'fontSize': '12px', 'color': '#333'})
            ], style={'padding': '10px 0', 'marginTop': '10px'})

        ], style={'flex': '1', 'padding': '5px', 'overflowY': 'auto'})

    ], style={'display': 'flex', 'height': 'calc(100vh - 50px)'})
], style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'margin': '0', 'padding': '0'})


def create_heatmap_figure(data, x, y, title, times, frame_idx=0):
    """Create animated heatmap with FIXED axis ranges (30x60, aspect 1:2)."""
    data_pct = data * 100  # Convert to percentage

    frames = []
    for i in range(len(times)):
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=np.clip(data_pct[i], 0, 65),
                x=x, y=y,
                colorscale='Turbo',
                zmin=0, zmax=65,
                showscale=False,
            )],
            name=str(i)
        ))

    fig = go.Figure(
        data=[go.Heatmap(
            z=np.clip(data_pct[frame_idx], 0, 65),
            x=x, y=y,
            colorscale='Turbo',
            zmin=0, zmax=65,
            showscale=False,
        )],
        frames=frames
    )

    # Fixed plot dimensions: 1200px width, 600px height (2:1 aspect ratio)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        width=1200,  # Fixed width
        height=600,  # Fixed height
        xaxis=dict(
            title='x [m]',
            range=[0, DOMAIN_LX],  # FIXED 0-60
            autorange=False,  # Prevent auto-adjustment
            fixedrange=True,  # Prevent zoom/pan
            constrain='domain',
            showgrid=False,
            dtick=10,  # Tick every 10m
        ),
        yaxis=dict(
            title='y [m]',
            range=[0, DOMAIN_LY],  # FIXED 0-30
            autorange=False,  # Prevent auto-adjustment
            fixedrange=True,  # Prevent zoom/pan
            constrain='domain',
            showgrid=False,
            dtick=10,  # Tick every 10m
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'y': 1.0, 'x': 1.0, 'xanchor': 'right',
            'buttons': [
                {'label': '▶', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]},
                {'label': '⏸', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        sliders=[{
            'active': frame_idx,
            'pad': {'t': 30},
            'len': 0.9, 'x': 0.05,
            'currentvalue': {'prefix': 't = ', 'suffix': ' s', 'visible': True, 'xanchor': 'center'},
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                 'label': f'{times[i]:.0f}', 'method': 'animate'}
                for i in range(len(times))
            ]
        }],
        margin=dict(l=60, r=20, t=35, b=60)
    )

    return fig


def create_empty_figure(title, message):
    """Create empty placeholder figure with FIXED dimensions."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                      font=dict(size=14, color='#7f8c8d'))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        width=1200,  # Fixed width
        height=600,  # Fixed height
        xaxis=dict(title='x [m]', range=[0, DOMAIN_LX], autorange=False, fixedrange=True, showgrid=True, dtick=10),
        yaxis=dict(title='y [m]', range=[0, DOMAIN_LY], autorange=False, fixedrange=True, showgrid=True, dtick=10),
        margin=dict(l=60, r=20, t=35, b=60)
    )
    return fig


@callback(
    [Output('nn-plot', 'figure'), Output('ns-plot', 'figure'),
     Output('nn-time', 'children'), Output('ns-time', 'children'),
     Output('timing-info', 'children'), Output('loading-output', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('c_inlet', 'value'), State('Q_inlet', 'value'),
     State('gravity', 'value'), State('viscosity', 'value'),
     State('r_particle', 'value'), State('injection_duration', 'value'),
     State('sim_time', 'value')],
    prevent_initial_call=True
)
def run_comparison(n, c_inlet, Q_inlet, gravity, viscosity, r_particle, injection_duration, sim_time):
    global MODEL

    # Validate and set defaults for None values
    c_inlet = float(c_inlet) if c_inlet is not None else 0.35
    Q_inlet = float(Q_inlet) if Q_inlet is not None else 0.05
    gravity = float(gravity) if gravity is not None else 9.81
    viscosity = float(viscosity) if viscosity is not None else 1.0
    r_particle = float(r_particle) if r_particle is not None else 200.0
    injection_duration = float(injection_duration) if injection_duration is not None else 5.0
    sim_time = float(sim_time) if sim_time is not None else 80.0

    # Clamp to valid ranges (must match dataset.py param_ranges for stability)
    c_inlet = max(0.15, min(0.40, c_inlet))
    Q_inlet = max(0.02, min(0.10, Q_inlet))
    gravity = max(0, min(9.81, gravity))
    viscosity = max(1, min(10, viscosity))
    r_particle = max(100, min(300, r_particle))
    injection_duration = max(1, min(60, injection_duration))
    sim_time = max(10, min(400, sim_time))

    nx, ny = 60, 30
    Lx, Ly = 60.0, 30.0
    n_output_steps = 21
    dT = sim_time / (n_output_steps - 1)

    mu0 = viscosity * 0.001
    r_p = r_particle * 1e-6

    # ===== Numerical Solver =====
    t0_ns = time.perf_counter()

    solver = ProppantSolver(
        nx=nx, ny=ny,
        Lx=Lx, Ly=Ly,
        T=sim_time, dT=dT,
        c_inlet=c_inlet,
        Q_inlet=Q_inlet,
        g=gravity,
        mu0=mu0,
        r_particle=r_p,
        injection_duration=injection_duration,  # Now uses slider value (default 5s)
        inlet_fraction=0.5,
        inlet_position=0.5,
    )

    times_ns, traj_ns = solver.solve()
    time_ns = (time.perf_counter() - t0_ns) * 1000

    x = solver.x
    y = solver.y

    # NS figure
    fig_ns = create_heatmap_figure(traj_ns, x, y, f"NS: c₀={c_inlet}, g={gravity}", times_ns)

    # ===== Neural Network =====
    nn_available = MODEL is not None
    time_nn = 0
    avg_error = 0

    if nn_available:
        t0_nn = time.perf_counter()
        device = next(MODEL.parameters()).device
        n_params = MODEL.n_params

        if n_params == 5:
            params = torch.tensor([[c_inlet, Q_inlet * 20, gravity, mu0 * 100, r_p * 1000]],
                                  dtype=torch.float32, device=device)
        else:
            inj_mode = 1  # single_pulse
            # Normalization must match dataset.py param_ranges
            params = torch.tensor([[
                c_inlet / 0.4,       # c_inlet max=0.4
                Q_inlet / 0.1,       # Q_inlet max=0.1
                gravity / 9.81,      # g max=9.81
                mu0 / 0.01,          # mu0 max=0.01
                r_p / 0.0003,        # r_particle max=0.0003
                0.5,                 # inlet_fraction (fixed)
                1.0,                 # rk_stages=3 -> (3-2)/1=1.0 (fixed)
                0.0,                 # lim_type=koren -> 0/2=0.0 (fixed)
                inj_mode / 2.0,      # injection_mode
            ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = MODEL(params)
            print(f"[NN] Raw pred shape: {pred.shape}, range: [{pred.min().item():.4f}, {pred.max().item():.4f}]", flush=True)

            # pred shape: (batch, n_times, nx, ny) -> need (n_times, ny, nx)
            # Model outputs [0, 1] via Sigmoid, scale to concentration [0, 0.635]
            traj_nn_raw = pred[0].cpu().numpy().transpose(0, 2, 1) * 0.635

            print(f"[NN] After transpose & scale: {traj_nn_raw.shape}, range=[{traj_nn_raw.min():.4f}, {traj_nn_raw.max():.4f}]", flush=True)
            print(f"[NN] Timestep means: t0={traj_nn_raw[0].mean():.4f}, t_mid={traj_nn_raw[len(traj_nn_raw)//2].mean():.4f}, t_last={traj_nn_raw[-1].mean():.4f}", flush=True)

            # Clip to valid concentration range (safety)
            traj_nn_raw = np.clip(traj_nn_raw, 0, 0.635)

            from scipy.ndimage import zoom
            if traj_nn_raw.shape[1:] != (ny, nx):
                zoom_factors = (1, ny / traj_nn_raw.shape[1], nx / traj_nn_raw.shape[2])
                traj_nn = zoom(traj_nn_raw, zoom_factors, order=1)
            else:
                traj_nn = traj_nn_raw

        time_nn = (time.perf_counter() - t0_nn) * 1000

        n_common = min(len(times_ns), traj_nn.shape[0])
        traj_nn_matched = traj_nn[:n_common]
        times_common = times_ns[:n_common]

        fig_nn = create_heatmap_figure(traj_nn_matched, x, y, "FNO Prediction", times_common)

        errors = []
        for i in range(n_common):
            diff = np.linalg.norm(traj_nn_matched[i] - traj_ns[i])
            norm = np.linalg.norm(traj_ns[i]) + 1e-8
            errors.append(diff / norm * 100)
        avg_error = np.mean(errors)
    else:
        fig_nn = create_empty_figure("FNO Prediction", "No trained model.<br>Run: python app.py --train")

    nn_time_text = f"NN: {time_nn:.1f} ms" if nn_available else "NN: Model not loaded"
    ns_time_text = f"NS: {time_ns:.0f} ms"
    speedup = time_ns / time_nn if nn_available and time_nn > 0 else 0

    timing_content = [
        html.B("Computation Time:"), html.Br(),
        html.Span("Neural Network: ", style={'color': '#3498db'}),
        html.Span(f"{time_nn:.1f} ms" if nn_available else "N/A"), html.Br(),
        html.Span("Numerical Solver: ", style={'color': '#e74c3c'}),
        html.Span(f"{time_ns:.0f} ms"),
        html.Hr(style={'margin': '8px 0'}),
        html.Span("Speedup: ", style={'fontWeight': 'bold'}),
        html.Span(f"{speedup:.0f}x" if speedup > 0 else "N/A", style={'color': '#27ae60', 'fontWeight': 'bold'}),
    ]

    if nn_available:
        error_color = '#27ae60' if avg_error < 20 else '#f39c12' if avg_error < 50 else '#e74c3c'
        timing_content.extend([
            html.Hr(style={'margin': '8px 0'}),
            html.B("Avg Error: "),
            html.Span(f"{avg_error:.1f}%", style={'color': error_color, 'fontWeight': 'bold'}),
        ])

    return fig_nn, fig_ns, nn_time_text, ns_time_text, timing_content, ""


def run_app(debug=True, port=8050):
    print(f"Server: http://localhost:{port}")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_app()
