import json
import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import warnings
import torch
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.fno import create_model
from src.solver.generation import generate_for_params, load_generation_config
from src.solver.to_torch import build_torch_data
from src.training.dataset import build_processed_from_torch_data

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
CHECKPOINT_DIR = ROOT / 'checkpoints'
# Try adaptive first, then classic
CHECKPOINT_PATH_ADAPTIVE = CHECKPOINT_DIR / 'adaptive' / 'best.pt'
CHECKPOINT_PATH_CLASSIC = CHECKPOINT_DIR / 'classic' / 'best.pt'
CHECKPOINT_PATH = CHECKPOINT_PATH_ADAPTIVE if CHECKPOINT_PATH_ADAPTIVE.exists() else CHECKPOINT_PATH_CLASSIC
CSV_PATH = ROOT / 'simulation_results.csv'
CONFIG_PATH = ROOT / 'configs' / 'default.json'

MODEL = None
DATA_META = None
SAMPLE_FILES = []
DF_INDEX = None
GIF_CACHE = {}


def load_data_index():
    global DATA_META, SAMPLE_FILES
    meta_path = DATA_DIR / 'metadata.json'
    if not meta_path.exists():
        DATA_META = None
        SAMPLE_FILES = []
        return
    with open(meta_path) as f:
        DATA_META = json.load(f)
    SAMPLE_FILES = sorted(DATA_DIR.glob('sample_*.npz'))


def load_sim_index():
    global DF_INDEX
    if not CSV_PATH.exists():
        DF_INDEX = pd.DataFrame(columns=['c_in', 'w0', 'mu0', 'Q', 'chi', 'c_in_times', 'dT', 'timeseries_path'])
        return
    DF_INDEX = pd.read_csv(CSV_PATH)


def load_model():
    global MODEL
    try:
        # Try to find checkpoint in adaptive or classic subdirectories
        checkpoint_path = None
        smoothing_type = None

        # Try adaptive first
        if CHECKPOINT_PATH_ADAPTIVE.exists():
            checkpoint_path = CHECKPOINT_PATH_ADAPTIVE
            smoothing_type = 'adaptive'
        elif (CHECKPOINT_DIR / 'adaptive' / 'last.pt').exists():
            checkpoint_path = CHECKPOINT_DIR / 'adaptive' / 'last.pt'
            smoothing_type = 'adaptive'
            print(f"best.pt не найден в adaptive/, используем last.pt")
        # Try classic
        elif CHECKPOINT_PATH_CLASSIC.exists():
            checkpoint_path = CHECKPOINT_PATH_CLASSIC
            smoothing_type = 'classic'
        elif (CHECKPOINT_DIR / 'classic' / 'last.pt').exists():
            checkpoint_path = CHECKPOINT_DIR / 'classic' / 'last.pt'
            smoothing_type = 'classic'
            print(f"best.pt не найден в classic/, используем last.pt")
        # Fallback to old location (root checkpoints/)
        elif (CHECKPOINT_DIR / 'best.pt').exists():
            checkpoint_path = CHECKPOINT_DIR / 'best.pt'
            smoothing_type = 'classic'
            print(f"Используется старый формат чекпоинта из корня checkpoints/")
        elif (CHECKPOINT_DIR / 'last.pt').exists():
            checkpoint_path = CHECKPOINT_DIR / 'last.pt'
            smoothing_type = 'classic'
            print(f"Используется старый формат чекпоинта из корня checkpoints/")
        else:
            MODEL = None
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'base_model_state_dict' in checkpoint:
            state_dict = checkpoint['base_model_state_dict']
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)

        if 'grid_x' in state_dict:
            grid_shape = state_dict['grid_x'].shape
            nx = grid_shape[2]
            ny = grid_shape[3]
        else:
            nx = int(DATA_META.get('nx', 100)) if DATA_META else 100
            ny = int(DATA_META.get('ny', 100)) if DATA_META else 100

        if 'lift.0.weight' in state_dict:
            in_channels = state_dict['lift.0.weight'].shape[1]
            n_params = in_channels - 2
        elif 'lift.weight' in state_dict:
            in_channels = state_dict['lift.weight'].shape[1]
            n_params = in_channels - 2
        else:
            n_params = int(len(DATA_META.get('param_names', []))) if DATA_META else 7

        n_times = None
        if 'project.6.weight' in state_dict:
            n_times = state_dict['project.6.weight'].shape[0]
        elif 'project.5.weight' in state_dict:
            n_times = state_dict['project.5.weight'].shape[0]
        elif 'project.2.weight' in state_dict:
            n_times = state_dict['project.2.weight'].shape[0]
        elif 'spatial_smooth.kernel' in state_dict:
            n_times = state_dict['spatial_smooth.kernel'].shape[0]
        else:
            n_times = int(DATA_META.get('n_times', 201)) if DATA_META else 201

        # Detect if checkpoint uses adaptive smoothing
        # First check by directory, then by state_dict keys
        if smoothing_type:
            use_adaptive = (smoothing_type == 'adaptive')
        else:
            has_adaptive = 'adaptive_temporal_smooth.kernel' in state_dict
            has_old_smoothing = 'spatial_smooth.kernel' in state_dict or any('lowpass' in k for k in state_dict.keys())
            use_adaptive = has_adaptive and not has_old_smoothing

        MODEL = create_model(
            nx=nx, ny=ny, n_times=n_times, n_params=n_params, device=device,
            model_cfg={'use_error_corrector': False, 'use_adaptive_smoothing': use_adaptive}
        )

        # Filter incompatible weights
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if 'error_corrector' in k:
                continue
            if use_adaptive and ('spatial_smooth' in k or 'temporal_smooth' in k or 'lowpass' in k):
                continue
            if not use_adaptive and 'adaptive_temporal_smooth' in k:
                continue
            filtered_state_dict[k] = v

        MODEL.load_state_dict(filtered_state_dict, strict=False)
        MODEL.eval()
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        MODEL = None


def create_empty_figure(title, message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color='#7f8c8d')
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=11)),
        margin=dict(l=50, r=80, t=30, b=50)
    )
    return fig


def create_contour(z, x, y, title, zmax):
    fig = go.Figure(
        data=[go.Contour(
            z=np.clip(z, 0, zmax),
            x=x, y=y,
            colorscale='Turbo',
            zmin=0, zmax=zmax,
            contours=dict(coloring='heatmap', showlines=False),
            ncontours=60,
            colorbar=dict(title=dict(text='c', side='right'), thickness=12)
        )]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=11)),
        xaxis=dict(title='x', constrain='domain'),
        yaxis=dict(title='y', scaleanchor='x', scaleratio=1),
        margin=dict(l=45, r=20, t=35, b=35),
        autosize=True
    )
    return fig


load_data_index()
load_sim_index()
if DATA_META:
    DOMAIN_LX = float(DATA_META.get('L', 60.0))
    DOMAIN_LY = float(DATA_META.get('H', 60.0))
    NX = int(DATA_META.get('nx', 100))
    NY = int(DATA_META.get('ny', 100))
    NT = int(DATA_META.get('n_times', 1))
    CMAX = float(DATA_META.get('cmax', 0.635))
    DT_CONST = float(DATA_META.get('dT', 0.0))
    TMAX_CONST = float(DATA_META.get('Tmax', 0.0))
else:
    DOMAIN_LX = 60.0
    DOMAIN_LY = 60.0
    NX = 100
    NY = 100
    NT = 1
    CMAX = 0.635
    DT_CONST = 0.0
    TMAX_CONST = 0.0

load_model()

x_grid = np.linspace(0, DOMAIN_LX, NX, endpoint=False) + DOMAIN_LX / NX / 2
y_grid = np.linspace(0, DOMAIN_LY, NY, endpoint=False) + DOMAIN_LY / NY / 2


def _available_values(df: pd.DataFrame, name: str, selected: dict):
    if df is None or df.empty:
        return []
    d = df
    for k, v in selected.items():
        if v is not None and k in d.columns:
            d = d[d[k] == v]
    if d.empty or name not in d.columns:
        return []
    vals = sorted(d[name].unique().tolist())
    return vals


def _norm_params(raw: np.ndarray):
    if DATA_META is None:
        return raw.astype(np.float32)
    pmin = np.array(DATA_META.get('param_min', [0] * len(raw)), dtype=np.float32)
    pmax = np.array(DATA_META.get('param_max', [1] * len(raw)), dtype=np.float32)
    denom = pmax - pmin
    denom = np.where(denom == 0, 1.0, denom)
    x = (raw.astype(np.float32) - pmin) / denom
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


app = dash.Dash(__name__, title="Proppant Simulator")

app.layout = html.Div([
    html.H1("Proppant Transport: Model vs Dataset",
            style={'textAlign': 'center', 'padding': '12px', 'backgroundColor': '#2c3e50',
                   'color': 'white', 'margin': '0', 'fontSize': '22px'}),
    html.Div([
        html.Div([
            html.H4("Parameters", style={'color': '#2c3e50', 'marginTop': '0'}),
            html.Label("c_in"),
            dcc.Dropdown(id='c_in', options=[], value=None),
            html.Label("w0"),
            dcc.Dropdown(id='w0', options=[], value=None),
            html.Label("mu0"),
            dcc.Dropdown(id='mu0', options=[], value=None),
            html.Label("Q"),
            dcc.Dropdown(id='Q', options=[], value=None),
            html.Label("chi"),
            dcc.Dropdown(id='chi', options=[], value=None),
            html.Label("c_in_times"),
            dcc.Dropdown(id='c_in_times', options=[], value=None),
            html.Br(),
            html.Button('Generate', id='gen-btn', n_clicks=0,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '14px',
                              'backgroundColor': '#27ae60', 'color': 'white',
                              'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                               'fontWeight': 'bold'}),
            html.Label("GIF stride", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='gif_stride',
                min=1,
                max=20,
                value=4,
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Button('GIF', id='gif-btn', n_clicks=0,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '14px',
                               'backgroundColor': '#8e44ad', 'color': 'white',
                               'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                               'fontWeight': 'bold', 'marginTop': '8px'}),
            html.Br(),
            dcc.Checklist(
                id='show-gifs',
                options=[{'label': ' Show GIFs', 'value': 'show'}],
                value=[],
                style={'fontWeight': 'bold', 'marginTop': '10px'}
            ),
            html.Br(),
            html.Label("Time", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='time_idx',
                min=0,
                max=max(NT - 1, 0),
                value=0,
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(id='meta-info', style={'marginTop': '10px', 'padding': '10px',
                                               'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
        ], style={'width': '360px', 'padding': '15px', 'backgroundColor': '#f8f9fa',
                  'borderRight': '2px solid #ddd', 'overflowY': 'auto'}),
        html.Div([
            html.Div([
                html.H3("Model", style={'color': '#3498db', 'margin': '0'}),
                dcc.Graph(id='nn-plot', style={'height': '38vh'}),
                dcc.Loading(id='loading-gif-model', children=html.Div(id='gif-view-model', style={'padding': '10px', 'textAlign': 'center'}), type="circle"),
            ], style={'padding': '5px', 'borderBottom': '1px solid #eee'}),
            html.Div([
                html.H3("Solver", style={'color': '#e74c3c', 'margin': '0'}),
                dcc.Graph(id='gt-plot', style={'height': '38vh'}),
                dcc.Loading(id='loading-gif-solver', children=html.Div(id='gif-view-solver', style={'padding': '10px', 'textAlign': 'center'}), type="circle"),
            ], style={'padding': '5px'}),
        ], style={'flex': '1', 'padding': '10px', 'overflowY': 'auto', 'backgroundColor': 'white'})
    ], style={'display': 'flex', 'height': 'calc(100vh - 50px)'})
], style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'margin': '0', 'padding': '0'})


def _gif_bytes(conc: np.ndarray, vmax: float, stride: int, fps: int = 10) -> bytes:
    stride = max(1, int(stride))
    fps = max(1, int(fps))
    norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax), clip=True)
    cmap = cm.get_cmap('turbo')
    frames = []
    for i in range(0, conc.shape[0], stride):
        rgba = cmap(norm(conc[i]))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        frames.append(Image.fromarray(rgb, mode='RGB'))
    if not frames:
        return b""
    bio = BytesIO()
    duration = int(1000 / fps)
    frames[0].save(bio, format='GIF', save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return bio.getvalue()


@callback(
    [
        Output('c_in', 'options'), Output('c_in', 'value'),
        Output('w0', 'options'), Output('w0', 'value'),
        Output('mu0', 'options'), Output('mu0', 'value'),
        Output('Q', 'options'), Output('Q', 'value'),
        Output('chi', 'options'), Output('chi', 'value'),
        Output('c_in_times', 'options'), Output('c_in_times', 'value'),
        Output('time_idx', 'max'), Output('time_idx', 'value'),
        Output('nn-plot', 'figure'), Output('gt-plot', 'figure'),
        Output('meta-info', 'children'),
        Output('gif-view-model', 'children'),
        Output('gif-view-solver', 'children')
    ],
    [
        Input('c_in', 'value'),
        Input('w0', 'value'),
        Input('mu0', 'value'),
        Input('Q', 'value'),
        Input('chi', 'value'),
        Input('c_in_times', 'value'),
        Input('time_idx', 'value'),
        Input('gen-btn', 'n_clicks'),
        Input('gif-btn', 'n_clicks'),
        Input('gif_stride', 'value'),
        Input('show-gifs', 'value'),
    ],
    prevent_initial_call=False
)
def update_plots(c_in, w0, mu0, Q, chi, c_in_times, time_idx, n_clicks, gif_clicks, gif_stride, show_gifs):
    global DATA_META, SAMPLE_FILES, DF_INDEX, MODEL, NX, NY, NT, CMAX, DT_CONST, TMAX_CONST, x_grid, y_grid

    show_gif_flag = 'show' in (show_gifs or [])
    try:
        load_sim_index()
    except Exception:
        pass
    df = DF_INDEX

    dT_fixed = DT_CONST
    try:
        gen_cfg = load_generation_config(CONFIG_PATH, ROOT)
        dts = gen_cfg.get('params', {}).get('dT_values', [])
        if isinstance(dts, list) and len(dts) == 1:
            dT_fixed = float(dts[0])
    except Exception:
        pass
    if DF_INDEX is not None and not DF_INDEX.empty and 'dT' in DF_INDEX.columns:
        uniq_dt = sorted(DF_INDEX['dT'].dropna().unique().tolist())
        if len(uniq_dt) == 1:
            dT_fixed = float(uniq_dt[0])

    if ctx.triggered_id == 'gen-btn':
        if None not in (c_in, w0, mu0, Q, chi, c_in_times):
            params = (float(c_in), float(w0), float(mu0), float(Q), float(chi), float(c_in_times), float(dT_fixed))
            generate_for_params([params], project_root=ROOT, config_path=CONFIG_PATH)
            ts_one = ROOT / 'simulation_timeseries' / f"c{params[0]:.3f}_w{params[1]:.3f}_mu{params[2]:.3f}_Q{params[3]:.3f}_chi{params[4]:.1f}_t{int(params[5]):d}_dT{params[6]:.1f}_series.npz"
            torch_path = build_torch_data(ROOT / 'simulation_timeseries', ROOT / 'torch_data', max_files=None, sort_by_mtime=False, files=[ts_one])
            build_processed_from_torch_data(DATA_DIR, torch_path, max_samples=1, config_path=CONFIG_PATH)
            load_data_index()
            load_sim_index()
            load_model()
            if DATA_META:
                global DOMAIN_LX, DOMAIN_LY
                DOMAIN_LX = float(DATA_META.get('L', DOMAIN_LX))
                DOMAIN_LY = float(DATA_META.get('H', DOMAIN_LY))
                NX = int(DATA_META.get('nx', NX))
                NY = int(DATA_META.get('ny', NY))
                NT = int(DATA_META.get('n_times', NT))
                CMAX = float(DATA_META.get('cmax', CMAX))
                DT_CONST = float(DATA_META.get('dT', DT_CONST))
                TMAX_CONST = float(DATA_META.get('Tmax', TMAX_CONST))
            x_grid = np.linspace(0, DOMAIN_LX, NX, endpoint=False) + DOMAIN_LX / NX / 2
            y_grid = np.linspace(0, DOMAIN_LY, NY, endpoint=False) + DOMAIN_LY / NY / 2

    selected = {}

    vals_c = _available_values(df, 'c_in', selected)
    if c_in not in vals_c:
        c_in = vals_c[0] if vals_c else None
    selected['c_in'] = c_in

    vals_w = _available_values(df, 'w0', selected)
    if w0 not in vals_w:
        w0 = vals_w[0] if vals_w else None
    selected['w0'] = w0

    vals_mu = _available_values(df, 'mu0', selected)
    if mu0 not in vals_mu:
        mu0 = vals_mu[0] if vals_mu else None
    selected['mu0'] = mu0

    vals_Q = _available_values(df, 'Q', selected)
    if Q not in vals_Q:
        Q = vals_Q[0] if vals_Q else None
    selected['Q'] = Q

    vals_chi = _available_values(df, 'chi', selected)
    if chi not in vals_chi:
        chi = vals_chi[0] if vals_chi else None
    selected['chi'] = chi

    vals_t = _available_values(df, 'c_in_times', selected)
    if c_in_times not in vals_t:
        c_in_times = vals_t[0] if vals_t else None
    selected['c_in_times'] = c_in_times

    c_in_opts = [{'label': str(v), 'value': v} for v in vals_c]
    w0_opts = [{'label': str(v), 'value': v} for v in vals_w]
    mu0_opts = [{'label': str(v), 'value': v} for v in vals_mu]
    Q_opts = [{'label': str(v), 'value': v} for v in vals_Q]
    chi_opts = [{'label': str(v), 'value': v} for v in vals_chi]
    t_opts = [{'label': str(v), 'value': v} for v in vals_t]

    fig_empty = create_empty_figure("No data", "Run: python app.py --generate")
    fig_nn = create_empty_figure("Model", "No trained model. Run: python app.py --train")
    fig_gt = fig_empty
    gif_div_model = html.Div()
    gif_div_solver = html.Div()
    meta_lines = [
        html.Div(f"Nx={NX}, Ny={NY}"),
        html.Div(f"Tmax={TMAX_CONST:.0f}, dT={DT_CONST:.3f}"),
        html.Div(f"dT (generation) = {dT_fixed:g}"),
    ]

    if None in (c_in, w0, mu0, Q, chi, c_in_times) or df is None or df.empty:
        return (
            c_in_opts, c_in, w0_opts, w0, mu0_opts, mu0, Q_opts, Q, chi_opts, chi, t_opts, c_in_times,
            max(NT - 1, 0), 0,
            fig_nn, fig_gt, meta_lines, gif_div_model, gif_div_solver
        )

    row = df[
        (df['c_in'] == c_in) &
        (df['w0'] == w0) &
        (df['mu0'] == mu0) &
        (df['Q'] == Q) &
        (df['chi'] == chi) &
        (df['c_in_times'] == c_in_times) &
        (df['dT'] == dT_fixed)
    ]
    if row.empty:
        meta_lines.append(html.Hr(style={'margin': '8px 0'}))
        meta_lines.append(html.Div("No simulation. Click Generate."))
        return (
            c_in_opts, c_in, w0_opts, w0, mu0_opts, mu0, Q_opts, Q, chi_opts, chi, t_opts, c_in_times,
            max(NT - 1, 0), 0,
            fig_nn, fig_gt, meta_lines, gif_div_model, gif_div_solver
        )

    ts_path = str(row.iloc[0]['timeseries_path'])
    ts_file = ROOT / ts_path
    if not ts_file.exists():
        meta_lines.append(html.Hr(style={'margin': '8px 0'}))
        meta_lines.append(html.Div("Timeseries file missing. Click Generate."))
        return (
            c_in_opts, c_in, w0_opts, w0, mu0_opts, mu0, Q_opts, Q, chi_opts, chi, t_opts, c_in_times,
            max(NT - 1, 0), 0,
            fig_nn, fig_gt, meta_lines, gif_div_model, gif_div_solver
        )

    d = np.load(ts_file, allow_pickle=True)
    Q_series = d['Q'].astype(np.float32)
    t_series = d['times'].astype(np.float32)
    conc = Q_series / np.float32(w0)
    conc = np.clip(conc, 0.0, np.float32(CMAX))

    ny_data, nx_data = conc.shape[1], conc.shape[2]

    dx = DOMAIN_LX / nx_data
    dy = DOMAIN_LY / ny_data
    x_grid_data = (np.linspace(0, DOMAIN_LX, nx_data, endpoint=False) + dx/2).astype(np.float32)
    y_grid_data = (np.linspace(0, DOMAIN_LY, ny_data, endpoint=False) + dy/2).astype(np.float32)

    tmax_idx = int(conc.shape[0] - 1)
    if time_idx is None:
        time_idx = 0
    if int(time_idx) < 0:
        time_idx = 0
    if int(time_idx) > tmax_idx:
        time_idx = tmax_idx
    k = int(time_idx)

    gt_frame = conc[k]
    zmax = float(c_in)
    fig_gt = create_contour(gt_frame, x_grid_data, y_grid_data, f"Solver t={t_series[k]:.2f}", zmax)

    pred_full = None
    if MODEL is not None and int(getattr(MODEL, 'n_params', -1)) == 7 and DATA_META is not None:
        Q_internal = -abs(Q)
        raw = np.array([c_in, w0, mu0, Q_internal, chi, c_in_times, dT_fixed], dtype=np.float32)
        p = _norm_params(raw)
        device = next(MODEL.parameters()).device
        inp = torch.from_numpy(p.reshape(1, -1)).to(device).float()
        with torch.no_grad():
            pred = MODEL(inp)[0].detach().cpu().numpy()
        pred = pred.transpose(0, 2, 1).astype(np.float32)
        if pred.shape[1:] != conc.shape[1:]:
            pt = torch.from_numpy(pred).unsqueeze(0)
            pt = F.interpolate(pt, size=conc.shape[1:], mode='bilinear', align_corners=False)
            pred = pt[0].numpy().astype(np.float32)
        pred = pred * np.float32(CMAX)
        pred_full = pred
        n_common = min(pred.shape[0], conc.shape[0])
        pred = pred[:n_common]
        conc_c = conc[:n_common]
        nn_frame = pred[min(k, n_common - 1)]
        zmax_nn = float(c_in)
        fig_nn = create_contour(nn_frame, x_grid_data, y_grid_data, f"NN t={t_series[min(k, n_common - 1)]:.2f}", zmax_nn)

    meta_lines.extend([
        html.Hr(style={'margin': '8px 0'}),
        html.Div(f"c_in={c_in} w0={w0} mu0={mu0} Q={Q} chi={chi} c_in_times={c_in_times} dT={dT_fixed:g}"),
        html.Div(f"frame={k}/{tmax_idx}, t={t_series[k]:.2f}"),
    ])

    # GIFs logic
    gif_div_model = html.Div()
    gif_div_solver = html.Div()

    stride_val = int(gif_stride or 1)
    if stride_val <= 0: stride_val = 1

    def _get_gif_img(ts_path, source, stride, c_in_val):
        key = (str(ts_path), source, stride, float(c_in_val), float(CMAX))
        if key in GIF_CACHE and GIF_CACHE[key]:
            b64 = base64.b64encode(GIF_CACHE[key]).decode('ascii')
            href = f"data:image/gif;base64,{b64}"
            color = "#3498db" if source == 'model' else "#e74c3c"
            label = "Model" if source == 'model' else "Solver"
            return html.Div([
                html.A(f"Download {label} GIF", href=href, download=f"{source}.gif", target="_blank",
                       style={'display': 'inline-block', 'marginBottom': '5px', 'color': color, 'fontWeight': 'bold'}),
                html.Br(),
                html.Img(src=href, style={'width': '100%', 'maxWidth': '500px', 'height': 'auto', 'border': '1px solid #ddd', 'borderRadius': '8px'})
            ])
        return html.Div()

    if ctx.triggered_id == 'gif-btn':
        print(f"Generating GIF for {ts_file} with stride {stride_val}...")
        # Solver
        key_s = (str(ts_file), 'solver', stride_val, float(c_in), float(CMAX))
        try:
            arr_s = np.clip(conc, 0.0, np.float32(CMAX))
            GIF_CACHE[key_s] = _gif_bytes(arr_s, vmax=float(c_in), stride=stride_val, fps=10)
        except Exception as e:
            print(f"GIF solver error: {e}")
            GIF_CACHE[key_s] = b""

        # Model
        if pred_full is not None:
            key_m = (str(ts_file), 'model', stride_val, float(c_in), float(CMAX))
            try:
                arr_m = np.clip(pred_full, 0.0, np.float32(CMAX))
                GIF_CACHE[key_m] = _gif_bytes(arr_m, vmax=float(c_in), stride=stride_val, fps=10)
            except Exception as e:
                print(f"GIF model error: {e}")
                GIF_CACHE[key_m] = b""

    # Always try to pull from cache if it exists for current selection, but only if shown
    if show_gif_flag:
        gif_div_solver = _get_gif_img(ts_file, 'solver', stride_val, c_in)
        gif_div_model = _get_gif_img(ts_file, 'model', stride_val, c_in)
    else:
        gif_div_solver = html.Div()
        gif_div_model = html.Div()

    return (
        c_in_opts, c_in, w0_opts, w0, mu0_opts, mu0, Q_opts, Q, chi_opts, chi, t_opts, c_in_times,
        tmax_idx, k,
        fig_nn, fig_gt, meta_lines, gif_div_model, gif_div_solver
    )


def run_app(debug=True, port=8050):
    print(f"Server: http://localhost:{port}")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_app()
