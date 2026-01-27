import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re
import json
from io import BytesIO
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    HAS_CONTEXT = True
except ImportError:
    HAS_CONTEXT = False

def parse_filename(filename):
    pattern = r'c([\d.]+)_w([\d.]+)_mu([\d.]+)_Q([-\d.]+)_chi([\d.]+)_t(\d+)_dT([\d.]+)_series\.npz'
    match = re.match(pattern, filename)
    if match:
        return {
            'c_in': float(match.group(1)),
            'w0': float(match.group(2)),
            'mu0': float(match.group(3)),
            'Q': float(match.group(4)),
            'chi': float(match.group(5)),
            'c_in_times': int(match.group(6)),
            'dT': float(match.group(7))
        }
    return None

def get_available_simulations():
    project_root = Path(__file__).parent.parent.parent
    timeseries_dir = project_root / 'simulation_timeseries'
    if not timeseries_dir.exists():
        return []

    files = list(timeseries_dir.glob('*_series.npz'))
    simulations = []
    for file in files:
        params = parse_filename(file.name)
        if params:
            params['filename'] = file.name
            params['filepath'] = str(file)
            simulations.append(params)
    return simulations

def load_timeseries(filepath):
    data = np.load(filepath, allow_pickle=True)
    Q = data['Q']
    times = data['times']
    return Q, times

def create_contour_plot(Q_frame, w0, c_in, times, time_idx, L=60, H=60):
    nx = int(Q_frame.shape[1])
    ny = int(Q_frame.shape[0])
    x = np.linspace(0, L, nx, endpoint=False) + L / nx / 2
    y = np.linspace(0, H, ny, endpoint=False) + H / ny / 2

    w = w0 * np.ones_like(Q_frame)
    c = Q_frame / w

    fig = go.Figure(data=go.Contour(
        z=c,
        x=x,
        y=y,
        colorscale='Turbo',
        zmin=0,
        zmax=float(c_in),
        contours=dict(coloring='heatmap', showlines=False),
        colorbar=dict(title='c', titleside='right', thickness=15)
    ))

    fig.update_layout(
        title=dict(
            text=f'Time: {times[time_idx]:.2f}',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(title='x'),
        yaxis=dict(title='y', dtick=10, scaleanchor='x', scaleratio=1),
        margin=dict(l=50, r=30, t=70, b=60),
        height=600
    )

    return fig

def make_gif(conc: np.ndarray, vmax: float, fps: int = 10, stride: int = 1) -> bytes:
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

st.set_page_config(page_title="Time Series Visualizer", layout="wide")

st.title("Визуализация временных состояний")

project_root = Path(__file__).parent.parent.parent
timeseries_dir = project_root / 'simulation_timeseries'
if not timeseries_dir.exists():
    st.error("Папка simulation_timeseries не найдена")
    st.stop()

simulations = get_available_simulations()
if not simulations:
    st.error("Не найдено симуляций в папке simulation_timeseries")
    st.stop()

df_sims = pd.DataFrame(simulations)

cfg_path = project_root / 'configs' / 'default.json'
try:
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    grid_cfg = cfg.get('solver_generation', {}).get('grid', {})
    L_cfg = float(grid_cfg.get('L', 60.0))
    H_cfg = float(grid_cfg.get('H', 60.0))
except Exception:
    L_cfg = 60.0
    H_cfg = 60.0

st.sidebar.header("Выбор параметров")

def get_available_values(df, param_name, selected_params):
    filtered_df = df.copy()
    for key, value in selected_params.items():
        if value is not None:
            filtered_df = filtered_df[filtered_df[key] == value]
    return sorted(filtered_df[param_name].unique())

def get_selectbox_index(available_values, current_value):
    if current_value is not None and current_value in available_values:
        return available_values.index(current_value)
    return 0

def get_session_value(key, default=None):
    try:
        if key in st.session_state:
            return st.session_state[key]
        return default
    except (AttributeError, RuntimeError):
        return default

def init_session_state():
    try:
        if HAS_CONTEXT:
            ctx = get_script_run_ctx()
            if ctx is None:
                return

        if 'selected_c_in' not in st.session_state:
            st.session_state.selected_c_in = None
        if 'selected_w0' not in st.session_state:
            st.session_state.selected_w0 = None
        if 'selected_mu0' not in st.session_state:
            st.session_state.selected_mu0 = None
        if 'selected_Q' not in st.session_state:
            st.session_state.selected_Q = None
        if 'selected_chi' not in st.session_state:
            st.session_state.selected_chi = None
        if 'selected_c_in_times' not in st.session_state:
            st.session_state.selected_c_in_times = None
        if 'selected_dT' not in st.session_state:
            st.session_state.selected_dT = None
    except (AttributeError, RuntimeError, KeyError):
        pass

init_session_state()

selected_params = {}

available_c_in = get_available_values(df_sims, 'c_in', selected_params)
if not available_c_in:
    st.warning("Нет доступных симуляций")
    st.stop()
idx_c_in = get_selectbox_index(available_c_in, get_session_value('selected_c_in'))
selected_c_in = st.sidebar.selectbox("c_in", available_c_in, index=idx_c_in)
try:
    st.session_state.selected_c_in = selected_c_in
except (AttributeError, RuntimeError):
    pass
selected_params['c_in'] = selected_c_in

available_w0 = get_available_values(df_sims, 'w0', selected_params)
if not available_w0:
    st.warning("Нет доступных симуляций с выбранным c_in")
    st.stop()
idx_w0 = get_selectbox_index(available_w0, get_session_value('selected_w0'))
selected_w0 = st.sidebar.selectbox("w0", available_w0, index=idx_w0)
try:
    st.session_state.selected_w0 = selected_w0
except (AttributeError, RuntimeError):
    pass
selected_params['w0'] = selected_w0

available_mu0 = get_available_values(df_sims, 'mu0', selected_params)
if not available_mu0:
    st.warning("Нет доступных симуляций с выбранными параметрами")
    st.stop()
idx_mu0 = get_selectbox_index(available_mu0, get_session_value('selected_mu0'))
selected_mu0 = st.sidebar.selectbox("mu0", available_mu0, index=idx_mu0)
try:
    st.session_state.selected_mu0 = selected_mu0
except (AttributeError, RuntimeError):
    pass
selected_params['mu0'] = selected_mu0

available_Q = get_available_values(df_sims, 'Q', selected_params)
if not available_Q:
    st.warning("Нет доступных симуляций с выбранными параметрами")
    st.stop()
idx_Q = get_selectbox_index(available_Q, get_session_value('selected_Q'))
selected_Q = st.sidebar.selectbox("Q", available_Q, index=idx_Q)
try:
    st.session_state.selected_Q = selected_Q
except (AttributeError, RuntimeError):
    pass
selected_params['Q'] = selected_Q

available_chi = get_available_values(df_sims, 'chi', selected_params)
if not available_chi:
    st.warning("Нет доступных симуляций с выбранными параметрами")
    st.stop()
idx_chi = get_selectbox_index(available_chi, get_session_value('selected_chi'))
selected_chi = st.sidebar.selectbox("chi", available_chi, index=idx_chi)
try:
    st.session_state.selected_chi = selected_chi
except (AttributeError, RuntimeError):
    pass
selected_params['chi'] = selected_chi

available_c_in_times = get_available_values(df_sims, 'c_in_times', selected_params)
if not available_c_in_times:
    st.warning("Нет доступных симуляций с выбранными параметрами")
    st.stop()
idx_c_in_times = get_selectbox_index(available_c_in_times, get_session_value('selected_c_in_times'))
selected_c_in_times = st.sidebar.selectbox("c_in_times", available_c_in_times, index=idx_c_in_times)
try:
    st.session_state.selected_c_in_times = selected_c_in_times
except (AttributeError, RuntimeError):
    pass
selected_params['c_in_times'] = selected_c_in_times

available_dT = get_available_values(df_sims, 'dT', selected_params)
if not available_dT:
    st.warning("Нет доступных симуляций с выбранными параметрами")
    st.stop()
idx_dT = get_selectbox_index(available_dT, get_session_value('selected_dT'))
selected_dT = st.sidebar.selectbox("dT", available_dT, index=idx_dT)
try:
    st.session_state.selected_dT = selected_dT
except (AttributeError, RuntimeError):
    pass
selected_params['dT'] = selected_dT

filtered = df_sims[
    (df_sims['c_in'] == selected_c_in) &
    (df_sims['w0'] == selected_w0) &
    (df_sims['mu0'] == selected_mu0) &
    (df_sims['Q'] == selected_Q) &
    (df_sims['chi'] == selected_chi) &
    (df_sims['c_in_times'] == selected_c_in_times) &
    (df_sims['dT'] == selected_dT)
]

if filtered.empty:
    st.warning("Нет симуляций с выбранными параметрами")
    st.stop()

selected_sim = filtered.iloc[0]
filepath = selected_sim['filepath']

try:
    Q, times = load_timeseries(filepath)

    st.sidebar.header("Выбор момента времени")

    max_time_idx = len(times) - 1
    time_value = st.sidebar.slider(
        "Время",
        min_value=float(times[0]),
        max_value=float(times[-1]),
        value=float(times[0]),
        step=(float(times[-1]) - float(times[0])) / max(1, max_time_idx)
    )

    closest_idx = np.argmin(np.abs(times - time_value))

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_contour_plot(Q[closest_idx], selected_w0, selected_c_in, times, closest_idx, L=L_cfg, H=H_cfg)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Информация")
        st.write(f"**Параметры:**")
        st.write(f"- c_in: {selected_c_in}")
        st.write(f"- w0: {selected_w0}")
        st.write(f"- mu0: {selected_mu0}")
        st.write(f"- Q: {selected_Q}")
        st.write(f"- chi: {selected_chi}")
        st.write(f"- c_in_times: {selected_c_in_times}")
        st.write(f"- dT: {selected_dT}")

        st.write(f"**Данные:**")
        st.write(f"- Кадров: {len(times)}")
        st.write(f"- Размер: {Q.shape[1]}×{Q.shape[2]}")
        st.write(f"- Время: {times[0]:.2f} - {times[-1]:.2f}")

        st.write(f"**Текущий кадр:**")
        st.write(f"- Индекс: {closest_idx}")
        st.write(f"- Время: {times[closest_idx]:.2f}")
        w = selected_w0 * np.ones_like(Q[closest_idx])
        c = Q[closest_idx] / w
        st.write(f"- c min: {c.min():.6f}")
        st.write(f"- c max: {c.max():.6f}")
        st.write(f"- c mean: {c.mean():.6f}")

        st.subheader("GIF")
        fps = st.slider("fps", min_value=1, max_value=30, value=10, step=1)
        stride = st.slider("stride", min_value=1, max_value=max(1, len(times) // 20), value=max(1, len(times) // 50), step=1)
        if st.button("Сгенерировать GIF"):
            conc_all = (Q / np.float32(selected_w0)).astype(np.float32)
            conc_all = np.clip(conc_all, 0.0, np.float32(selected_c_in))
            gif_bytes = make_gif(conc_all, vmax=float(selected_c_in), fps=int(fps), stride=int(stride))
            if gif_bytes:
                st.image(gif_bytes)
                st.download_button("Скачать GIF", data=gif_bytes, file_name="simulation.gif", mime="image/gif")

except Exception as e:
    st.error(f"Ошибка при загрузке данных: {str(e)}")

