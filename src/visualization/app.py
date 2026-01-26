"""
Proppant Transport Simulator - Full physics with CPU solver.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.solver_wrapper import ProppantSolver

app = dash.Dash(__name__, title="Proppant Simulator")

app.layout = html.Div([
    html.H1("Proppant Transport Simulator",
            style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2c3e50',
                   'color': 'white', 'margin': '0'}),

    html.Div([
        # Controls panel
        html.Div([
            html.H4("Injection", style={'color': '#2c3e50'}),

            html.Label("Inlet Concentration c₀", style={'fontWeight': 'bold'}),
            dcc.Slider(id='c_inlet', min=0.1, max=0.5, value=0.35, step=0.05,
                      marks={0.1: '0.1', 0.25: '0.25', 0.4: '0.4', 0.5: '0.5'}),

            html.Label("Flow Rate Q [m²/s]", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='Q_inlet', min=0.02, max=0.1, value=0.05, step=0.01,
                      marks={0.02: '0.02', 0.05: '0.05', 0.1: '0.1'}),

            html.Hr(),
            html.H4("Physics", style={'color': '#2c3e50'}),

            html.Label("Gravity [m/s²]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='gravity', min=0, max=15, value=9.81, step=1,
                      marks={0: '0', 5: '5', 10: '10', 15: '15'}),

            html.Label("Viscosity μ₀ [mPa·s]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='viscosity', min=1, max=50, value=1, step=1,
                      marks={1: '1', 10: '10', 25: '25', 50: '50'}),

            html.Label("Particle Radius [μm]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='r_particle', min=100, max=500, value=200, step=50,
                      marks={100: '100', 200: '200', 300: '300', 500: '500'}),

            html.Hr(),
            html.H4("Simulation", style={'color': '#2c3e50'}),

            html.Label("Time [s]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='sim_time', min=20, max=200, value=80, step=20,
                      marks={20: '20', 50: '50', 100: '100', 200: '200'}),

            html.Br(),
            html.Button('RUN SIMULATION', id='run-btn', n_clicks=0,
                       style={'width': '100%', 'padding': '18px', 'fontSize': '18px',
                              'backgroundColor': '#27ae60', 'color': 'white',
                              'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                              'fontWeight': 'bold', 'marginTop': '10px'}),

            html.Div(id='info', style={'marginTop': '15px', 'padding': '12px',
                                        'backgroundColor': '#ecf0f1', 'borderRadius': '5px',
                                        'fontSize': '14px'})

        ], style={'width': '320px', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                  'borderRight': '2px solid #ddd', 'overflowY': 'auto'}),

        # Plot area
        html.Div([
            dcc.Graph(id='plot', style={'height': '88vh'})
        ], style={'flex': '1', 'padding': '10px'})

    ], style={'display': 'flex', 'height': 'calc(100vh - 60px)'})
], style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'margin': '0', 'padding': '0'})


@callback(
    [Output('plot', 'figure'), Output('info', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('c_inlet', 'value'), State('Q_inlet', 'value'),
     State('gravity', 'value'), State('viscosity', 'value'),
     State('r_particle', 'value'), State('sim_time', 'value')],
    prevent_initial_call=True
)
def run_sim(n, c_inlet, Q_inlet, gravity, viscosity, r_particle, sim_time):
    # Create solver
    t0 = time.perf_counter()

    solver = ProppantSolver(
        nx=60, ny=30,
        Lx=60.0, Ly=30.0,
        T=sim_time, dT=sim_time / 20,
        c_inlet=c_inlet,
        Q_inlet=Q_inlet,
        g=gravity,
        mu0=viscosity * 0.001,  # mPa·s to Pa·s
        r_particle=r_particle * 1e-6,  # μm to m
    )

    times, traj = solver.solve()
    elapsed = (time.perf_counter() - t0) * 1000

    # Color scale
    c_max = min(0.65, np.nanmax(traj) * 1.1)
    c_max = max(c_max, 0.1)

    # Build frames
    frames = []
    for i in range(len(times)):
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=np.clip(traj[i], 0, 0.65),
                x=solver.x,
                y=solver.y,
                colorscale='Viridis',
                zmin=0, zmax=c_max,
                showscale=(i == 0),
                colorbar=dict(title='c', titleside='right') if i == 0 else None
            )],
            name=str(i)
        ))

    # Initial figure
    fig = go.Figure(
        data=[go.Heatmap(
            z=np.clip(traj[0], 0, 0.65),
            x=solver.x,
            y=solver.y,
            colorscale='Viridis',
            zmin=0, zmax=c_max,
            colorbar=dict(title='c', titleside='right', thickness=15)
        )],
        frames=frames
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"Proppant Transport | c₀={c_inlet} | g={gravity} m/s² | μ={viscosity} mPa·s",
            x=0.5, font=dict(size=16)
        ),
        xaxis=dict(title='x [m]', scaleanchor='y'),
        yaxis=dict(title='y [m]'),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'y': 1.12, 'x': 0.5, 'xanchor': 'center',
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                },
                {
                    'label': '⟲ Reset',
                    'method': 'animate',
                    'args': [['0'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                }
            ]
        }],
        sliders=[{
            'active': 0,
            'pad': {'t': 50, 'b': 10},
            'len': 0.9, 'x': 0.05, 'y': 0,
            'currentvalue': {
                'prefix': 't = ',
                'suffix': ' s',
                'visible': True,
                'xanchor': 'center',
                'font': {'size': 14}
            },
            'steps': [
                {
                    'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f'{times[i]:.1f}',
                    'method': 'animate'
                }
                for i in range(len(times))
            ]
        }],
        margin=dict(l=60, r=30, t=80, b=80)
    )

    info_content = [
        html.Span(f"Computed in {elapsed/1000:.1f} s", style={'color': '#27ae60', 'fontWeight': 'bold'}),
        html.Br(),
        html.Span(f"{len(times)} frames | {solver.nx}x{solver.ny} grid"),
        html.Br(),
        html.Span(f"c_max = {np.nanmax(traj):.3f}")
    ]

    return fig, info_content


def run_app(debug=True, port=8050):
    print(f"Server: http://localhost:{port}")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_app()
