"""
Proppant Transport Simulator - Injection from left boundary.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.proppant_transport import (
    SimulationParams,
    PhysicalParams,
    ProppantTransportSolver,
)

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

            html.Hr(),
            html.H4("Flow", style={'color': '#2c3e50'}),

            html.Label("Max Velocity [m/s]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='U_max', min=0.2, max=1.5, value=0.8, step=0.1,
                      marks={0.2: '0.2', 0.5: '0.5', 1.0: '1.0', 1.5: '1.5'}),

            html.Hr(),
            html.H4("Physics", style={'color': '#2c3e50'}),

            html.Label("Gravity [m/s²]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='gravity', min=0, max=15, value=9.81, step=0.5,
                      marks={0: '0', 5: '5', 10: '10', 15: '15'}),

            html.Label("Viscosity [mPa·s]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='viscosity', min=5, max=100, value=10, step=5,
                      marks={5: '5', 25: '25', 50: '50', 100: '100'}),

            html.Label("Particle Diameter [μm]", style={'fontWeight': 'bold'}),
            dcc.Slider(id='d_p', min=200, max=800, value=400, step=50,
                      marks={200: '200', 400: '400', 600: '600', 800: '800'}),

            html.Br(),
            html.Button('RUN SIMULATION', id='run-btn', n_clicks=0,
                       style={'width': '100%', 'padding': '18px', 'fontSize': '18px',
                              'backgroundColor': '#27ae60', 'color': 'white',
                              'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                              'fontWeight': 'bold', 'marginTop': '10px'}),

            html.Div(id='info', style={'marginTop': '15px', 'padding': '12px',
                                        'backgroundColor': '#ecf0f1', 'borderRadius': '5px',
                                        'fontSize': '14px'})

        ], style={'width': '300px', 'padding': '20px', 'backgroundColor': '#f8f9fa',
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
    [State('c_inlet', 'value'), State('U_max', 'value'),
     State('gravity', 'value'), State('viscosity', 'value'), State('d_p', 'value')],
    prevent_initial_call=True
)
def run_sim(n, c_inlet, U_max, gravity, viscosity, d_p):
    # Physical parameters
    phys = PhysicalParams(
        g=gravity,
        mu_f=viscosity * 0.001,  # mPa·s to Pa·s
        U_max=U_max,
        d_p=d_p * 1e-6  # μm to m
    )

    # Simulation parameters
    sim = SimulationParams(
        nx=120, ny=60,
        T=2.5, dt=0.002,
        save_every=20,
        c_inlet=c_inlet
    )

    solver = ProppantTransportSolver(sim, phys)

    # Solve (empty initial condition, inject from left)
    t0 = time.perf_counter()
    times, traj = solver.solve()
    elapsed = (time.perf_counter() - t0) * 1000

    # Color scale limits
    c_min, c_max = 0, min(0.7, traj.max() * 1.1)

    # Build frames
    frames = []
    for i in range(len(times)):
        frames.append(go.Frame(
            data=[go.Contour(
                z=traj[i].T,
                x=solver.x,
                y=solver.y,
                colorscale='Viridis',
                zmin=c_min, zmax=c_max,
                contours=dict(coloring='heatmap', showlines=False),
                showscale=(i == 0),
                colorbar=dict(title='c', titleside='right') if i == 0 else None
            )],
            name=str(i)
        ))

    # Initial figure
    fig = go.Figure(
        data=[go.Contour(
            z=traj[0].T,
            x=solver.x,
            y=solver.y,
            colorscale='Viridis',
            zmin=c_min, zmax=c_max,
            contours=dict(coloring='heatmap', showlines=False),
            colorbar=dict(title='c', titleside='right', thickness=15)
        )],
        frames=frames
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"Proppant Injection | c₀={c_inlet} | g={gravity} m/s² | μ={viscosity} mPa·s",
            x=0.5, font=dict(size=16)
        ),
        xaxis=dict(range=[0, 2], title='x [m]', dtick=0.5),
        yaxis=dict(range=[0, 1], title='y [m]', scaleanchor='x', scaleratio=0.5, dtick=0.2),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'y': 1.15, 'x': 0.5, 'xanchor': 'center',
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 20}
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
            'pad': {'t': 60, 'b': 10},
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
                    'label': f'{times[i]:.2f}',
                    'method': 'animate'
                }
                for i in range(len(times))
            ]
        }],
        margin=dict(l=60, r=30, t=80, b=80)
    )

    info_content = [
        html.Span(f"✓ Computed in {elapsed:.0f} ms", style={'color': '#27ae60', 'fontWeight': 'bold'}),
        html.Br(),
        html.Span(f"{len(times)} frames | {sim.nx}×{sim.ny} grid"),
        html.Br(),
        html.Span(f"c_max = {traj.max():.3f}")
    ]

    return fig, info_content


def run_app(debug=True, port=8050):
    print(f"Server: http://localhost:{port}")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    run_app()
