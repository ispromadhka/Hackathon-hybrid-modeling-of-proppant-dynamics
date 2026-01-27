import numpy as np
def load_timeseries(filepath):
    data = np.load(filepath, allow_pickle=True)
    Q = data['Q']
    times = data['times']
    return Q, times

path = r'C:\Users\OneVay\Documents\hack\Hackathon-hybrid-modeling-of-proppant-dynamics\simulation_timeseries\c0.050_w0.010_mu0.001_Q-0.010_chi15.0_t50_dT2.0_series.npz'
f = load_timeseries(path)
print(f)