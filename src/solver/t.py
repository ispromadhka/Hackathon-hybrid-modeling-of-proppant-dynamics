c_min, c_max, c_step = 0.05, 0.45, 0.05
w_min, w_max, w_step = 0.01, 0.03, 0.005
mu_min, mu_max, mu_step = 0.001, 0.1, 0.005
q_min, q_max, q_step = 0.01, 0.5, 0.05
chi_min, chi_max, chi_step = 15, 30, 3
time_min, time_max, time_step = 50, 200, 25
dT_values = [2]

c_in = np.arange(c_min, c_max + c_step/2, c_step)
w0 = np.arange(w_min, w_max + w_step/2, w_step)
mu0 = np.arange(mu_min, mu_max + mu_step/2, mu_step)
Q = -np.arange(q_min, q_max + q_step/2, q_step)
chi = np.arange(chi_min, chi_max + chi_step/2, chi_step)
c_in_times_list = np.arange(time_min, time_max + time_step/2, time_step)

param_combinations = list(itertools.product(c_in, w0, mu0, Q, chi, c_in_times_list, dT_values))