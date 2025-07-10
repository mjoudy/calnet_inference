import numpy as np
from mlflow_tracking import init_mlflow, log_step_metrics

def simulate_data(*args, **kwargs):
    init_mlflow()
    # For demonstration, simulate random data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    log_step_metrics('simulate_data', {'X_shape': X.shape, 'y_shape': y.shape})
    return X, y

def sim_calcium(spikes, tau=100):
    N, total_dur = spikes.shape
    wup_time = 100
    spikes = spikes[:, wup_time:]
    sim_dur = spikes.shape[1]

    dt = 1
    const_A = np.exp((-1 / tau) * dt)
    calcium_nsp_noisy = np.zeros((N, sim_dur))

    noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
    spikes_noisy = spikes + noise_intra
    calcium_nsp_noisy[:, 0] = spikes_noisy[:, 0]

    for t in range(1, sim_dur):
        calcium_nsp_noisy[:, t] = const_A * calcium_nsp_noisy[:, t - 1] + spikes_noisy[:, t]

    noise_recording = np.random.normal(0, 1, calcium_nsp_noisy.shape)
    calcium_nsp_noisy += noise_recording
    log_step_metrics('sim_calcium', {'N': N, 'sim_dur': sim_dur, 'tau': tau})
    return calcium_nsp_noisy
