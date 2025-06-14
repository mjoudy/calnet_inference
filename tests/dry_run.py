import numpy as np
import zarr
import os

os.makedirs("data", exist_ok=True)

def make_spikes(path="data/spikes.zarr"):
    spikes = np.random.poisson(0.1, size=(5, 100))
    zarr.save(path, spikes)

def make_conn_matrix(path="data/conn_matrix.npy"):
    conn = np.random.randn(5, 5)
    np.fill_diagonal(conn, 0)
    np.save(path, conn)

if __name__ == "__main__":
    make_spikes()
    make_conn_matrix()
    print("✅ Dummy input data created in 'data/' folder.")
