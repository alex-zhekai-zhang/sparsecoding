import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so SNN.config can be imported when running this file directly
import SNN.config as config

T = 20000    # steps: Du et al. run 200 time units at dt = 0.01
dt = 0.01
def spiking_LCA(input, lamby, Phi, T, dt=dt, record=True):   # port of Du et al.'s SLCA_rk2 (XuexingDuSLCA/Neuron_models.py); assumes unit-norm atoms
    Phi = torch.tensor(Phi, dtype=torch.float32)
    x = torch.tensor(input, dtype=torch.float32)
    b = Phi.T @ x                    # constant bias current: no input spikes
    w = Phi.T @ Phi
    w.fill_diagonal_(0)              # Phi^T Phi - I for unit-norm atoms
    decay = 1 - dt + dt ** 2 / 2     # rk2 step of dg/dt = -g (tau = 1)
    V = torch.zeros_like(b)
    g = torch.zeros_like(b)          # synaptic current
    cross = torch.zeros_like(b)      # last step's spikes, weighted exp(t_cross - dt)
    count = torch.zeros_like(b)
    spikes, n_out, start = [], 0, time.perf_counter()
    for t in range(T):
        g = g * decay - w @ cross    # synapse updates first, from last step's spikes (as in SLCA_rk2.update)
        drive = b + g - lamby
        V_new = V + dt * drive
        s = V_new >= 1.0
        dt_cross = torch.where(s, dt * (1 - V) / (V_new - V), 0.)   # time of threshold crossing within the step
        cross = torch.where(s, torch.exp(dt_cross - dt), 0.)
        V = torch.where(s, -(dt_cross - dt) * drive, V_new)          # reset to the drive left after crossing
        count = count + s.float()
        n_out = n_out + s.sum()
        if record:
            spikes += [s]
    P = x.shape[1] if x.dim() > 1 else 1
    print(f"du: {T} steps (dt={dt:.3g}, {T * dt:.0f} time units), {n_out.item() / P:.0f} recurrent spikes/patch, 0 input spikes/patch, {time.perf_counter() - start:.1f}s")
    a = count / (T * dt)             # spikes per unit time over the whole run, from t = 0
    if not record:
        return a
    return a, torch.stack(spikes)

if __name__ == "__main__":
    Phi = config.Phi
    image = config.image - config.image.mean()
    a, spikes = spiking_LCA(image, config.lamby, Phi, T)
    print(a)
    plt.imshow(spikes.T, aspect='auto', interpolation='nearest')
    plt.show()
