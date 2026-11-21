import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so SNN.config can be imported when running this file directly
import SNN.config as config

T = 10240
f = 0.5   # decode the last f*T steps
def spiking_LCA(input, lamby, Phi, lr, epsilon, T, num_bases, record=True, dt=None, live=True, S=512):   # T = number of steps, dt = simulated time per step; live=False is snapshot mode, in cycles of S steps
    Phi = torch.tensor(Phi, dtype=torch.float32)
    x = torch.tensor(input, dtype=torch.float32)   
    x = torch.cat([torch.clamp(x, min=0), torch.clamp(-x, min=0)])   #On Off Split
    I = torch.eye(num_bases)
    feedforward = lr * torch.cat([Phi, -Phi]).T * epsilon  #On Off split
    recurrent = (I - lr * (Phi.T @ Phi)) * epsilon         
    theta = 1.0                             
    dt = 1.0 / (2.5 * np.abs(input).max())   # same range rule as latency_SLCA: max rate 1/dt = 2.5x the largest ON/OFF input, so large coefficients are not clipped
    u = torch.zeros_like(feedforward @ x)   # same shape as the input current, so x can be one patch or a matrix of patches (one per column)
    v = torch.zeros_like(u)
    v_in = torch.zeros_like(x)              # membrane of the ON/OFF input cells

    vs, spikes = [], []
    count = 0
    discard = int(T * (1 - f))
    s = v >= theta
    start, n_out, n_in = time.perf_counter(), 0, 0   # time and spike counts, for comparing against latency_SLCA
    if not live:   # snapshot mode (option A): one ISTA step per cycle of S steps; each neuron bursts a/q spikes (at most 1 per step), each worth q
        q = 1 / (dt * S)   # value of one spike; S spikes = 1/dt = 2.5x the largest input, the same range and grid as latency_SLCA
        #n_x = torch.clamp(torch.round(x / q), max=S)   # input cells send x rounded to the grid, as latency's input slots do
        cnt = torch.zeros_like(u)
        for c in range(T // S):
            u_prev = u
            u = (1 - epsilon) * u   # per-cycle leak
            #v = torch.clamp(v, min=0) + u_prev - lr * lamby   # leftover + (u - lambda); negative drive is not carried over, as in latency
            v = torch.clamp(v, min=0)   # leftover; negative drive is not carried over, as in latency
            drive = (u_prev - lr * lamby) / S   # constant input current: over the cycle the neuron integrates u - lambda once
            v_in = torch.full_like(x, q / 2)   # input cells start at half a spike, so their count rounds x to the grid, as latency's input slots do
            cnt = torch.zeros_like(u)
            for t in range(S):
                #s = v >= q   # deprecated
                v = v + drive
                s = v >= q   # IF neuron at a constant rate: spikes spread evenly over the cycle, at most 1 per step
                v = v - s.float() * q
                cnt = cnt + s.float()
                #s_in = (n_x > t).float()   # deprecated
                v_in = v_in + x / S   # input cells: constant current x/S
                s_in = (v_in >= q).float()
                v_in = v_in - s_in * q
                n_out, n_in = n_out + s.sum(), n_in + s_in.sum()
                u = u + (recurrent @ s.float() + feedforward @ s_in) * q   # v only sees u at the start of the next cycle
                if record:
                    vs += [v]; spikes += [s]
        print(f"rate snapshot: {S * (T // S)} steps (S={S}, q={q:.3g}), {n_out.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} recurrent spikes/patch, {n_in.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} input spikes/patch, {time.perf_counter() - start:.1f}s")
        if not record:
            return cnt * q   # decode the final cycle
        spikes = torch.stack(spikes)
        return spikes[-S:].float().sum(0) * q, torch.stack(vs), spikes
    for t in range(T):
        v_in = v_in + x * dt                   # IF neurons firing at rate x
        s_in = v_in >= 1.0
        v_in = v_in - s_in.float()
        u = recurrent @ s.float() + feedforward @ s_in.float() + (1 - epsilon * dt) * u   # leaky integrator: LCA.py's u, with spikes in place of a and x
        v = v + (u - lamby * lr) * dt          # IF neuron: integrate u - lambda
        s = v >= theta #spikes
        v = v - (s).float() * theta #reset
        n_out, n_in = n_out + s.sum(), n_in + s_in.sum()
        if record:
            vs += [v]; spikes += [s]
        elif t >= discard:
            count = count + s.float()   # count the last f*T steps
    print(f"rate: {T} steps (dt={dt:.3g}, {T * dt:.0f} time units), {n_out.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} recurrent spikes/patch, {n_in.item() / (x.shape[1] if x.dim() > 1 else 1):.0f} input spikes/patch, {time.perf_counter() - start:.1f}s")
    if not record:
        return count / ((T - discard) * dt)   # spikes per unit time over the last f*T steps
    spikes = torch.stack(spikes)
    a = spikes[discard:].float().mean(0) / dt
    return a, torch.stack(vs), spikes

if __name__ == "__main__":
    Phi = config.Phi
    image = config.image - config.image.mean()
    a, vs, spikes = spiking_LCA(image, config.lamby, Phi, config.lr, config.epsilon, T, Phi.shape[1])
    print(a)
    plt.imshow(spikes.T, aspect='auto', interpolation='nearest')
    plt.show()
