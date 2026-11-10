# Sparse Coding for Spiking Neural networks

See [`Alternative_Coding_Methods.pdf`](Alternative_Coding_Methods.pdf) for formulations.

To visualize Phi, run ```python SNN/config.py n``` for an index n in Phi, or just run ```python SNN/config.py``` to see all of it.
---

# Spiking LCA comparison on Kodak

Non-negative classical LCA (`nonnegative_lca`), rate-coded (`spiking_lca`), latency-coded (`latency_slca`) and temporal-pattern-coded (`temporal_pattern_slca`) spiking LCA, run on all 24 Kodak images with whitening preprocessing (`--preprocess whiten`).

## Setup

| | |
|---|---|
| Images | Kodak `kodim01`–`kodim24`, grayscale, downscaled to 512 px on the long side (512×336 or 336×512, 2688 patches each) |
| Whitening | Whole image filtered in Fourier space by R(f) = f·exp(−(f/f₀)⁴), f₀ = 0.4 cycles/pixel, f = √(fx² + fy²) (`np.fft.fftfreq` units), then cut into 8×8 patches and scaled to unit variance per image |
| Dictionary | `SNN/phi_learned_whiten.npz`: 64×128 (single, not doubled), learned with non-negative ISTA (λ = 0.1, η = 0.1, 10,000 iterations, batch 256, 200 ISTA steps) on 100k random patches of whitened Kodak images (f₀ = 0.4), columns at norm 1 |
| Parameters | λ = 0.1, ε = 0.1, lr = 1/‖Φ‖₂² = 0.188 (`SNN/config.py`) |
| Non-negative LCA | 400 iterations, a = max(u − λ·lr, 0) (`classical/nonnegative_LCA.py`) |
| Rate | 204,800 steps, dt = 1 / max\|x\| per image |
| Latency | 400 cycles × 512 steps = 204,800 steps, T = max\|x\| per image, leftover v carried across cycles (see Rounding error) |
| Temporal pattern | 400 cycles × 9 bits = 3,600 steps, same T, so q = T/512, the same grid as latency, leftover v carried across cycles |

Metric: SNR = 10·log10(mean(W²) / MSE), where W is the whitened image and MSE = mean((Ŵ − W)²), with Ŵ the reconstruction Φa put back together and the scale undone, before unwhitening. "Active" is the number of nonzero coefficients per patch, out of 128.

## Summary (mean ± std over 24 images)

| Method | SNR (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch | Steps | Wall-clock / image |
|---|---:|---:|---:|---:|---:|---:|
| Non-negative LCA | 21.14 ± 0.74 | 37.3 ± 5.1 | — | — | 400 | — |
| Rate | 21.38 ± 0.71 | 40.9 ± 5.2 | 439,996 | 860,211 | 204,800 | 440 s |
| Latency | 20.59 ± 0.75 | 35.9 ± 5.4 | 15,452 | 25,011 | 204,800 | 541 s |
| Temporal pattern | 20.99 ± 0.73 | 35.9 ± 5.4 | 37,794 | 65,999 | 3,600 | 6.2 s |

Wall-clock was measured with 6 jobs running at once, 3 threads each. Earlier results are in `legacy_tests/`.

## Per-image SNR (dB) and active coefficients per patch

| Image | LCA | Rate | Latency | TP | Active (LCA) | Active (rate) | Active (lat / TP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| kodim01 | 20.63 | 20.70 | 19.00 | 20.28 | 42.8 | 45.7 | 42.1 / 42.2 |
| kodim02 | 20.25 | 20.60 | 20.14 | 20.18 | 40.7 | 45.2 | 38.6 / 38.5 |
| kodim03 | 21.43 | 21.60 | 20.93 | 21.20 | 33.5 | 37.2 | 31.8 / 31.8 |
| kodim04 | 20.98 | 21.21 | 20.66 | 20.89 | 39.5 | 44.0 | 37.9 / 37.8 |
| kodim05 | 20.47 | 20.61 | 20.11 | 20.28 | 43.5 | 47.0 | 42.6 / 42.6 |
| kodim06 | 20.73 | 21.12 | 20.58 | 20.69 | 38.5 | 42.3 | 37.4 / 37.4 |
| kodim07 | 22.34 | 22.52 | 22.29 | 22.30 | 32.4 | 36.1 | 31.2 / 31.1 |
| kodim08 | 21.17 | 21.08 | 19.41 | 20.64 | 39.6 | 42.0 | 38.8 / 38.9 |
| kodim09 | 22.33 | 22.45 | 21.07 | 22.05 | 29.9 | 33.0 | 28.0 / 28.0 |
| kodim10 | 21.26 | 21.58 | 21.20 | 21.21 | 34.9 | 38.9 | 32.7 / 32.7 |
| kodim11 | 20.82 | 21.18 | 20.63 | 20.77 | 38.1 | 42.0 | 36.7 / 36.8 |
| kodim12 | 21.62 | 21.83 | 20.96 | 21.48 | 35.4 | 39.1 | 33.6 / 33.6 |
| kodim13 | 19.55 | 19.95 | 19.50 | 19.51 | 46.6 | 49.4 | 45.9 / 45.8 |
| kodim14 | 20.53 | 20.76 | 20.35 | 20.44 | 44.1 | 48.0 | 43.0 / 43.0 |
| kodim15 | 21.90 | 22.06 | 20.48 | 21.69 | 32.0 | 35.8 | 30.4 / 30.5 |
| kodim16 | 20.68 | 21.06 | 20.56 | 20.63 | 39.6 | 43.3 | 38.4 / 38.4 |
| kodim17 | 21.37 | 21.66 | 21.31 | 21.32 | 37.6 | 42.2 | 35.9 / 35.9 |
| kodim18 | 20.45 | 20.72 | 20.03 | 20.28 | 41.8 | 45.6 | 40.5 / 40.5 |
| kodim19 | 21.71 | 21.97 | 21.12 | 21.58 | 34.5 | 37.7 | 33.0 / 33.0 |
| kodim20 | 22.55 | 22.83 | 22.15 | 22.45 | 24.9 | 28.1 | 23.9 / 23.9 |
| kodim21 | 21.25 | 21.59 | 20.86 | 21.17 | 33.4 | 36.1 | 32.1 / 32.0 |
| kodim22 | 20.64 | 20.81 | 19.94 | 20.41 | 41.7 | 45.9 | 40.3 / 40.3 |
| kodim23 | 22.14 | 22.34 | 20.48 | 21.85 | 30.0 | 33.7 | 27.8 / 27.8 |
| kodim24 | 20.47 | 20.87 | 20.33 | 20.40 | 39.2 | 42.6 | 38.2 / 38.2 |

## Files and reproduction

- `output/kodak/kodimXX/` contains `gray.png` (the grayscale original), `whitened.png` (W, scaled by max\|W\| into [0, 1]) and one subfolder `<method>/` per method holding `<method>_recon.png` (Ŵ on the same scale as `whitened.png`), `<method>_recon_unwhitened.png` (Ŵ unwhitened, clipped to [0, 1]) and `<method>.log` (all printed output: spike counts, MSE, SNR, active).
- `output/kodak/results.csv` has every number above, one row per image and method.
- Earlier dictionaries: `SNN/phi_learned_whiten_doubleddict.npz` (doubled dictionary run), `SNN/phi_learned_whiten_f04_with_negatives.npz` (signed ISTA, f₀ = 0.4), `SNN/phi_learned_whiten_f06_with_negatives.npz` (signed ISTA, f₀ = 0.6).

To regenerate a single result:

```
mkdir -p output/kodak/kodimXX/<method>
python sparse_code_image.py --method <nonnegative_lca|spiking_lca|latency_slca|temporal_pattern_slca> --preprocess whiten \
    --image images/kodak/kodimXX.png --out-dir output/kodak > output/kodak/kodimXX/<method>/<method>.log
```

## Rounding error

In practice, the latency and temporal coding methods' discretization of values needs correction by removing the hard reset of v at the beginning of each cycle, which results in correction over time. For example, in latency coding, if the true u_i - lambda * lr is 4.5dt, the true firing timestep is timestep S - 4.5, but it will only spike at timestep S - 4.0, as it only checks whether the threshold is breached at integer time steps. Thus, it fires 0.5 timesteps too late and represents a value of 4.0dt, 0.5dt too low. A similar rounding error applies to temporal coding LCA, where a value such as 8.2q would be represented as a spike train of [...,1, 0, 0, 0] = 8q, meaning the represented is again slightly too low. Thus, we correct for this rounding by carrying over the leftover v membrane potential to the next cycle. Over the next few cycles, this extra membrane potential contributes to a correction to the rounding. With u_i - lr*lamdba = 4.5dt, a carrying the leftover v over instead of overwriting would result in v having a membrane potential of 0.5dt after the cycle finishes, carrying extra potential to the next cycle. After the next few cycles, the 0.5dt extra potential will cause some spikes to have slightly earlier firing times, correcting for the slightly late firing time of the original rounding error by adding that missing value to the state u_i in the next cycles. However, since u_i is a leaky integrator, this correction is not exact, as recent cycle leftovers are weighted higher than older ones. 

See [`errorbound.png`] for informal calculations; formalization will be added soon!
