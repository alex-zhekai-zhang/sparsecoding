# Sparse Coding for Spiking Neural networks

See [`Alternative_Coding_Methods.pdf`](Alternative_Coding_Methods.pdf) for formulations.

To visualize Phi, run ```python SNN/config.py n``` for an index n in Phi, or just run ```python SNN/config.py``` to see all of it.
---

# Spiking LCA comparison on Kodak

Rate-coded (`spiking_lca`), latency-coded (`latency_slca`) and temporal-pattern-coded (`temporal_pattern_slca`) spiking LCA, run on all 24 Kodak images with whitening preprocessing (`--preprocess whiten`). The control (`mean`) is an all-zero code.

## Setup

| | |
|---|---|
| Images | Kodak `kodim01`–`kodim24`, grayscale, downscaled to 512 px on the long side (512×336 or 336×512, 2688 patches each) |
| Whitening | Whole image filtered in Fourier space by R(f) = f·exp(−(f/f₀)⁴), f₀ = 0.4 cycles/pixel, f = √(fx² + fy²) (`np.fft.fftfreq` units), then cut into 8×8 patches and scaled to unit variance per image |
| Dictionary | `SNN/phi_learned_whiten.npz`: 64×128 (single, not doubled), learned with signed ISTA (λ = 0.1, 10,000 iterations, batch 256) on 100k random patches of whitened Kodak images (f₀ = 0.4), columns at norm 1 |
| Parameters | λ = 0.1, ε = 0.1, lr = 1/‖Φ‖₂² = 0.160 (`SNN/config.py`) |
| Rate | 204,800 steps, dt = 1 / max\|x\| per image |
| Latency | 400 cycles × 512 steps = 204,800 steps, T = max\|x\| per image, leftover v carried across cycles (see Rounding error) |
| Temporal pattern | 400 cycles × 9 bits = 3,600 steps, same T, so q = T/512, the same grid as latency, leftover v carried across cycles |

Metric: SNR = 10·log10(mean(W²) / MSE), where W is the whitened image and MSE = mean((Ŵ − W)²), with Ŵ the reconstruction Φa put back together and the scale undone, before unwhitening. "Active" is the number of nonzero coefficients per patch, out of 128.

## Summary (mean ± std over 24 images)

| Method | SNR (dB) | Active / patch | Recurrent spikes / patch | Input spikes / patch | Steps | Wall-clock / image |
|---|---:|---:|---:|---:|---:|---:|
| Control (zero code) | 0.00 ± 0.00 | 0 | — | — | — | — |
| Rate | 15.05 ± 0.71 | 41.7 ± 4.9 | 528,804 | 860,211 | 204,800 | 506 s |
| Latency | 13.04 ± 0.61 | 35.8 ± 5.0 | 15,343 | 25,011 | 204,800 | 522 s |
| Temporal pattern | 13.13 ± 0.63 | 35.7 ± 4.9 | 37,228 | 65,999 | 3,600 | 9.8 s |

Wall-clock was measured with 6 jobs running at once, 3 threads each. Earlier results are in `legacy_tests/`.

## Per-image SNR (dB) and active coefficients per patch

| Image | Rate | Latency | TP | Active (rate) | Active (lat / TP) |
|---|---:|---:|---:|---:|---:|
| kodim01 | 14.48 | 12.61 | 12.98 | 46.2 | 41.8 / 41.8 |
| kodim02 | 14.50 | 12.65 | 12.66 | 45.4 | 37.3 / 37.3 |
| kodim03 | 15.40 | 13.27 | 13.31 | 37.9 | 31.6 / 31.5 |
| kodim04 | 14.55 | 12.52 | 12.60 | 44.6 | 37.4 / 37.3 |
| kodim05 | 13.94 | 12.08 | 12.11 | 48.0 | 42.3 / 42.3 |
| kodim06 | 15.07 | 13.23 | 13.24 | 43.8 | 38.0 / 38.0 |
| kodim07 | 16.16 | 14.09 | 14.12 | 37.6 | 32.1 / 32.0 |
| kodim08 | 14.83 | 13.06 | 13.28 | 42.9 | 39.2 / 39.1 |
| kodim09 | 16.11 | 13.79 | 14.00 | 33.5 | 27.8 / 27.8 |
| kodim10 | 15.56 | 13.55 | 13.56 | 39.3 | 32.1 / 32.1 |
| kodim11 | 15.07 | 13.06 | 13.06 | 42.7 | 36.5 / 36.5 |
| kodim12 | 16.11 | 14.14 | 14.26 | 39.2 | 32.9 / 32.9 |
| kodim13 | 13.64 | 11.90 | 11.91 | 49.8 | 45.1 / 45.0 |
| kodim14 | 14.36 | 12.49 | 12.51 | 48.7 | 42.6 / 42.6 |
| kodim15 | 15.39 | 13.18 | 13.33 | 37.6 | 31.6 / 31.6 |
| kodim16 | 15.11 | 13.30 | 13.31 | 44.1 | 38.1 / 38.1 |
| kodim17 | 15.14 | 13.03 | 13.03 | 43.1 | 35.9 / 35.9 |
| kodim18 | 14.22 | 12.28 | 12.35 | 46.3 | 39.9 / 39.8 |
| kodim19 | 15.64 | 13.65 | 13.70 | 38.4 | 32.7 / 32.7 |
| kodim20 | 16.25 | 13.98 | 14.09 | 31.5 | 26.3 / 26.3 |
| kodim21 | 15.39 | 13.24 | 13.32 | 36.8 | 31.8 / 31.8 |
| kodim22 | 14.47 | 12.43 | 12.56 | 46.9 | 39.9 / 39.8 |
| kodim23 | 15.41 | 12.93 | 13.30 | 34.2 | 27.8 / 27.8 |
| kodim24 | 14.37 | 12.45 | 12.50 | 43.4 | 38.0 / 37.9 |

## Files and reproduction

- `output/kodak/kodimXX/` contains `gray.png` (the grayscale original), `whitened.png` (W, scaled by max\|W\| into [0, 1]) and one subfolder `<method>/` per method holding `<method>_recon.png` (Ŵ on the same scale as `whitened.png`), `<method>_recon_unwhitened.png` (Ŵ unwhitened, clipped to [0, 1]) and `<method>.log` (all printed output: spike counts, MSE, SNR, active).
- `output/kodak/results.csv` has every number above, one row per image and method.
- Earlier dictionaries: `SNN/phi_learned_whiten_doubleddict.npz` (doubled dictionary run), `SNN/phi_learned_whiten_f06.npz` (f₀ = 0.6).

To regenerate a single result:

```
python sparse_code_image.py --method <mean|spiking_lca|latency_slca|temporal_pattern_slca> --preprocess whiten \
    --image images/kodak/kodimXX.png --out-dir output/kodak > output/kodak/kodimXX/<method>.log
```

## Rounding error

In practice, the latency and temporal coding methods' discretization of values needs correction by removing the hard reset of v at the beginning of each cycle, which results in correction over time. For example, in latency coding, if the true u_i - lambda * lr is 4.5dt, the true firing timestep is timestep S - 4.5, but it will only spike at timestep S - 4.0, as it only checks whether the threshold is breached at integer time steps. Thus, it fires 0.5 timesteps too late and represents a value of 4.0dt, 0.5dt too low. A similar rounding error applies to temporal coding LCA, where a value such as 8.2q would be represented as a spike train of [...,1, 0, 0, 0] = 8q, meaning the represented is again slightly too low. Thus, we correct for this rounding by carrying over the leftover v membrane potential to the next cycle. Over the next few cycles, this extra membrane potential contributes to a correction to the rounding. With u_i - lr*lamdba = 4.5dt, a carrying the leftover v over instead of overwriting would result in v having a membrane potential of 0.5dt after the cycle finishes, carrying extra potential to the next cycle. After the next few cycles, the 0.5dt extra potential will cause some spikes to have slightly earlier firing times, correcting for the slightly late firing time of the original rounding error by adding that missing value to the state u_i in the next cycles. However, since u_i is a leaky integrator, this correction is not exact, as recent cycle leftovers are weighted higher than older ones. 

See [`errorbound.png`] for informal calculations; formalization will be added soon!
