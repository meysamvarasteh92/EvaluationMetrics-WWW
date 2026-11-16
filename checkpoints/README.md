# Checkpoints

Pretrained recommenders and LXR explainers, loaded with `torch.load`. File names must match `RECOMMENDER_CHECKPOINTS` and `LXR_CHECKPOINTS` in `config.py`. Checkpoint files are not tracked by git.

| Dataset | MF (`MLP`) | VAE |
|---|---|---|
| ML1M | `MLP1_ML1M_0.0076_256_7.pt` | `VAE_ML1M_0.0007_128_10.pt` |
| Yahoo | `MLP2_Yahoo_0.0083_128_1.pt` | `VAE_Yahoo_0.0001_128_13.pt` |
| Pinterest | `MLP_Pinterest_0.0062_512_21_0.pt` | `VAE_Pinterest_12_18_0.0001_256.pt` |

The matching LXR explainer checkpoints (`LXR_<dataset>_<recommender>_*.pt`) are listed in `config.py`.
