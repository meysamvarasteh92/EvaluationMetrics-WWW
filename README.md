# Counterfactual Explanations for Recommendation: Are We Hitting the Target?

Code for the short paper *"Counterfactual Explanations for Recommendation: Are We Hitting the Target?"*

Perturbation-based fidelity metrics for recommender explanations (POS@K, DEL, INS, CDCG) only check what happens to the **target** item when the explanation is removed from the user's history. They ignore what happens to the **rest of the recommendation list**, so an explanation that reshuffles the whole list gets the same credit as one that moves only the target. This repository implements the conventional metrics, their non-target counterparts, and the **list-aware** metrics proposed in the paper, which credit an explanation only when its effect is specific to the target.

## Metrics

For each test user, the target $t$ is the top-1 recommendation. For explanation sizes $K_e = 1, \dots, \pi$, the $K_e$ most important history items are removed (or kept alone) and the top-$K$ list is recomputed.

| Conventional metric | List-aware refinement | An explanation succeeds when… |
|---|---|---|
| POS@K | **Counterfactual Validity** | the target leaves the top-$K$ **and** the rest of the list stays similar (RBO $\geq \tau$) |
| DEL | **Necessity** | removing the explanation lowers the target's score more than the average non-target score |
| INS | **Sufficiency** | keeping only the explanation preserves the target's score more than the average non-target score |
| CDCG | **Rank Degradation** | the target drops in rank more than the average non-target item |

Each list-aware metric is reported as a coverage (share of users where the target is affected more than the non-targets) and a gap Δ (mean non-target effect minus target effect). Definitions are in `metrics_discrete.ipynb`, next to the code.

Default settings: $K = 20$, $\pi = 10$, $\tau = 0.7$, $c = 5$, RBO persistence $p = 0.9$.

## Experimental setup

- **Datasets:** ML-1M, Yahoo! Music, Pinterest
- **Recommenders:** MF and VAE (MF is called `MLP` in the code and checkpoint names)
- **Explainers:** Jaccard, Cosine (similarity-based); LIME, SHAP (post-hoc); ACCENT, SPINRec, LXR (counterfactual)

The paper reports ML-1M with MF. Results for the other datasets and for VAE are produced by the same notebooks.

## Repository structure

```
recommenders_architecture.ipynb  MF (`MLP`), VAE and NCF model definitions
help_functions.ipynb             ranking utilities (top-K list, target item)
lime.ipynb                       LIME for recommendation
LXR_training.ipynb               training the LXR explainer
create dictionaries.ipynb        step 1: explanation dictionaries for every explainer
metrics_discrete.ipynb           step 2: conventional, non-target and list-aware metrics
plot.ipynb                       step 3: figures
```

The notebooks import each other with the [`ipynb`](https://pypi.org/project/ipynb/) package, so run them from the repository root.

## Setup

```bash
pip install -r requirements.txt
```

## Data and checkpoints

We use the preprocessed splits, item-similarity files, SHAP values and pretrained recommender and LXR checkpoints from the LXR benchmark (Barkan et al.). Set the data and checkpoint paths at the top of each notebook to where you stored them.

## Reproducing the results

1. **Generate explanations.** Run `create dictionaries.ipynb` for the chosen dataset and recommender. It saves one explanation dictionary per explainer: `{user_id: [(history_item, score), ...]}`, most important first.
   SPINRec explanations are generated with the authors' implementation and saved in the same format.
2. **Compute the metrics.** Run `metrics_discrete.ipynb` with the same dataset and recommender to compute the metrics for $K_e = 1, \dots, \pi$.
3. **Figures.** Run `plot.ipynb`.

## License

MIT, see [LICENSE](LICENSE).
