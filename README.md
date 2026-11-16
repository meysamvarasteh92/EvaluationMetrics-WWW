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

Each list-aware metric is reported as a coverage (share of users where the target is affected more than the non-targets) and a gap Δ (mean non-target effect minus target effect). Definitions are in `evaluate_metrics.ipynb`, next to the code.

Default settings (`config.py`): $K = 20$, $\pi = 10$, $\tau = 0.7$, $c = 5$, RBO persistence $p = 0.9$.

## Experimental setup

- **Datasets:** ML-1M, Yahoo! Music, Pinterest
- **Recommenders:** MF and VAE (MF is called `MLP` in the code and checkpoint names)
- **Explainers:** Jaccard, Cosine (similarity-based); LIME, SHAP (post-hoc); ACCENT, SPINRec, LXR (counterfactual)

The paper reports ML-1M with MF. Results for the other datasets and for VAE are produced by the same notebooks.

## Repository structure

```
config.py                        paths, dataset sizes, checkpoint names, metric hyperparameters
recommenders_architecture.ipynb  MF and VAE model definitions
help_functions.ipynb             ranking utilities (top-K list, target item)
lime.ipynb                       LIME for recommendation
generate_explanations.ipynb      step 1: explanation dictionaries for every explainer
evaluate_metrics.ipynb           step 2: conventional, non-target and list-aware metrics
results.ipynb                    step 3: tables and target vs. non-target figure
data/                            preprocessed datasets and explanation dictionaries (not tracked)
checkpoints/                     trained recommenders and LXR explainers (not tracked)
results/                         metric pickles, tables and figures (created by the notebooks)
```

The notebooks import each other with the [`ipynb`](https://pypi.org/project/ipynb/) package, so run them from the repository root.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10.

## Data and checkpoints

We use the preprocessed splits, item-similarity files, SHAP values and pretrained recommender and LXR checkpoints from the LXR benchmark (Barkan et al.). Place them as described in [`data/README.md`](data/README.md) and [`checkpoints/README.md`](checkpoints/README.md).

## Reproducing the results

1. **Generate explanations.** Open `generate_explanations.ipynb`, set `data_name` and `recommender_name`, and run all cells. This writes `data/<dataset>/<recommender>_<explainer>_expl_dict.pkl` for Jaccard, Cosine, LIME, SHAP, ACCENT and LXR.
   SPINRec explanations are generated with the authors' implementation and saved in the same format: `{user_id: [(history_item, score), ...]}`, most important first, as `data/<dataset>/<recommender>_spinrec_expl_dict.pkl`.
2. **Compute the metrics.** Run `evaluate_metrics.ipynb` with the same `data_name` and `recommender_name`. For every explainer it writes `results/<dataset>/<recommender>_<explainer>_metrics.pkl`, a dict mapping each metric to its values for $K_e = 1, \dots, \pi$.
3. **Tables and figures.** Run `results.ipynb` to build the comparison table (CSV) and the target vs. non-target scatter plot (PDF) in `results/<dataset>/`.

## License

MIT, see [LICENSE](LICENSE).
