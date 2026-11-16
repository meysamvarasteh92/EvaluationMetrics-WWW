# Data

Each dataset has its own folder: `data/ML1M/`, `data/Yahoo/`, `data/Pinterest/`. The files below follow the preprocessing of the LXR benchmark (Barkan et al.). Dataset contents are not tracked by git.

## Inputs (needed before running the notebooks)

| File | Used by | Content |
|---|---|---|
| `test_data_<dataset>.csv` | both | test users (index = user id) |
| `static_test_data_<dataset>.csv` | both | test users' binary history vectors; the last two columns are dropped |
| `jaccard_based_sim_<dataset>.pkl` | Jaccard explainer | `{(i, j): similarity}` for item pairs with `i <= j` |
| `cosine_based_sim_<dataset>.pkl` | Cosine explainer | `{(i, j): similarity}` for item pairs with `i <= j` |
| `item_to_cluster_<recommender>_<dataset>.pkl` | SHAP explainer | item → cluster id |
| `shap_values_<recommender>_<dataset>.pkl` | SHAP explainer | per-user SHAP values over item clusters (first column = user id) |

`<recommender>` is `MLP` (MF) or `VAE`.

## Generated

| File | Written by | Content |
|---|---|---|
| `<recommender>_<explainer>_expl_dict.pkl` | `generate_explanations.ipynb` (SPINRec: authors' code) | `{user_id: [(history_item, score), ...]}`, most important first |
