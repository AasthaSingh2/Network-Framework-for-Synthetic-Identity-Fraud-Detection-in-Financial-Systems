# Synthetic Identity Fraud Detection in Digital Lending

A college research prototype organized as three separate fraud-analysis modules. The dashboard can review saved application-stage and graph-node records side by side, but the available artifacts do not establish that an applicant is the same entity as a graph node. For that reason, the dashboard does not calculate a combined risk score or make a lending decision.

## Architecture

- **Module A: Application-stage identity analysis.** `01_Application_Stage_Synthetic_Identity.ipynb` generates a synthetic applicant dataset and engineers identity-coherence features. `application_stage_final_20k.csv` contains saved experiment outputs, and `xgb_identity_model.json` contains an XGBoost model artifact. The saved CSV includes historical out-of-fold `xgb_proba` values and a 70% XGBoost / 30% ICS-risk blend. These are historical outputs, not live inference.
- **Module B: Synthetic linked graph.** `Linked_Synthetic_Dataset (1).ipynb` generates synthetic applicants, accounts, and transactions and contains GAT experiments. `graph_risk_results.csv` contains precomputed node risk outputs. `app.py` displays these saved records; it does not build a graph or score new applicants.
- **Module C: Uploaded network framework.** The six exact notebooks are `_01_data (1).ipynb`, `_02_explore_data (1).ipynb`, `_03_graph_construction (2).ipynb`, `_04_model_training (3).ipynb`, `_05_explainability (2).ipynb`, and `_06_fraud_alert.ipynb`. This is a separate IBM AML/BAF transaction and application-data workflow, with GAT/GCN and tabular baselines plus explainability. Its labels and data are not interchangeable with Modules A or B.
- **Integration layer.** `risk_review.py` validates and retrieves saved applicant/node snapshots and checks model artifact metadata. The Risk Review dashboard page shows the records independently and explains why an aggregate decision is unavailable. It does not accept a new applicant feature form or call the saved XGBoost model. Separately, the existing model was loaded read-only and smoke-tested against eight existing CSV rows; this does not add live scoring to the dashboard.

## Run the Dashboard

The project-local `.venv` is configured with the dashboard and research-notebook dependencies. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The standard dashboard and Risk Review page read the checked-in CSV files. Applicant lookup displays a saved historical output; it is not an XGBoost prediction for new input. Graph lookup displays a precomputed graph row. There is no applicant-to-node mapping in the checked-in data.

## Run Tests

The focused integration tests use Python's built-in `unittest` and temporary files:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests cover saved-record lookup, invalid/missing records, score range validation, model-feature metadata compatibility, and the rule that no combined decision is produced without a validated mapping and aggregation method. They do not execute or train any notebook model.

## Current Integration Boundary

The Risk Review page can display a historical application record by `APP...` ID and a precomputed graph row by `NODE_...` ID. The IDs come from different namespaces and no mapping file is present. Therefore the page provides component evidence only and deliberately does not average or combine the scores into a decision. A future scoring workflow needs an approved applicant-to-graph link and an evaluated aggregation policy before that can change.

Although the Module A and Module B generators both produce applicant IDs beginning with `APP100000`, they generate separate synthetic populations. The overlapping format (and potentially identical strings) is not proof of shared entities and must not be used as a join key. The graph-results CSV has node IDs only and contains no applicant ID field.

## Dependencies and External Data

- Dashboard and integration helpers: `streamlit`, `pandas` (NumPy is used by Streamlit dependencies).
- Module A notebook additionally needs Faker, XGBoost, LightGBM, scikit-learn, SHAP, matplotlib, seaborn, and NumPy/pandas.
- Module B notebook additionally needs Faker, scikit-learn, PyTorch, PyTorch Geometric, and NetworkX.
- Module C requires the authorized input files `IBM Transaction Dataset/HI-Small_Trans.csv` and `BAF NeurIPS Dataset/Base.csv`. `_03_graph_construction (2).ipynb` creates `outputs/ibm_graph.pt`, `outputs/account_to_idx.pkl`, and `outputs/scaler.pkl`; `_04_model_training (3).ipynb` creates `outputs/fraud_gat_model.pt`; the later notebooks consume the graph/checkpoint. These datasets and generated Module C artifacts are currently absent. The separately mentioned `coherence_gat_lambda2_v1.pth` belongs to the Module B historical Colab experiment, not Module C; the edited local Module B export path is `generated_outputs/models/coherence_gat_lambda2_historical.pth`.
- Dataset provenance and access: [IBM AML-Data](https://github.com/IBM/AML-Data) identifies the data license as CDLA-Sharing-1.0 and points to the improved [IBM AML Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), which lists `HI-Small_Trans.csv`. [Feedzai's BAF repository](https://github.com/feedzai/bank-account-fraud) points to the [BAF Kaggle dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022); Kaggle lists `Base.csv` at approximately 213.43 MB, one million rows, with target `fraud_bool` and license CC BY-NC-SA 4.0. Both exact Kaggle downloads require account sign-in. No download was performed and no authentication was bypassed. The public IBM Box share exposes an older 45M transaction collection and was not substituted for the exact HI-Small file.
- The `.venv` was installed with the notebook dependencies, including torch 2.14.1+cpu, torch-geometric 2.8.1, xgboost 3.4.1, lightgbm 4.7.0, scikit-learn 1.9.1, imbalanced-learn 0.14.2, pandas 3.1.0, NumPy 2.5.4, NetworkX 3.6.1, matplotlib 3.11.2, seaborn 0.13.2, SHAP 0.54.1, pyvis 0.3.2, Faker 40.43.0, joblib 1.6.0, and Streamlit 1.66.1. Installation was confined to the project `.venv`; root `requirements.txt` was not changed and still lists only `streamlit` and `pandas`.

## Historical Results and Evaluation

The following are recorded notebook outputs and have not been independently reproduced in the current environment:

- Module A's historical stratified five-fold out-of-fold evaluation reports XGBoost ROC-AUC 0.9577, PR-AUC about 0.888, and hard-fraud recall 0.603. Feature and ICS transformations were calculated globally before cross-validation in that historical path, so these are not clean final holdout metrics.
- Module B's historical applicant-level split reports baseline GAT accuracy 0.9285, precision 0.8526, recall 0.8210, F1 0.8365, ROC-AUC 0.9705, and PR-AUC 0.8657. The Coherence-Aware GAT lambda-2 output reports accuracy 0.9078, precision 0.7684, recall 0.8395, F1 0.8024, ROC-AUC 0.9552, PR-AUC 0.8350, and a validation-selected threshold of 0.5416. The graph is constructed before node splits, so this is transductive evaluation.
- Module C has saved notebook outputs for IBM AML / BAF experiments, including an older sampled GAT result. They are separate historical experiments, use different targets, and are not combined with Module A or B metrics here.

Synthetic results show behavior on generated patterns only; they do not establish real-world lending or financial-crime performance.

## Provenance and Limitations

The Module C notebooks were uploaded from the public network-framework repository identified by the project owner. No source license file was verified in this workspace; confirm attribution and license requirements before adapting its code. Original notebooks, model artifacts, CSV outputs, and backups are preserved.

The saved Module A XGBoost model loaded successfully with XGBoost 3.4.1 and returned finite probabilities in [0,1] for eight existing application CSV rows using the model-declared feature order. This was read-only inference; no model or data artifact was written, and the dashboard still does not score new user input. A CPU PyTorch Geometric `GATConv` smoke test passed. The six integration unit tests pass, and `app.py`, `risk_review.py`, and the test module compile. Module B dashboard scores are precomputed. Module C remains blocked until its authorized datasets are manually acquired. No cross-module mapping or scientifically validated aggregation rule currently exists, so the interface intentionally withholds a final combined decision. No notebook was run and no model was trained or checkpoint generated during setup.
