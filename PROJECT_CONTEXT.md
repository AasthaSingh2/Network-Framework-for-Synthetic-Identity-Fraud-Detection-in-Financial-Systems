# Project Context

## Repository Map

This repository contains three related but separate research modules. They are not currently one integrated scoring pipeline.

### Module A: Application-Stage Identity Fraud

- Canonical notebook: `01_Application_Stage_Synthetic_Identity.ipynb`
- Artifacts: `application_stage_final_20k.csv`, `xgb_identity_model.json`
- Purpose: synthetic applicant-level identity coherence and fraud-risk estimation.
- Entity/target contract: one synthetic row per applicant, `applicant_id` values `APP100000` through `APP119999`; `label=1` is generated synthetic fraud and `label=0` genuine. `fraud_difficulty` subdivides synthetic fraud into easy/medium/hard and genuine rows into `Genuine`.
- Existing implementation: generates about 20,000 applicants; engineers identity and sharing features; trains XGBoost/LightGBM experiments; records an XGBoost probability plus ICS-derived risk blend.
- Evaluation status: historical notebook outputs are saved, but the notebook has no current execution record. A separate leakage-aware train/validation/test evaluation cell has been added but has not been run. Treat all existing metrics as historical until reproduced.
- Verified artifact check: the XGBoost JSON declares 15 ordered feature names; all 15 names are columns in `application_stage_final_20k.csv`. After installing XGBoost in the project `.venv`, the saved model loaded read-only and produced finite probabilities in [0,1] for eight existing CSV rows selected in the model-declared feature order. No model or data artifact was written.

### Module B: Synthetic Linked-Graph Fraud

- Canonical notebook: `Linked_Synthetic_Dataset (1).ipynb`
- Artifacts: `graph_risk_results.csv`, `app.py`
- Purpose: synthetically generate applicants, accounts, and transactions, then investigate graph-level risk with degree features, ICS, baseline GAT, GAT + ICS, and Coherence-Aware GAT experiments.
- Entity/target contract: the notebook independently generates 4,000 synthetic applicants, starts its own `applicant_id` sequence at `APP100000`, generates account IDs `ACC...`, and gives graph nodes `NODE_...` IDs in the exported CSV. `label=1` is that notebook's generated synthetic fraud label; it is not linked to Module A's similarly formatted applicant IDs.
- The notebook now generates its synthetic inputs in memory and directs generated exports to `generated_outputs/`; it does not depend on Module A's applicant CSV or XGBoost model.
- Dashboard status: `app.py` displays precomputed graph-risk results. It does not construct graphs or score arbitrary new applicants live. It was started successfully from `.venv` at `http://localhost:8765`; the 4,906-row CSV contains every required dashboard column and has no nulls in the score fields. The deprecated dataframe width argument was replaced with `width="stretch"`, and a clean restart showed no corresponding warning.
- Integration UI: `app.py` now has a Risk Review page that independently looks up saved applicant IDs from Module A and graph node IDs from Module B. It reports Module A values as historical saved outputs and graph values as precomputed CSV values. There is no applicant-to-node key, so the page deliberately provides no combined decision. The model metadata panel reports XGBoost availability and feature contract; it does not attempt model loading.
- Module A/B connection status: unsupported as a record join. Both generators happen to produce strings beginning `APP100000`, but they are separate generated populations with different sizes/features and no crosswalk; equal-looking IDs are collisions by construction/format, not verified entity identity. Module B's output CSV has a `node_id` but no `applicant_id`.
- Evaluation status: reported metrics are historical notebook outputs. Current notebook cells have not been executed in this workspace. The graph experiments are transductive: node connections/features are constructed over the generated graph before node-level splits are applied.

### Module C: Uploaded Network/Financial-Transaction Framework

- Exact uploaded notebook files (six notebooks; `_04_model_training (3).ipynb` is one file):
	1. `_01_data (1).ipynb`
	2. `_02_explore_data (1).ipynb`
	3. `_03_graph_construction (2).ipynb`
	4. `_04_model_training (3).ipynb`
	5. `_05_explainability (2).ipynb`
	6. `_06_fraud_alert.ipynb`
- Purpose: a separate IBM AML transaction graph and BAF application-data research workflow. It includes behavioral/graph features, GAT/GCN and tabular baselines, plus notebook explainability and alert/report experiments.
- Per-notebook execution and artifact chain:
	- `_01_data (1).ipynb`: checks/listings for `IBM Transaction Dataset/` and `BAF NeurIPS Dataset/`; reads `HI-Small_Trans.csv` and `Base.csv`. It does not generate substitutes. The first cell now fails early with an explicit missing-file list. Its original is preserved as `_01_data (1).ipynb.original.bak`.
	- `_02_explore_data (1).ipynb`: reads the same two CSV inputs, performs EDA, and writes `ibm_eda.png` and `baf_eda.png` in the current working directory.
	- `_03_graph_construction (2).ipynb`: reads IBM `HI-Small_Trans.csv`; constructs a transaction graph and writes `outputs/ibm_graph.pt`, `outputs/account_to_idx.pkl`, `outputs/scaler.pkl`, and visualization files including `outputs/fraud_network_interactive.html`.
	- `_04_model_training (3).ipynb`: loads `outputs/ibm_graph.pt`; trains GAT/GCN/RF+SMOTE/MLP+SMOTE; writes `outputs/fraud_gat_model.pt` and training/comparison figures. It requires PyTorch, PyTorch Geometric, scikit-learn, imbalanced-learn, NumPy, and plotting dependencies.
	- `_05_explainability (2).ipynb`: loads `outputs/ibm_graph.pt` and `outputs/fraud_gat_model.pt`; uses PyG GNNExplainer and SHAP; writes explanation figures under `outputs/`.
	- `_06_fraud_alert.ipynb`: loads the graph and GAT checkpoint, generates node alerts/report visuals, and writes `outputs/final_dashboard.png`.
- Exact missing-input checklist verified in the active project, downloaded Module C source mirror, and likely user folders: `IBM Transaction Dataset/HI-Small_Trans.csv`; `BAF NeurIPS Dataset/Base.csv`; `outputs/ibm_graph.pt`; `outputs/account_to_idx.pkl`; `outputs/scaler.pkl`; `outputs/fraud_gat_model.pt`; generated `outputs/` visualization artifacts. Search covered Downloads, Documents, Desktop, OneDrive Documents/Desktop, the duplicate project folder, and the downloaded Module C source folder; no matching files were found. The project ZIP contains no matching dataset/checkpoint entries. The synthetic Module B intermediates `linked_applicants.csv`, `linked_accounts.csv`, and `linked_transactions.csv` are absent on disk, but the edited Module B notebook creates them in memory and exports under `generated_outputs/` when run.
- Checkpoint clarification: `outputs/fraud_gat_model.pt` is produced by Module C `_04_model_training (3).ipynb` and consumed by Module C `_05_explainability (2).ipynb` and `_06_fraud_alert.ipynb`. `coherence_gat_lambda2_v1.pth` is not referenced as a Module C artifact in these notebooks. It belongs to the separate Module B Coherence-Aware GAT experiment in its older saved Colab workflow; the edited local export cell now saves to `generated_outputs/models/coherence_gat_lambda2_historical.pth`. Neither file is present in this workspace.
- Module C compatibility: its graph construction, behavioral transaction features, and explanation ideas may inform Module B after methodological review. IBM AML laundering labels and BAF application labels are not interchangeable with Module A's synthetic identity-fraud labels or Module B's synthetic labels. Module C artifacts/metrics must remain separate unless a compatible target, features, and evaluation split are established. No integration is verified.
- Compatibility matrix: A-to-B applicant join is unsupported (no crosswalk; identical-looking APP prefixes do not establish identity); B-to-C graph/model join is unsupported (different source populations, entity construction, feature schemas, and targets); A-to-C applicant join is unsupported (synthetic identity target versus BAF fraud_bool/IBM laundering-derived account labels). Only separate side-by-side display of saved evidence is currently supported and tested; score aggregation is not.
- Historical outputs are present in notebooks, but none of the six currently has an executed-cell record in this workspace. They do not prove current reproducibility.
- Provenance: these six notebooks were uploaded from the public repository named in the project instructions. No license file or attribution terms have been confirmed in this workspace. Check the source repository's license and preserve attribution before copying or adapting code into other modules.

## Shared Status and Limits

- `requirements.txt` currently contains only `streamlit` and `pandas`; it is sufficient for the dashboard's visible imports but not for running all research notebooks. The additional research dependencies for this setup were installed only in `.venv`; `requirements.txt` was not changed.
- Active interpreter: project-local `.venv\Scripts\python.exe`, Python 3.13.7 on Windows x64. No global Python environment was changed.
- Setup package versions verified in `.venv`: torch 2.14.1+cpu; torch-geometric 2.8.1; xgboost 3.4.1; lightgbm 4.7.0; scikit-learn 1.9.1; imbalanced-learn 0.14.2; pandas 3.1.0; NumPy 2.5.4; NetworkX 3.6.1; matplotlib 3.11.2; seaborn 0.13.2; SHAP 0.54.1; pyvis 0.3.2; Faker 40.43.0; joblib 1.6.0; Streamlit 1.66.1. Existing packages were retained; setup was installed only in `.venv`.
- `risk_review.py` provides read-only CSV lookups, score-range checks, and XGBoost JSON/CSV feature metadata checks. Its dashboard flow does not load or execute the model; a separate setup smoke test loaded the saved artifact and predicted on eight pre-existing CSV rows without writing files.
- `tests/test_risk_review.py` contains six standard-library unit tests. Last run: all six passed using `.venv` Python and pandas.
- `README.md` documents the current modular architecture, setup, exact test command, historical metrics, external data requirements, and limitations.
- Dashboard verification: Streamlit started after integration at `http://localhost:8765`, and the browser page showed the Risk Review navigation entry. Real applicant/graph lookup and no-aggregate behavior were verified through helper tests and a direct saved-artifact check. Browser automation did not complete a click-through of the Risk Review radio item, so full interactive page flow is not claimed as tested.
- The workspace directory is not a Git repository, so Git status/history cannot be used here to determine prior edits.
- A duplicate `Major_Project_V1_GitHub` folder exists under Downloads. Its saved CSVs, XGBoost JSON, requirements, and `app.py` were hash-compared: the four non-app artifacts are identical; its `app.py` is an older/different copy. A separately downloaded Network Framework folder contains the six Module C notebooks only. The existing project ZIP has eight entries and no matching Module C inputs/checkpoints.
- Dataset provenance verified: IBM `HI-Small_Trans.csv` is part of the IBM AML synthetic transaction dataset; IBM's maintained `IBM/AML-Data` README points to the improved Kaggle dataset and identifies the data license as CDLA-Sharing-1.0. Kaggle lists the exact HI-Small file and license. The original IBM Box share is browsable but hosts an older 45M transaction collection and is not the requested exact Kaggle file.
- BAF `Base.csv` provenance verified from Feedzai's `feedzai/bank-account-fraud` repository: it directs users to Kaggle for the dataset suite. Kaggle reports `Base.csv` at approximately 213.43 MB, one million records, target `fraud_bool`, and CC BY-NC-SA 4.0. The Kaggle page redirects to sign-in/sign-up. No authentication or agreement was bypassed.
- Dataset files were not downloaded: both exact files are distributed through Kaggle and the available download route requires account sign-in. Manual user action is required. Expected destinations remain `IBM Transaction Dataset/HI-Small_Trans.csv` and `BAF NeurIPS Dataset/Base.csv`.
- No model training, checkpoint generation, data-file modification, or end-to-end notebook execution has been performed. The existing XGBoost model was loaded with XGBoost 3.4.1 and used for read-only predictions on eight rows from the saved application CSV; predictions were finite and in [0,1], with no model or CSV written. PyG `GATConv` passed a small CPU forward smoke test. Existing dashboard startup/browser render and CSV compatibility were tested; six integration unit tests passed; app/helper/test syntax checks passed.
- Importing torch produced a PyTorch deprecation FutureWarning for `torch.jit.script`; core imports and the GAT forward test still succeeded.
- Synthetic experiment results are not evidence of performance on real lending or transaction-fraud data.

## Recommended Next Steps

1. Manually sign in to Kaggle, accept/view its dataset terms, and download the IBM HI-Small and BAF Base files; place them at the exact expected paths above. Do not copy/upload them publicly without following CDLA-Sharing-1.0 and CC BY-NC-SA 4.0 terms.
2. Verify downloaded file sizes, checksums, headers, row counts, and labels before running the Module C input cell.
3. Keep Module C training/checkpoint creation paused for separate approval. Installing packages does not mean model artifacts exist or that the notebook pipeline has executed.
4. Reproduce the leakage-aware Module A holdout evaluation only after explicit training approval.
5. Run Module B from a clean kernel and document its transductive evaluation setup and generated outputs.
