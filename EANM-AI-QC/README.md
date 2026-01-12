# EANM-AI-QC

Unified reference implementation for quantum ML workflows relevant to nuclear medicine:
- Tabular quantum ML models (amplitude-encoded quantum kernel SVM and QCNN variants on feature vectors)
- NIfTI PET ± mask QCNN pipelines (amplitude encoding of flattened volumes)

By default, quantum circuits run on a classical simulator (PennyLane `default.qubit`).

## Repository structure (modular)

- `eanm_ai_qc/io/`
  - `tabular.py`: tabular CSV loader
  - `nifti.py`: NIfTI PET±mask loader
  - `encoding.py`: amplitude encoding (pad to `2^n` + L2 normalization)
- `eanm_ai_qc/models/`
  - `pl_kernel_svm.py`: PennyLane amplitude-kernel SVM
  - `pl_qcnn_muw.py`: MUW-like QCNN (ArbitraryUnitary head)
  - `pl_qcnn_alt.py`: ALT QCNN (StronglyEntanglingLayers head)
- `eanm_ai_qc/explain/`
  - `shap_explain.py`: SHAP (model-agnostic, permutation explainer)
  - `lime_explain.py`: LIME (tabular explainer)
- `eanm_ai_qc/runner.py`
  - experiment runner writing metrics and predictions into `Results/` (and optionally SHAP/LIME when enabled)

The CLI entry point is `qnm_qai.py`.

## Installation

```bash
cd ~/EANM-AI-QC
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## One-command demo run

```bash
bash examples/run_all_examples.sh
```

The demo runner **does not run SHAP/LIME explanations**. It only trains models, evaluates them, and writes performance + predictions to `Results/`.


## Jupyter notebooks

Notebooks are provided in `notebooks/`:

- `00_quickstart.ipynb`: end-to-end demo (no explanations)
- `01_tabular_demo.ipynb`: real tabular radiomics workflow
- `02_nifti_demo.ipynb`: NIfTI PET±mask workflow (synthetic mockups)
- `03_explainability.ipynb`: SHAP/LIME step (separate; can be slow)

Typical use:

```bash
cd ~/EANM-AI-QC
source .venv/bin/activate
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name eanm-ai-qc --display-name "EANM-AI-QC (.venv)"
jupyter lab
```

Then select the `EANM-AI-QC (.venv)` kernel and open the notebooks.

## Explanations (SHAP/LIME)

Run explanations separately *after* the models exist:

```bash
bash examples/run_explain_all.sh
```

Outputs are written under:
- `Results/<dataset>/<method>/explain/shap/`
- `Results/<dataset>/<method>/explain/lime/`

Runtime note: SHAP/LIME can take minutes to hours on a CPU simulator. The quantum-kernel SVM is the slowest because each explanation requires many kernel circuit evaluations.

## Runtime expectations and controls


If you are used to classical Python ML, these runs can feel unexpectedly slow. This is normal for *quantum circuit simulation* and for model-agnostic explainability.

### Why it can be slow

- **Quantum circuits are simulated** (PennyLane `default.qubit`). Simulation cost grows rapidly with the number of qubits because state vectors scale as `2^n`.
- **Amplitude encoding pads to `2^n`**. Example: 306 tabular features are padded to 512 amplitudes → **9 qubits**.
- **Quantum-kernel SVM is expensive**:
  - training requires a full kernel matrix `K_train` with roughly `N_train²/2` quantum circuit evaluations
  - testing/inference requires `N_test × N_train` evaluations
- **SHAP and LIME multiply cost** because they call the model many times on perturbed inputs:
  - model-agnostic explainers typically require hundreds to thousands of model evaluations per explained sample
  - for the **kernel SVM**, each model evaluation internally computes kernels against all training reference vectors, so the explainer overhead can dominate runtime

### What to do in practice

- Disable explainability for quick runs:
  ```bash
  python3 qnm_qai.py run --input <data> --methods pl_kernel_svm,pl_qcnn_alt,pl_qcnn_muw --no-explain
  ```
- Reduce training rows while debugging:
  ```bash
  python3 qnm_qai.py run --input <data> --max-samples-per-method 20 --no-explain
  ```
- Reduce qubit count for QCNN on tabular data (optional runtime control):
  ```bash
  python3 qnm_qai.py run --input <csv> --max-features-qcnn-tabular 64 --no-explain
  ```
- Keep NIfTI inputs small (VOIs around lesions rather than whole-body) to avoid large amplitude vectors → too many qubits.

### Progress output

The runner prints basic progress messages:
- `[RUN] ...` high-level phases per dataset and method
- `[pl_kernel_svm] ...` kernel matrix build progress
- `[pl_qcnn_alt] epoch ...` / `[pl_qcnn_muw] epoch ...` training epochs
- `[SHAP] ...` / `[LIME] ...` explainability start + per-sample LIME progress


## No preprocessing

The code takes the input features **as-is**. This means no scaling, standardization, feature engineering or e.g. imputation are done.

Minimal *encoding steps* are still required because quantum circuits cannot accept invalid numbers and amplitude encoding requires a normalized state vector:
- `NaN/Inf` values are replaced with `0.0`
- vectors are zero-padded to length `2^n`
- vectors are L2-normalized to form a valid amplitude-encoded quantum state

## Real tabular dataset (radiomics, PSMA-11 PET)

The repository includes a real tabular example dataset containing **anonymized PSMA-11 PET radiomic features**
extracted from **primary prostate lesions**, with an associated **binary label** intended to predict **Gleason risk**.

Source:
- https://osf.io/3nkx8/files/osfstorage

Files:
- `demo_data/tabular/raw/FDB.csv` and `demo_data/tabular/raw/LDB.csv` (semicolon-separated)
- `demo_data/tabular/real_train.csv` (features `f0..fN` + `label`)
- `demo_data/tabular/real_infer.csv` (features only)
- `demo_data/tabular/real_feature_map.csv` (maps `f#` to original feature names)

## NIfTI datasets (synthetic mockups)

The NIfTI volumes generated by `examples/make_synthetic_nifti.py` are **randomly-generated mockups** intended only to validate:
- loader correctness (PET±mask pairing)
- amplitude encoding + QCNN pipeline execution
- model saving/loading and inference
- output generation (metrics, SHAP, LIME)

Since these NIfTI volumes are synthetic, QCNN performance on them is **not** representative of real clinical capability.
Use your own NIfTI datasets for any meaningful assessment.

## Practical recommendation for simulator feasibility

Circuit width is determined by the amplitude vector length (`2^n` → `n` qubits).
To avoid overly complex circuits in a simulator environment:
- keep NIfTI inputs small (VOIs around lesions instead of whole-body volumes)
- limit training sample counts while debugging (`--max-samples-per-method`)
- start with few epochs and scale up only after the pipeline is stable

## QCNN variants: muw vs alt

Here, *muw* stands for the QCNN algorithm utilized in Nuclear Medicine studies at the Medical University of Vienna (L. Papp: laszlo.papp@meduniwien.ac.at) - paper is currently in review.

Both QCNN options:
- accept any amplitude-encoded vector (tabular feature vectors or flattened NIfTI volumes)
- use convolution + pooling blocks on qubits
- output the probability of measuring `|1⟩` on a designated qubit

### muw (MUW-like)

Final head: `qml.ArbitraryUnitary` on pooled qubits (high expressivity) provides prediction outputs.

Pros
- highly expressive final layer

Cons
- often harder to optimize (flat gradients / instability), especially on small datasets or with few epochs
- more expensive parameterization in the final block

### alt (alternative)

Final head: `qml.templates.StronglyEntanglingLayers` on pooled qubits (typically more stable) provides prediction probabilities.

Pros
- generally easier to optimize and more stable for demos/debugging

Cons
- potentially less expressive than an ArbitraryUnitary head in the pooled subspace

## Results output contract

For each dataset run, the runner writes:

### Summary metrics CSV (required)
`Results/<dataset_name>__results.csv`

Rows:
- one row per AI method

Columns include (train and test prefixes):
- confusion matrix counts: `tn, fp, fn, tp`
- sensitivity, specificity, PPV, NPV
- accuracy, AUC, balanced accuracy
- bookkeeping: `n_train_used`, `n_test`, `n_qubits`, `pad_len`

### Per-method artifacts
`Results/<dataset_name>/<method>/`
- `predictions/train.csv`, `predictions/test.csv`, optionally `predictions/infer.csv`
- `model/metadata.json` and model weights
- `explain/shap/` and `explain/lime/` outputs

## Determinism (repeatable runs)

Runs are seeded and should be repeatable on the same machine.

- Default seed: `0`
- Override with: `--seed <int>`

The CLI also sets common BLAS thread pools to 1 thread (best-effort) to reduce nondeterministic floating-point reductions.

