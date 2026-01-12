# EANM AI Committee - Quantum Computing Engagement Plan

This repository contains basic teaching material, tutorials and educational videos for those interested to engage with quantum computing (QC) within the Euopean Association of Nuclear Medicine (EANM). Thus, the collection of materials here is intended for beginners with little to no knowledge in QC. 
In addition, we also share advanced materials containing tutorials, hands-on as well as code repositories.
The repository is intended to be updated regularly, with eventually having an EANM AI Committee-created code base for QC as well. The initial version was made for the perspective paper:

> The Dawn of Quantum AI in Nuclear Medicine: an EANM Perspective. L. Papp, D. Visvikis, M. Sollini, K. Shi, M. Kirienko. The EANM Journal, 2026.


If you are new to the topic of QC, we recommend to process this repository from top-to-bottom, by following up with the contents below.

If you are ready to engage with our source code which is custom-tailored for Nuclear Medicine use cases, see
[Section `Nuclear Medicine Use Cases (Code)`](#nuclear-medicine-use-cases-code).

## QC Concepts - Youtube videos
Before doing anything else, we recommend to get familiar with the concept of QC by going through these videos:

Quantum Computers Explained – Limits of Human Technology – Kurzgesagt – In a Nutshell (2015, ~7 min). An animated overview of why classical computers face physical limits and how quantum bits (qubits) leverage superposition and entanglement to compute beyond those limits. Explains key concepts (qubits, superposition, measurement) in simple terms and discusses potential impacts:
- [Quantum Computers Explained – Limits of Human Technology](https://youtu.be/JhHMJCUmq28)

Veritasium has multiple videos about QC and quantum mechanics, utilizing beautiful visuals and intuitive explanations.
- [How Does a Quantum Computer Work?](https://www.youtube.com/watch?v=g_IaVepNDT4)

- [What makes quantum computers SO powerful?](https://www.youtube.com/watch?v=-UrdExQW0cs)

- [Something Strange Happens When You Trust Quantum Mechanics](https://youtu.be/qJZ1Ez28C-A?si=Ejk39rgtDCFM8ZAC)

- [Is This What Quantum Mechanics Looks Like?](https://www.youtube.com/watch?v=WIyTZDHuarQ)

- [There Is Something Faster Than Light](https://www.youtube.com/watch?v=NIk_0AW5hFU)

- [Parallel Worlds Probably Exist. Here’s Why](https://www.youtube.com/watch?v=kTXTPe3wahc)

Quantum Computers, Explained Cleo Abram (Huge If True). Clear explanations of qubits vs bits, why quantum computers are important, and applications like simulating molecules for medicine:
- [Quantum Computers, explained with MKBHD](https://www.youtube.com/watch?v=e3fz3dqhN44)

A visually rich exploration of quantum computing through the lens of math. 3Blue1Brown derives how qubit state vectors work and walks through Grover’s search algorithm step by step (advanced):
- [But what is quantum computing? (Grover's Algorithm)](https://www.youtube.com/watch?v=RQWpF2Gb-gU)

## QC Concepts - Teaching Material (Beginner)

Once you familiarized yourself with the videos above, you may proceed to access the following teaching material:

IBM Quantum “Understanding Quantum Information and Computation” – Beginner-level video series (free, 16 parts). A rigorous introduction to quantum computing fundamentals, with video lectures by IBM’s John Watrous and accompanying text. Emphasizes core concepts over heavy math, making it ideal for newcomers.
- [Qiskit playlist - Understanding Quantum Computing](https://www.youtube.com/playlist?list=PLOFEBzvs-VvqKKMXX4vbi4EB1uaErFMSO)

- [Basics of Quantum Information](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information)

FutureLearn – Understanding Quantum Computers – Beginner MOOC (Keio Univ.). An introductory 4-week course covering key quantum computing concepts with minimal mathematics. Explains qubits, superposition, entanglement, and potential applications in plain language.
- [Learning Resources on Quantum Computing](https://qosf.org/learn_quantum/)

openHPI: Introduction to Quantum Computing with Qiskit – Beginner course. Free course by IBM Quantum and HPI, teaching basic quantum computing using Qiskit in the cloud. Covers qubits, circuits, and simple algorithms with hands-on demos (no prior QC experience required).
- [Introduction to Quantum Computing with Qiskit (with IBM Quantum)](https://open.hpi.de/courses/qc-qiskit2022)

## Hands-On Tutorials (Beginner to Intermediate)

Once you are up-to more coding, try the following resources:

Qiskit Textbook & Tutorials – Interactive Jupyter Notebooks (Beginner-friendly). The open-source Qiskit Textbook offers step-by-step notebooks from basic quantum circuits to algorithms.
[Link](https://medium.com/qiskit/how-to-start-experimenting-with-quantum-image-processing-283dddcc6ba0) and [Link](https://github.com/Qiskit/textbook/tree/main/notebooks/ch-applications)

Qiskit’s official tutorial repository also includes a Quantum CNN notebook implementing a QCNN to classify simple images:
[Link](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html)

PennyLane Demos and Codebook – Hands-on QML tutorials (Beginner-Intermediate). Xanadu’s PennyLane provides extensive demonstrations on quantum machine learning. For example, the Quanvolutional Neural Network demo shows how to apply a quantum filter to MNIST images (quantum analog of CNN) and combine with classical networks:
[Link](https://pennylane.ai/qml/demos/tutorial_quanvolution)

The interactive Quantum Codebook is also available for guided practice on topics from basic qubit operations to variational algorithms:
[Link](https://monitsharma.github.io/Learn-Quantum-Computing-For-Free/)

AWS Braket Example: Hybrid Quantum ML – Intermediate tutorial (AWS platform). Amazon Braket’s public examples include a Quantum Machine Learning Hybrid Jobs notebook demonstrating a typical hybrid workflow (classical optimizer + quantum circuit model) on AWS. It guides users through data upload, setting hyperparameters, and running a simple QML algorithm on cloud simulators or QPUs. (Requires an AWS account; free tier available for small-scale use):
[Link](https://github.com/amazon-braket/amazon-braket-examples)

Hybrid QNN on MNIST (Qiskit + PyTorch) – Platform: Qiskit, Skill: Intermediate. A lightweight community example of a quantum-classical neural network for image classification. Implements a classical CNN feature extractor with a quantum fully-connected layer for MNIST digit data: 
[Link](https://github.com/dohun-qml/quantum-neural-network)

Quantum-Classical CNN (CIFAR-10 Binary Classifier) – Platform: Qiskit/PyTorch, Skill: Intermediate. End-to-end example of a variational quantum classifier on real images. Uses a parametrized quantum circuit as one layer in a CNN to distinguish two classes of CIFAR-10 (airplane vs. automobile): 
[Link](https://github.com/DRA-chaos/Quantum-Classical-Hyrid-Neural-Network-for-binary-image-classification-using-PyTorch-Qiskit-pipeline)

Qiskit Community Tutorials – QSVM & VQE demos – Platform: Qiskit, Skill: Beginner. The Qiskit community notebooks include simple machine learning and optimization demos. For instance, a Quantum SVM classifier for a toy image dataset (mapping pixel data to qubit feature maps and using QSVM): 
[Link](https://github.com/qiskit-community/ibm-quantum-challenge-fall-2021/blob/main/content/challenge-3/challenge-3.ipynb)


## Reviews & Reviews

In case you are interested in where the field stands, read the following reviews:

Review of medical image processing using quantum-enabled algorithms (2024) – Fei Yan et al. A comprehensive review of how quantum and quantum-inspired algorithms enhance traditional medical image processing, covering applications in disease diagnosis and medical image security. The authors survey advances in quantum-assisted image analysis (e.g. segmentation, optimization methods) and outline current performance limitations and future development plans for quantum techniques in imaging:
[Link](https://link.springer.com/article/10.1007/s10462-024-10932-x)

Quantum machine learning in medical image analysis: A survey (2023) – Lin Wei et al. – Neurocomputing. A survey examining the applications of quantum machine learning in medical image analysis. It details early QML approaches in imaging and is noted as the first comprehensive review in this subfield:
[Link](https://www.sciencedirect.com/science/article/pii/S0925231223000589)

Quantum algorithms and complexity in healthcare applications: a systematic review with machine learning-optimized analysis (2025) – Agostino Marengo & Vito Santamato – Frontiers in Computer Science. Analyzing 63 studies, the authors identify two dominant themes: quantum computing for AI in healthcare (e.g. quantum machine learning for diagnostics and predictive analytics) and quantum computing for health data security (e.g. quantum cryptography for EHR/privacy). The review highlights theoretical advances from quantum-enhanced algorithms for biomedical data analysis to blockchain-like security frameworks, and it concludes that quantum algorithms show promise to optimize complex diagnostic computations and protect medical data:
[Link](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1584114/full)

Quantum Computing in Medicine (2024) – James C. L. Chow – Medical Sciences (Basel). An open-access review covering the state of quantum computing in medical research and practice. It outlines fundamental QC concepts (qubits, superposition, entanglement) and surveys milestone applications: from quantum-accelerated drug discovery and molecular modeling in chemistry/proteomics, to quantum approaches in genomics (e.g. DNA sequencing) and medical diagnostics. Quantum machine learning techniques are explained (including quantum-enhanced imaging and radiotherapy simulations), and the article discusses practical challenges such as hardware scalability, error correction, and integration into clinics. The paper describes prospects for quantum–classical hybrid systems and emerging quantum hardware that could accelerate adoption of QC in personalized therapy planning and complex biological simulations:
[Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11586987/)

For reading a large selection of QC papers related to AI/ML, we recommend to follow-up with Christophe Pere's git repository which is actively maintained:
https://github.com/Christophe-pere/Roadmap-to-QML

## Nuclear Medicine Use Cases (Code)

The EANM-AI-QC folder contains a unified reference implementation for quantum ML workflows relevant to nuclear medicine:

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

- `00_quickstart.ipynb`: run full demo and plot test confusion matrices + ROC (no SHAP/LIME)
- `01_tabular_demo.ipynb`: tabular run + plots + per-case test predictions
- `02_nifti_demo.ipynb`: NIfTI PET±mask run + plots (synthetic mockups)
- `03_explainability.ipynb`: SHAP/LIME generation + visual review (separate; can be slow)
- `04_results_dashboard.ipynb`: dashboard over `Results/` (tables, confusion matrices, ROC, SHAP/LIME if present)

Use notebooks for **exploration, debugging, and transparent reporting** (e.g., inspecting `Results/*.csv`, looking at per-case predictions).
For **batch runs** or long simulations, prefer the CLI scripts (`qnm_qai.py`, `examples/*.sh`) and treat notebooks as a front-end that calls the same commands.

Typical setup (make sure Jupyter uses the same `.venv` as the terminal):

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

To control runtime, call the underlying Python script with caps:

```bash
python3 examples/run_explain_all.py   --results-dir Results   --max-features 16   --max-test-samples 2   --background-samples 8   --shap-max-evals 256   --lime-num-samples 300
```

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

- Fast run (no explainability):
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
- `[SHAP] ...` / `[LIME] ...` when explanations are run

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

## Results output

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

Threshold note (important):
- AUC is threshold-free.
- Confusion-matrix-derived metrics (accuracy, sensitivity, specificity, PPV, NPV, balanced accuracy) depend on the **decision threshold** applied to `prob_1`.
- If the code selects a non-0.5 decision threshold (for improved balanced accuracy), this value is recorded in the run metadata and/or results.

### Per-method artifacts

`Results/<dataset_name>/<method>/`

- `predictions/train.csv`, `predictions/test.csv`, optionally `predictions/infer.csv`
  - columns include `id`, `prob_1`, `pred_01`, `pred_label` and, when available, `true_label`
- `model/metadata.json` and model weights
- `explain/shap/` and `explain/lime/` outputs (after running `bash examples/run_explain_all.sh`)

## Determinism (repeatable runs)

Runs are seeded and should be repeatable on the same machine.

- Default seed: `0`
- Override with: `--seed <int>`

The CLI also sets common BLAS thread pools to 1 thread (best-effort) to reduce nondeterministic floating-point reductions.

If you enable `--shots` (finite sampling), results become stochastic by design unless you also control sampling seeds.
