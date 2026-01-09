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

Quantum Computers, Explained Cleo Abram (Huge If True). Tech YouTuber MKBHD joins Abram to tour an IBM quantum computer and ask “dumb” questions. Clear explanations of qubits vs bits, why quantum computers are important, and applications like simulating molecules for medicine:
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


## Reviews - Medical (Imaging) Use Cases

In case you are interested in where the field stands, read the following reviews:

Review of medical image processing using quantum-enabled algorithms (2024) – Fei Yan et al. A comprehensive review of how quantum and quantum-inspired algorithms enhance traditional medical image processing, covering applications in disease diagnosis and medical image security. The authors survey advances in quantum-assisted image analysis (e.g. segmentation, optimization methods) and outline current performance limitations and future development plans for quantum techniques in imaging:
[Link](https://link.springer.com/article/10.1007/s10462-024-10932-x)

Quantum machine learning in medical image analysis: A survey (2023) – Lin Wei et al. – Neurocomputing. A survey examining the applications of quantum machine learning in medical image analysis. It details early QML approaches in imaging and is noted as the first comprehensive review in this subfield:
[Link](https://www.sciencedirect.com/science/article/pii/S0925231223000589)

Quantum algorithms and complexity in healthcare applications: a systematic review with machine learning-optimized analysis (2025) – Agostino Marengo & Vito Santamato – Frontiers in Computer Science. Analyzing 63 studies, the authors identify two dominant themes: quantum computing for AI in healthcare (e.g. quantum machine learning for diagnostics and predictive analytics) and quantum computing for health data security (e.g. quantum cryptography for EHR/privacy). The review highlights theoretical advances from quantum-enhanced algorithms for biomedical data analysis to blockchain-like security frameworks, and it concludes that quantum algorithms show promise to optimize complex diagnostic computations and protect medical data:
[Link](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1584114/full)

Quantum Computing in Medicine (2024) – James C. L. Chow – Medical Sciences (Basel). An open-access review covering the state of quantum computing in medical research and practice. It outlines fundamental QC concepts (qubits, superposition, entanglement) and surveys milestone applications: from quantum-accelerated drug discovery and molecular modeling in chemistry/proteomics, to quantum approaches in genomics (e.g. DNA sequencing) and medical diagnostics. Quantum machine learning techniques are explained (including quantum-enhanced imaging and radiotherapy simulations), and the article discusses practical challenges such as hardware scalability, error correction, and integration into clinics. The paper describes prospects for quantum–classical hybrid systems and emerging quantum hardware that could accelerate adoption of QC in personalized therapy planning and complex biological simulations:
[Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11586987/)

## Nuclear Medicine Use Cases (Code)

The folder EANM-AI-QC is a unified reference implementation for:
- **Tabular quantum ML** (quantum-kernel SVM)
- **3D NIfTI PET quantum CNNs** (PET ± ROI mask pairs)

This code runs quantum circuits on a classical simulator by default (PennyLane `default.qubit`).
It prints explicit evaluation diagnostics (train/test confusion matrices and per-test-case predictions)
to prevent misinterpretation of metrics vs inference-only predictions.

### Quick start (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

```bash
cd ~/EANM-AI-QC
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Generate synthetic demo inputs

```bash
python3 examples/make_synthetic_tabular.py
python3 examples/make_synthetic_nifti.py
```

### Run all examples

```bash
bash examples/run_all_examples.sh
```

Outputs:
- models saved under `models/`
- prediction CSVs saved under `outputs/`

### Tabular input requirements

- CSV/TSV with a label column named **`label`** (default).
- If your label column has another name, pass `--label-col <name>` during training.
- Inference CSVs can omit the label column.

### NIfTI input requirements (PET ± mask)

Supported layouts:

#### A) Explicit split

```
dataset_root/
  Train/
    case001_0_PET.nii.gz
    case001_0_mask.nii.gz   (optional)
  Test/
    ...
```

#### B) Flat layout (internal split)

```
dataset_root/
  case001_0_PET.nii.gz
  case001_0_mask.nii.gz
  ...
```

#### Mask semantics

- `mask > 0` means voxel belongs to VOI
- `mask == 0` means voxel is excluded
- applied as: `PET := where(mask>0, PET, 0)`

If no mask exists (or masks are disabled via `--mask-pattern NONE`) the full PET volume is used.

#### Resolution handling

- no forced image resolution
- flatten volume and pad to `2^n` for amplitude encoding
- optional resampling via `--target-shape Z Y X` (requires SciPy)

### Output CSV interpretation

Prediction CSV columns:
- `id`: sample id (row index for tabular; case id for NIfTI)
- `prob_1`: predicted probability for class `labelmap.classes[1]`
- `pred_01`: thresholded prediction (0/1 at 0.5)
- `pred_label`: predicted label token in original label namespace

### Evaluation prints during training

The `train` command prints:
- TRAIN and TEST confusion matrices (clearly labeled)
- sensitivity, specificity, PPV, NPV
- per-test-case (id, prob_1, pred, true)

This is the correct way to connect aggregate metrics and individual predictions.


### Included real tabular dataset (FDB/LDB)

This repo snapshot includes:
- `demo_data/tabular/raw/FDB.csv`
- `demo_data/tabular/raw/LDB.csv`
- `demo_data/tabular/real_train.csv`
- `demo_data/tabular/real_infer.csv`
- `demo_data/tabular/real_feature_map.csv`

`examples/run_all_examples.sh` trains the tabular model on `real_train.csv`.

### Notes on included example data

#### Real tabular dataset (radiomics PSMA-11 PET)

The repository includes a **real tabular example dataset** containing **anonymized PSMA-11 PET radiomic features** extracted from **primary prostate lesions**, with an associated **binary label** intended to predict **Gleason risk**.

This real tabular data example is sourced from the OSF repository:
- https://osf.io/3nkx8/files/osfstorage

In this codebase, the tabular features are used to demonstrate the end-to-end workflow:
- load CSV features + label
- train a quantum-kernel model (simulator backend)
- print train/test diagnostics (confusion matrices, sensitivity/specificity, PPV/NPV)
- run inference and save prediction CSV outputs

#### NIfTI datasets (synthetic mockups)

The NIfTI volumes generated by the example scripts are **randomly-generated mockups**. They exist to validate that:
- the PET±mask loader works,
- the amplitude-encoding and QCNN pipelines run,
- model saving/loading and inference work end-to-end.

Since these NIfTI volumes are synthetic, the prediction results of the QCNN models (**muw** or **alt**) are **not** representative of the methods’ real capabilities. Use your own NIfTI datasets (ideally with clinically meaningful VOIs and labels) for any performance assessment.

#### Practical recommendation for simulator feasibility

Amplitude encoding requires vectors of length `2^n` (padding is applied automatically), and QCNN circuit width is driven by `n_qubits`.

To avoid generating overly complex circuits in a simulator environment:
- prefer **small NIfTI inputs**, e.g. VOIs around lesions rather than whole-body volumes,
- consider resampling to a modest target shape (if appropriate),
- start with small `epochs` and only scale up after the pipeline is stable.

### QCNN variants: muw vs alt

Both QCNN options (**muw** or **alt**) share the same NIfTI loader and preprocessing, but differ in circuit design.
Here **muw** implies the qCNN models investiaged by the Medical University of Vienna, Applied Quantum Computing Group (laszlo.papp@meduniwien.ac.at) - manuscript reporting the results is currently in review.


#### muw (MUW-like)

Core characteristics:
- amplitude embedding on all qubits
- repeated convolution + pooling blocks (pairwise IsingXX/YY/ZZ entanglers + controlled pooling)
- a final **ArbitraryUnitary** block on the remaining pooled qubits
- output is the probability of measuring `|1⟩` on a designated qubit

Pros:
- high expressivity at the end due to the ArbitraryUnitary block
- structurally close to the MUW-style QCNN pattern that motivated this code path

Cons:
- optimization can be brittle on small datasets or noisy targets (flat gradients / poor conditioning)
- the ArbitraryUnitary parameterization can make training less stable and slower

#### alt (alternative)

Core characteristics:
- amplitude embedding on all qubits
- convolution + pooling blocks
- a final **StronglyEntanglingLayers** block as the trainable “dense” head
- output is the probability of measuring `|1⟩` on a designated qubit

Pros:
- usually easier to optimize than a full ArbitraryUnitary head
- tends to be more stable for quick demonstrations and debugging

Cons:
- potentially less expressive than an ArbitraryUnitary head in the final pooled space
- still subject to simulator scaling limits as `n_qubits` grows
