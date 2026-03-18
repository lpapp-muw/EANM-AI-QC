# EANM AI Committee — Quantum Computing Engagement Resources

This repository contains teaching material, tutorials, and educational videos for those interested in engaging with quantum computing (QC) within the European Association of Nuclear Medicine (EANM). The collection is intended for beginners with little to no knowledge in QC, as well as advanced materials with hands-on tutorials and code repositories.

The initial version was made for the perspective paper:

> The Dawn of Quantum AI in Nuclear Medicine: an EANM Perspective. L. Papp, D. Visvikis, M. Sollini, K. Shi, M. Kirienko. The EANM Journal, 2026 (in revision).

If you are new to the topic of QC, we recommend processing this repository from top to bottom.

If you are ready to engage with source code custom-tailored for nuclear medicine use cases, see [CLARYON — The Code Repository](#claryon--the-code-repository).

---

## QC Concepts — YouTube Videos

Before doing anything else, we recommend getting familiar with the concept of QC by going through these videos:

**Quantum Computers Explained — Limits of Human Technology** — Kurzgesagt — In a Nutshell (2015, ~7 min). An animated overview of why classical computers face physical limits and how quantum bits (qubits) leverage superposition and entanglement to compute beyond those limits:
- [Quantum Computers Explained – Limits of Human Technology](https://youtu.be/JhHMJCUmq28)

**Veritasium** has multiple videos about QC and quantum mechanics, utilizing beautiful visuals and intuitive explanations:
- [How Does a Quantum Computer Work?](https://www.youtube.com/watch?v=g_IaVepNDT4)
- [What makes quantum computers SO powerful?](https://www.youtube.com/watch?v=-UrdExQW0cs)
- [Something Strange Happens When You Trust Quantum Mechanics](https://youtu.be/qJZ1Ez28C-A?si=Ejk39rgtDCFM8ZAC)
- [Is This What Quantum Mechanics Looks Like?](https://www.youtube.com/watch?v=WIyTZDHuarQ)
- [There Is Something Faster Than Light](https://www.youtube.com/watch?v=NIk_0AW5hFU)
- [Parallel Worlds Probably Exist. Here's Why](https://www.youtube.com/watch?v=kTXTPe3wahc)

**Quantum Computers, Explained** — Cleo Abram (Huge If True). Clear explanations of qubits vs bits, why quantum computers are important, and applications like simulating molecules for medicine:
- [Quantum Computers, explained with MKBHD](https://www.youtube.com/watch?v=e3fz3dqhN44)

**3Blue1Brown** — A visually rich exploration of quantum computing through the lens of math. Derives how qubit state vectors work and walks through Grover's search algorithm step by step (advanced):
- [But what is quantum computing? (Grover's Algorithm)](https://www.youtube.com/watch?v=RQWpF2Gb-gU)

---

## QC Concepts — Teaching Material (Beginner)

Once you have familiarized yourself with the videos above, you may proceed to the following teaching material:

**IBM Quantum "Understanding Quantum Information and Computation"** — Beginner-level video series (free, 16 parts). A rigorous introduction to quantum computing fundamentals, with video lectures by IBM's John Watrous and accompanying text:
- [Qiskit playlist — Understanding Quantum Computing](https://www.youtube.com/playlist?list=PLOFEBzvs-VvqKKMXX4vbi4EB1uaErFMSO)
- [Basics of Quantum Information](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information)

**FutureLearn — Understanding Quantum Computers** — Beginner MOOC (Keio Univ.). An introductory 4-week course covering key quantum computing concepts with minimal mathematics:
- [Learning Resources on Quantum Computing](https://qosf.org/learn_quantum/)

**openHPI: Introduction to Quantum Computing with Qiskit** — Beginner course. Free course by IBM Quantum and HPI, teaching basic quantum computing using Qiskit in the cloud:
- [Introduction to Quantum Computing with Qiskit (with IBM Quantum)](https://open.hpi.de/courses/qc-qiskit2022)

---

## Hands-On Tutorials (Beginner to Intermediate)

**Qiskit Textbook & Tutorials** — Interactive Jupyter Notebooks (beginner-friendly). Step-by-step notebooks from basic quantum circuits to algorithms:
- [Qiskit Quantum Image Processing](https://medium.com/qiskit/how-to-start-experimenting-with-quantum-image-processing-283dddcc6ba0)
- [Qiskit Textbook Notebooks](https://github.com/Qiskit/textbook/tree/main/notebooks/ch-applications)

**Qiskit Quantum CNN** — Official tutorial implementing a QCNN to classify simple images:
- [Quantum Convolutional Neural Networks](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html)

**PennyLane Demos and Codebook** — Hands-on QML tutorials (beginner-intermediate). Xanadu's PennyLane provides extensive demonstrations on quantum machine learning, including the Quanvolutional Neural Network demo:
- [PennyLane Quanvolution Tutorial](https://pennylane.ai/qml/demos/tutorial_quanvolution)
- [Quantum Codebook — Guided Practice](https://monitsharma.github.io/Learn-Quantum-Computing-For-Free/)

**AWS Braket Example: Hybrid Quantum ML** — Intermediate tutorial (AWS platform). Amazon Braket's public examples include a Quantum Machine Learning Hybrid Jobs notebook:
- [Amazon Braket Examples](https://github.com/amazon-braket/amazon-braket-examples)

**Hybrid QNN on MNIST (Qiskit + PyTorch)** — A lightweight community example of a quantum-classical neural network for image classification:
- [Quantum Neural Network](https://github.com/dohun-qml/quantum-neural-network)

**Quantum-Classical CNN (CIFAR-10 Binary Classifier)** — End-to-end example of a variational quantum classifier on real images:
- [Quantum-Classical Hybrid Neural Network](https://github.com/DRA-chaos/Quantum-Classical-Hyrid-Neural-Network-for-binary-image-classification-using-PyTorch-Qiskit-pipeline)

**Qiskit Community Tutorials — QSVM & VQE demos** — Simple machine learning and optimization demos:
- [Quantum SVM Challenge](https://github.com/qiskit-community/ibm-quantum-challenge-fall-2021/blob/main/content/challenge-3/challenge-3.ipynb)

---

## Reviews

In case you are interested in where the field stands:

**Review of medical image processing using quantum-enabled algorithms (2024)** — Fei Yan et al. A comprehensive review of how quantum and quantum-inspired algorithms enhance traditional medical image processing:
- [Link](https://link.springer.com/article/10.1007/s10462-024-10932-x)

**Quantum machine learning in medical image analysis: A survey (2023)** — Lin Wei et al. — Neurocomputing. The first comprehensive review of QML in medical image analysis:
- [Link](https://www.sciencedirect.com/science/article/pii/S0925231223000589)

**Quantum algorithms and complexity in healthcare applications (2025)** — Agostino Marengo & Vito Santamato — Frontiers in Computer Science. Analyzing 63 studies on quantum computing for AI in healthcare and health data security:
- [Link](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1584114/full)

**Quantum Computing in Medicine (2024)** — James C. L. Chow — Medical Sciences (Basel). An open-access review covering quantum computing in medical research and practice:
- [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11586987/)

For a large, actively maintained collection of QC papers related to AI/ML, see Christophe Pere's repository:
- [Roadmap to QML](https://github.com/Christophe-pere/Roadmap-to-QML)

---

## CLARYON — The Code Repository

All source code for quantum and classical ML in nuclear medicine has been consolidated into **CLARYON** (CLassical-quantum AI for Reproducible Explainable OpeN-source medicine):

**Repository**: [https://github.com/lpapp-muw/CLARYON](https://github.com/lpapp-muw/CLARYON)

CLARYON provides:
- 21 registered models (gradient boosting, neural networks, 9 quantum ML methods, CNNs)
- YAML-driven experiment pipeline with reproducible cross-validation
- mRMR feature selection and preprocessing (automatically adapted for quantum models)
- SHAP and LIME explainability with publication-ready plots
- Structured LaTeX methods and results sections auto-generated from config
- Geometric Difference score for quantum advantage assessment (Huang et al., 2021)
- Model complexity presets (quick/small/medium/large/exhaustive/auto) for non-expert users
- NIfTI medical imaging support
- 12 curated medical benchmark datasets including a real PSMA-11 PET radiomics dataset
- Inference mode for deploying trained models on new patient data

### Quick start

```bash
pip install claryon[all]
claryon -v run -c config.yaml
```

See the [CLARYON README](https://github.com/lpapp-muw/CLARYON) for full documentation, installation, and usage instructions.

---

## Related Repositories

| Repository | Description |
|---|---|
| [CLARYON](https://github.com/lpapp-muw/CLARYON) | Production framework — all models, pipeline, evaluation, reporting, datasets |
| [Roadmap to QML](https://github.com/Christophe-pere/Roadmap-to-QML) | Curated collection of QC/QML papers |

---

## References

### Quantum ML in Nuclear Medicine

- Moradi S, Brandner C, Spielvogel C, Krajnc D, Hillmich S, Wille R, Drexler W, Papp L. "Clinical data classification with noisy intermediate scale quantum computers." *Scientific Reports* 12, 1851 (2022). [https://doi.org/10.1038/s41598-022-05971-9](https://doi.org/10.1038/s41598-022-05971-9)

- Moradi S, Spielvogel C, Krajnc D, Brandner C, Hillmich S, Wille R, Traub-Weidinger T, Li X, Hacker M, Drexler W, Papp L. "Error mitigation enables PET radiomic cancer characterization on quantum computers." *Eur J Nucl Med Mol Imaging* 50, 3826-3837 (2023). [https://doi.org/10.1007/s00259-023-06362-6](https://doi.org/10.1007/s00259-023-06362-6)

- Papp L, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.

### Quantum Advantage Assessment

- Huang H-Y, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, McClean JR. "Power of data in quantum machine learning." *Nature Communications* 12, 2631 (2021). [https://doi.org/10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9)

### Quantum ML Foundations

- Havlicek V, Corcoles AD, Temme K, Harrow AW, Kandala A, Chow JM, Gambetta JM. "Supervised learning with quantum-enhanced feature spaces." *Nature* 567, 209-212 (2019). [https://doi.org/10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)

- Schuld M, Petruccione F. *Supervised Learning with Quantum Computers*. Springer (2018). [https://doi.org/10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9)

---

## History

This repository originally contained the EANM-AI-QC source code (v0.8.0), a quantum ML framework for nuclear medicine. In 2026, the codebase was consolidated into [CLARYON](https://github.com/lpapp-muw/CLARYON), which extends it with classical models, preprocessing, feature selection, model presets, benchmark datasets, and publication-ready reporting. This repository is retained as an educational hub and collection of teaching resources for the EANM AI Committee.

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.
