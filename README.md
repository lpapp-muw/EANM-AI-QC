# EANM-AI-QC
EANM AI Committee - Quantum Computing Engagement Plan

## Introduction
This repository contains basic teaching material, tutorials and educational videos for those interested to engage with quantum computing (QC) within the Euopean Association of Nuclear Medicine (EANM). Thus, the collection of materials here is intended for beginners with little to no knowledge in QC. 
In addition, we also share advanced materials containing tutorials, hands-on as well as code repositories.
The repository is intended to be updated regularly, with eventually having an EANM AI Committee-created code base for QC as well.

## QC Concepts (Beginner)

IBM Quantum “Understanding Quantum Information and Computation” – Beginner-level video series (free, 16 parts). A rigorous yet accessible introduction to quantum computing fundamentals, with video lectures by IBM’s John Watrous and accompanying text. Emphasizes core concepts over heavy math, making it ideal for newcomers.
[Link](https://www.ibm.com/quantum/blog/understanding-quantum-information-and-computation)

FutureLearn – Understanding Quantum Computers – Beginner MOOC (Keio Univ.). An introductory 4-week course covering key quantum computing concepts with minimal mathematics. Explains qubits, superposition, entanglement, and potential applications in plain language.
[Link](https://qosf.org/learn_quantum/)

openHPI: Introduction to Quantum Computing with Qiskit – Beginner course. Free course by IBM Quantum and HPI, teaching basic quantum computing using Qiskit in the cloud. Covers qubits, circuits, and simple algorithms with hands-on demos (no prior QC experience required).
[Link](https://open.hpi.de/courses/qc-qiskit2022)

## Hands-On Tutorials (Beginner to Intermediate)

Qiskit Textbook & Tutorials – Interactive Jupyter Notebooks (Beginner-friendly). The open-source Qiskit Textbook offers step-by-step notebooks from basic quantum circuits to algorithms.
[Link](https://medium.com/qiskit/how-to-start-experimenting-with-quantum-image-processing-283dddcc6ba0)

https://github.com/Qiskit/textbook/tree/main/notebooks/ch-applications

Qiskit’s official tutorial repository also includes a Quantum CNN notebook implementing a QCNN to classify simple images:
https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html

PennyLane Demos and Codebook – Hands-on QML tutorials (Beginner-Intermediate). Xanadu’s PennyLane provides extensive demonstrations on quantum machine learning. For example, the Quanvolutional Neural Network demo shows how to apply a quantum filter to MNIST images (quantum analog of CNN) and combine with classical networks:
https://pennylane.ai/qml/demos/tutorial_quanvolution

The interactive Quantum Codebook is also available for guided practice on topics from basic qubit operations to variational algorithms:
https://monitsharma.github.io/Learn-Quantum-Computing-For-Free/

AWS Braket Example: Hybrid Quantum ML – Intermediate tutorial (AWS platform). Amazon Braket’s public examples include a Quantum Machine Learning Hybrid Jobs notebook demonstrating a typical hybrid workflow (classical optimizer + quantum circuit model) on AWS. It guides users through data upload, setting hyperparameters, and running a simple QML algorithm on cloud simulators or QPUs. (Requires an AWS account; free tier available for small-scale use):
https://github.com/amazon-braket/amazon-braket-examples


## Simple Code Repositories (Quantum ML & QNN Workflows)

Hybrid QNN on MNIST (Qiskit + PyTorch) – Platform: Qiskit, Skill: Intermediate. A lightweight community example of a quantum-classical neural network for image classification. Implements a classical CNN feature extractor with a quantum fully-connected layer for MNIST digit data: 
https://github.com/dohun-qml/quantum-neural-network

Quantum-Classical CNN (CIFAR-10 Binary Classifier) – Platform: Qiskit/PyTorch, Skill: Intermediate. End-to-end example of a variational quantum classifier on real images. Uses a parametrized quantum circuit as one layer in a CNN to distinguish two classes of CIFAR-10 (airplane vs. automobile): 
https://github.com/DRA-chaos/Quantum-Classical-Hyrid-Neural-Network-for-binary-image-classification-using-PyTorch-Qiskit-pipeline

Qiskit Community Tutorials – QSVM & VQE demos – Platform: Qiskit, Skill: Beginner. The Qiskit community notebooks include simple machine learning and optimization demos. For instance, a Quantum SVM classifier for a toy image dataset (mapping pixel data to qubit feature maps and using QSVM): 
https://github.com/qiskit-community/ibm-quantum-challenge-fall-2021/blob/main/content/challenge-3/challenge-3.ipynb


## Reviews - Medical (Imaging) Use Cases

Review of medical image processing using quantum-enabled algorithms (2024) – Fei Yan et al. – Artificial Intelligence Review. A comprehensive review of how quantum and quantum-inspired algorithms enhance traditional medical image processing, covering applications in disease diagnosis and medical image security. The authors survey advances in quantum-assisted image analysis (e.g. segmentation, optimization methods) and outline current performance limitations and future development plans for quantum techniques in imaging:
https://link.springer.com/article/10.1007/s10462-024-10932-x

Quantum Machine and Deep Learning for Medical Image Classification: A Systematic Review of Trends, Methodologies, and Future Directions (2025) – Eman A. Radhi et al. – Iraqi Journal for Computer Science and Mathematics. A PRISMA-guided systematic review of 28 studies (2018–2024) on quantum learning techniques for medical image classification. It charts the use of quantum support vector machines (QSVM), quantum convolutional neural networks (QCNN), and other hybrid quantum-classical models across tasks like brain tumor detection, skin lesion identification, and COVID-19 diagnosis, noting promising accuracy gains and discussing challenges in image encoding, hardware scalability, and model interpretability:
https://ijcsm.researchcommons.org/ijcsm/vol6/iss2/9/

Quantum machine learning in medical image analysis: A survey (2023) – Lin Wei et al. – Neurocomputing. A pioneering survey examining the nascent applications of quantum machine learning in medical image analysis. It provides a taxonomy of early QML approaches in imaging and is noted as the first comprehensive review in this subfield:
https://www.sciencedirect.com/science/article/pii/S0925231223000589

Quantum algorithms and complexity in healthcare applications: a systematic review with machine learning-optimized analysis (2025) – Agostino Marengo & Vito Santamato – Frontiers in Computer Science. A systematic literature review that uniquely combines text-mining (topic modeling with PSO-LDA) to map quantum computing research in healthcare. Analyzing 63 studies, the authors identify two dominant themes: quantum computing for AI in healthcare (e.g. quantum machine learning for diagnostics and predictive analytics) and quantum computing for health data security (e.g. quantum cryptography for EHR/privacy). The review highlights theoretical advances from quantum-enhanced algorithms for biomedical data analysis to blockchain-like security frameworks, and it concludes that quantum algorithms show promise to optimize complex diagnostic computations and protect medical data:
https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1584114/full

Quantum Computing in Medicine (2024) – James C. L. Chow – Medical Sciences (Basel). A broad open-access review covering the state of quantum computing in medical research and practice. It outlines fundamental QC concepts (qubits, superposition, entanglement) and surveys milestone applications: from quantum-accelerated drug discovery and molecular modeling in chemistry/proteomics, to quantum approaches in genomics (e.g. DNA sequencing) and medical diagnostics. Quantum machine learning techniques are explained (including quantum-enhanced imaging and radiotherapy simulations), and the article discusses practical challenges such as hardware scalability, error correction, and integration into clinics. Looking ahead, Chow describes prospects for quantum–classical hybrid systems and emerging quantum hardware that could accelerate adoption of QC in personalized therapy planning and complex biological simulations:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11586987/

