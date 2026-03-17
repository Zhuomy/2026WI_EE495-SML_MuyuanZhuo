# System Identification of Damped Harmonic Oscillators via Physics-Informed HNNs

**Northwestern University - EE495 Scientific Machine Learning (Final Project)** **Author:** Muyuan Zhuo

[![Presentation Video](https://www.youtube.com/watch?v=9uV-9g-HunI)]
[![Report](https://img.shields.io/badge/PDF-Project_Report-blue)](./report/EE495_SML_Project.pdf)

## 📌 Project Overview
This project extends the concept of **Hamiltonian Neural Networks (HNNs)** to non-conservative systems. Standard Neural Networks often overfit when exposed to sparse and noisy sensor data. While standard HNNs solve this by embedding energy conservation laws, they strictly assume conservative systems (where energy does not decay). 

In this repository, we propose a **Damped-HNN**. By hard-coding a learnable dissipation term into the network's physical priors, the model can:
1. Robustly reconstruct the phase space trajectory from severely degraded data (80 sparse points + 5% noise).
2. Perform accurate system identification by extracting the hidden damping coefficient.

---

## 🧮 Math Development (Physics Prior)

For an ideal, frictionless Hamiltonian system, the state variables $(q, p)$ evolve according to Hamilton's equations:

$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

To model a non-conservative system, we introduce a dissipation term characterized by a damping coefficient $c$. The modified governing equations become:

$$\begin{bmatrix} \dot{q} \\ \dot{p} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -1 & -c \end{bmatrix} \begin{bmatrix} \frac{\partial H}{\partial q} \\ \frac{\partial H}{\partial p} \end{bmatrix}$$

In our neural network architecture, $H_\theta(q,p)$ learns the unknown Hamiltonian, and the damping coefficient $c$ is treated as a learnable scalar parameter $c_\phi$. Automatic differentiation is used to compute the spatial gradients $\partial H / \partial q$ and $\partial H / \partial p$.

---

## 🚀 Reproducibility (How to Run)

The code is structured to be completely reproducible with fixed random seeds. 

**1. Install Dependencies**
```bash
pip install -r requirements.txt