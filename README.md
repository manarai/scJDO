
# scQDiff
## 🧬 scQDiff: Schrödinger Bridge Learning of Single-Cell Regulatory Dynamics
**scQDiff** learns a **time-conditioned drift field** and extracts **regulatory archetypes** from single-cell data using a Schrödinger-Bridge / score-based formulation—then simulates **forward** (naive→perturbed) and **reverse** transitions. Includes **graph Laplacian** priors, **Fokker–Planck residual**, **AnnData/scVelo/CellRank** support, **fate-conditioned archetypes**, and **trajectory simulation** utilities.

## Install
```bash
# Create conda env
conda create -n scqdiff python=3.10
conda activate scidiff

# Clone the repository
git clone https://github.com/manarai/scQDiff.git
cd sQIDiff

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

```
# 🌌 Mathematical Foundations

Given single-cell data across developmental or perturbation states, we model a continuous stochastic process:

\[
dX_t = u(X_t, t)\,dt + \sqrt{2\beta}\,dW_t
\]

where  

- \(X_t \in \mathbb{R}^d\): cell-state vector (e.g., gene expression)  
- \(u(X_t,t)\): **drift field**, describing deterministic gene-regulatory flow  
- \(\beta\): diffusion constant  
- \(dW_t\): Brownian noise  

scQDiff learns \(u(x,t)\) such that the simulated process transports the empirical distribution \(\rho_0\) (naïve) into \(\rho_1\) (perturbed) while minimizing the Schrödinger Bridge energy:

\[
\min_u \;\mathrm{KL}(P^u \,\Vert\, P^{\text{ref}}) \quad \text{s.t. } \rho_0, \rho_1 \text{ fixed.}
\]

The resulting field \(u(x,t)\) provides **directionality**, **causality**, and **temporal structure** for cellular transitions.

---

## 🔍 What scQDiff Learns

| Layer | Description | Output |
|-------|--------------|---------|
| **Regulatory drift** | Deterministic direction of transcriptional change | \(u(x,t)\) |
| **Temporal Jacobian** | Local causal gene–gene influences | \(J(t)=\frac{\partial u}{\partial x}\) |
| **Regulatory archetypes** | Low-rank temporal modes summarizing core programs | \(A_k, a_k(t)\) |
| **(Optional) Communication coupling** | Dynamic ligand–receptor signaling (via CellPhoneDB) | \(W_{ij}(t)\) |

---

## 🔗 Extension: scQDiff-Comm

The **scQDiff-Comm** module introduces **cell–cell communication coupling** using ligand–receptor data (e.g., CellPhoneDB):

\[
dX_t^{(i)} = u_{\text{intra}}(X_t^{(i)},t)\,dt
+ \sum_{j} W_{ij}(t)\, f(X_t^{(j)}-X_t^{(i)})\,dt
+ \sqrt{2\beta}\,dW_t^{(i)}.
\]

Here \(W_{ij}(t)\) encodes interaction strength between cells \(i\) and \(j\) over time, revealing dynamic communication archetypes such as **inflammatory relay**, **fibrosis loop**, or **exhaustion resolution**.

---

## 🧮 Core Features

- **Time-continuous modeling** of differentiation or perturbation trajectories  
- **Regulatory Jacobian tensors** capturing causal gene influence  
- **Forward & reverse simulation** to test hypothetical reprogramming paths  
- **Integration with RNA velocity, ATAC, and metabolomic embeddings**  
- **Cell-cell communication layer** (optional) using CellPhoneDB  
- **Cytoscape-exportable dynamic networks** (`.graphml`, `.cyjs`)  

---

## ⚙️ Installation


