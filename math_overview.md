\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{xcolor}

\geometry{a4paper, margin=1in}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=cyan,
    citecolor=magenta
}

\title{scIDiff -- Mathematical Overview}
\subtitle{Time-resolved regulatory operators for single-cell dynamics}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Introduction}
\texttt{scIDiff} (single-cell Inference of Differential operators) is an operator-centric framework for learning the time-dependent dynamical rules that govern cellular state transitions. Rather than only reconstructing trajectories, \texttt{scIDiff} infers the local Jacobian operators of a learned drift field and uses their temporal structure to reveal conserved regulatory programs (``operator archetypes'') and decision-sensitive control modes.

RNA velocity and Schr\"odinger Bridge constraints can be used as optional priors when learning the drift field, but all downstream operator and archetype analysis is independent of these choices.

\section{Stochastic model of cell-state dynamics}
Let \( X(t) \in \mathbb{R}^d \) denote the latent cellular state at pseudotime \( t \in [0, 1] \).

Cell dynamics are modeled as the stochastic differential equation
\[
dX(t) = f(X(t), t)\,dt + \sqrt{2\beta}\,dW(t)
\]
where
\begin{itemize}
    \item \( f(x,t) \) is the time-dependent drift field,
    \item \( \beta > 0 \) controls stochasticity,
    \item \( W(t) \) is a standard Brownian motion.
\end{itemize}
The drift \( f(x,t) \) represents the instantaneous regulatory program driving cellular motion in latent state space.

\section{Density evolution}
Let \( \rho(t, x) \) denote the probability density of \( X(t) \).  
It evolves according to the Fokker--Planck equation
\[
\frac{\partial \rho}{\partial t} = -\nabla \cdot \bigl( \rho f \bigr) + \beta \Delta \rho.
\]
\texttt{scIDiff} learns \( f(x,t) \) so that this stochastic flow explains the observed single-cell distribution over pseudotime.

\section{Local regulatory operators (Jacobians)}
The central object in \texttt{scIDiff} is the time-resolved Jacobian operator
\[
J(x,t) = \nabla_x f(x,t).
\]
This governs the linearized perturbation dynamics
\[
d(\delta X) = J(X,t)\,\delta X\,dt + \sqrt{2\beta}\,dW(t).
\]
The eigenvalues of \( J(x,t) \) define:
\begin{itemize}
    \item \textbf{Negative} \(\rightarrow\) stable (buffered states),
    \item \textbf{Near-zero} \(\rightarrow\) plastic (fragile states),
    \item \textbf{Positive} \(\rightarrow\) unstable (fate-deciding control directions).
\end{itemize}
These operators quantify stability, sensitivity, and commitment directly.

\section{Temporal Jacobian tensor}
Evaluating Jacobians along pseudotime gives
\[
J(t_1),\; J(t_2),\;\dots,\; J(t_T).
\]
Stacking them forms the \emph{temporal Jacobian tensor}
\[
\mathcal{T} \in \mathbb{R}^{T \times d \times d},
\]
which summarizes how regulatory sensitivity evolves during differentiation.

\section{Operator archetypes}
\texttt{scIDiff} factorizes the temporal tensor as
\[
\mathcal{T}(t) \approx \sum_{k=1}^{K} a_k(t) A_k,
\]
where
\begin{itemize}
    \item \( A_k \) are \emph{operator archetypes} (\( d \times d \) matrices),
    \item \( a_k(t) \) are their time-dependent activations.
\end{itemize}
Sequential and concurrent archetype activation defines a conserved regulatory grammar of cell fate.

\section{Control-relevant modes}
Eigenvectors of \( A_k \) with positive eigenvalues are \emph{unstable modes}: perturbations along these directions are amplified.

When projected into gene space, these modes identify control-relevant transcriptional programs.

\section{RNA velocity as an optional drift prior}
RNA velocity can guide drift learning via a reference field
\[
b(x,t) = \lambda\,g(t)\,w(x)\,\hat{v}(x),
\]
where
\begin{itemize}
    \item \( \hat{v}(x) \) is RNA velocity,
    \item \( w(x) \) is a confidence weight,
    \item \( g(t) \) is a temporal gate,
    \item \( \lambda \) controls strength.
\end{itemize}
The learned drift is
\[
f(x,t) = b(x,t) + u_\theta(x,t).
\]
RNA velocity improves directional alignment but does not alter Jacobian or archetype analysis.

\section{Endpoint-constrained dynamics (Schr\"odinger Bridge, optional)}
If start and end populations are known, \texttt{scIDiff} enforces them by minimizing control energy
\[
\min_u \mathbb{E}\left[ \int \| u(X,t) \|^2 dt \right]
\]
subject to
\[
\frac{\partial \rho}{\partial t} = -\nabla\cdot\bigl(\rho\,(b+u)\bigr) + \beta \Delta \rho,
\qquad
X(0) \sim \rho_0,\; X(1) \sim \rho_1.
\]
This changes only the drift field; Jacobians and archetypes are computed the same way.

\section*{Summary}
\texttt{scIDiff} infers time-resolved regulatory operators from single-cell data.  
By learning a drift field, computing its Jacobian, and decomposing the resulting temporal tensor into archetypes, \texttt{scIDiff} reveals a conserved grammar of fate decisions and identifies control-relevant unstable modes.

RNA velocity and Schr\"odinger Bridge provide optional guidance, but the core object of \texttt{scIDiff} is the time-varying regulatory operator.

\end{document}
