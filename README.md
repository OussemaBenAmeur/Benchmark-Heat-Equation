# Heat Equation Benchmark

A reproducible comparison of physics-informed neural networks, spline-based Kolmogorov–Arnold networks, and a classical finite difference solver for the one-dimensional heat equation.

The experiment compares accuracy, training cost, inference latency, and numerical reuse on the same initial–boundary value problem. All implementations, checks, plots, and recorded outputs are in [the main notebook](PIKAN_Heat_Equation.ipynb).

## Problem

The temperature field satisfies

$$
\frac{\partial T}{\partial t}=\alpha\frac{\partial^2 T}{\partial x^2},
\qquad (x,t)\in[0,1]^2,\quad \alpha=0.1,
$$

with initial condition $T(x,0)=\sin(\pi x)$ and homogeneous boundary conditions $T(0,t)=T(1,t)=0$. Its exact solution is

$$
T(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
$$

The networks train on the PDE residual, initial condition, and boundary conditions. Interior temperatures from the exact solution are reserved for evaluation.

## Methods

| Method | Implementation | Comparison |
|---|---|---|
| PINN | Tanh MLP with two hidden layers of width 20; 501 parameters | 400 Adam updates across five seeds |
| PIKAN-8 | Learned cubic B-spline edge functions with eight centers; 499 parameters | Same collocation points, loss weights, optimizer, and update count as PINN |
| PINN-time | Same MLP as PINN | Approximately the same training time as PIKAN-8 for each seed |
| PIKAN-12 | Twelve spline centers; 715 parameters | Exploratory run using seed 0 |
| Crank–Nicolson | Sparse finite differences with a reused LU factorization | 100 space and 100 time intervals; additional resolution sweep |
| Exact formula | Direct evaluation of the known solution | Reference for accuracy and query cost |

The spline network uses fixed knots and hidden tanh activations. It is a KAN-style variant, and its results apply to this implementation.

## Run the notebook

The recorded notebook uses Python 3.12 and PyTorch on CPU. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy pandas scipy matplotlib jupyterlab
.venv/bin/jupyter lab PIKAN_Heat_Equation.ipynb
```

Select the environment's Python kernel and run all cells in order. The committed notebook also contains outputs that can be inspected without rerunning it.

The configuration cell sets 400 updates, seeds `0`–`4`, 96 interior points, 48 initial points, 48 boundary points, and an Adam learning rate of `3e-3`. The physics, initial, and boundary loss weights are `1`, `10`, and `10`. Lower `STEPS` or shorten `SEEDS` for a draft run, then rerun the entire notebook before comparing results.

## Recorded results

Temperature error is measured on a held-out 101 × 101 grid. Values below come from the committed notebook; timings depend on hardware and software versions.

| Method | Runs | Mean relative L2 error | Mean training time |
|---|---:|---:|---:|
| PINN | 5 | 0.019404 | 0.438 s |
| PIKAN-8 | 5 | 0.010034 | 3.355 s |
| PINN-time | 5 | 0.004315 | 3.356 s |
| PIKAN-12 | 1 | 0.010143 | 3.762 s |

Crank–Nicolson reaches relative L2 error of approximately `3.17e-5` on the same grid, with a recorded full-field solve time of approximately `1.43 ms`.

At equal update counts, PIKAN-8 achieves lower mean temperature error than PINN while taking approximately 7.65 times as long to train. At approximately equal training time, PINN-time achieves the lower mean error. The classical solver is more accurate and cheaper for this fixed problem.

The notebook also compares repeated full-field requests and 1,024 new query coordinates. Cached numerical fields and interpolation are included alongside fresh solves and neural inference, so reuse costs can be assessed directly.

## Validation and scope

The notebook checks the exact solution's PDE residual, spline second derivatives against finite differences, model curvature, finite losses, numerical boundary conditions, and grid convergence. Evaluation includes temperature RMSE, relative L2 error, held-out PDE residuals, initial and boundary errors, and timing measurements.

This is a controlled study of one smooth PDE instance with a known solution. The networks accept only `(x, t)`; changing diffusivity, initial conditions, or boundary conditions requires a new training problem. Five matched seeds and one exploratory spline-grid change do not establish a general ranking of PINNs and KANs.

## Files

| File | Contents |
|---|---|
| [PIKAN_Heat_Equation.ipynb](PIKAN_Heat_Equation.ipynb) | Complete experiment, validation checks, figures, and measured conclusions |
| [Academic project proposal](PIKAN_Heat_Equation_Academic_Project_Proposal.pdf) | Original project motivation and proposed comparison |

References and possible extensions, including families of PDE instances and inverse problems, are discussed at the end of the notebook.
