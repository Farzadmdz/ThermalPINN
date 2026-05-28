
# ThermalPINN: A Physics-Informed Neural CFD Framework for Conjugate Heat Transfer and Advanced Vapor Chambers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/backend-PyTorch-orange)](https://pytorch.org/)
[![Domain](https://img.shields.io/badge/Domain-Aerospace%20%26%20Thermal-red)]()
[![Research](https://img.shields.io/badge/Research-Scientific%20Computing-purple)]()

---

## Abstract

**ThermalPINN** is an open-source scientific computing framework designed for high-fidelity simulation of incompressible fluid flow, conjugate heat transfer (CHT), and advanced thermal-management systems through the deep integration of Physics-Informed Neural Networks (PINNs) with conventional Computational Fluid Dynamics (CFD) workflows. The framework specifically addresses multi-physics engineering challenges arising in advanced bi-porous vapor chambers, aerospace thermal control architectures, high-power electronics cooling, compact heat exchangers, and microchannel heat sinks, wherein accurate coupling between solid and fluid thermal fields remains critically important. By embedding the governing partial differential equations directly into the neural network optimization landscape, ThermalPINN enables differentiable, mesh-free, and data-efficient surrogate modeling that accelerates scientific design-space exploration while rigorously preserving physical consistency.

---

## Scientific Motivation

Modern aerospace and electronics systems increasingly operate under aggressive thermal loads that push conventional CFD solvers to their computational limits during optimization, uncertainty quantification, and transient multi-physics analysis. Traditional finite-volume and finite-element methods, while accurate, impose substantial burdens: they demand extremely dense meshes, consume large computational resources, require long simulation times, and necessitate expensive geometry re-meshing procedures whenever the design space evolves. Physics-Informed Neural Networks offer an alternative approach by incorporating governing physical laws directly into the optimization objective of deep neural networks. Rather than relying exclusively on numerical discretization, ThermalPINN learns continuous field representations that are inherently constrained by conservation laws and boundary conditions. This hybrid CFD–AI methodology enables rapid thermal-field prediction, differentiable surrogate modeling, accelerated optimization workflows, reduced computational overhead, and improved scalability for high-dimensional engineering problems.

---

## Core Features

### Physics-Informed CFD Solver

ThermalPINN implements a PINN-based solver for both steady and transient incompressible Navier–Stokes equations, with full support for conjugate heat transfer modeling across coupled solid–fluid domains. The architecture readily accommodates two-dimensional geometries and remains extensible to three-dimensional computational domains.

### Advanced Vapor Chamber Modeling

The framework provides dedicated support for bi-porous wick structures, capillary-driven thermal transport, porous-media thermal resistance modeling, and Darcy–Forchheimer momentum corrections—capabilities essential for capturing the complex physics of advanced two-phase thermal spreaders.

### Hybrid CFD–PINN Workflow

ThermalPINN seamlessly integrates with OpenFOAM, Fluent, and scientific CFD datasets. It supports supervised pretraining from high-fidelity CFD fields and enforces residual-based physics regularization to ensure that learned solutions remain physically admissible.

### Scientific Visualization Pipeline

The framework exports ParaView-compatible VTK/VTU datasets, maintains MATLAB interoperability, and automatically generates 600 DPI journal-ready figures. Residual convergence visualization and high-fidelity thermal contour rendering are provided as standard outputs.

---

## Mathematical Formulation

At the core of ThermalPINN lies a neural operator that maps spatiotemporal coordinates to the complete thermofluid state:

$$\mathcal{N}_{\theta} : (t, \mathbf{x}) \rightarrow (u, v, w, p, T_f, T_s)$$

where $u$, $v$, $w$ denote the velocity components, $p$ represents pressure, $T_f$ signifies fluid temperature, and $T_s$ denotes solid temperature.

---

## Governing Equations

### Continuity Equation

$$\nabla \cdot \mathbf{u} = 0$$

### Momentum Conservation

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \frac{1}{\mathrm{Re}} \nabla^2 \mathbf{u}$$

### Energy Equation (Fluid)

$$\frac{\partial T_f}{\partial t} + (\mathbf{u} \cdot \nabla) T_f = \frac{1}{\mathrm{Pe}} \nabla^2 T_f$$

### Heat Conduction in Solid Region

$$\mathrm{Fo}_s \frac{\partial T_s}{\partial t} = \nabla^2 T_s$$

### Fluid–Solid Interface Conditions

$$T_f = T_s$$

$$-k_f \nabla T_f \cdot \mathbf{n} = -k_s \nabla T_s \cdot \mathbf{n}$$

---

## PINN Residual Loss Function

The composite optimization objective is defined as:

$$\mathcal{L}(\theta) = \lambda_{\mathrm{data}} \mathcal{L}_{\mathrm{data}} + \lambda_{\mathrm{NS}} \mathcal{L}_{\mathrm{NS}} + \lambda_{\mathrm{E}} \mathcal{L}_{\mathrm{E}} + \lambda_{\mathrm{BC}} \mathcal{L}_{\mathrm{BC}} + \lambda_{\mathrm{CHT}} \mathcal{L}_{\mathrm{CHT}}$$

Each term corresponds to physics residuals, boundary constraints, supervised CFD fields, and interface coupling conditions. Automatic differentiation via PyTorch evaluates all spatial and temporal derivatives continuously without explicit mesh discretization.

---

## Repository Architecture

```text
ThermalPINN/
├── configs/
│   ├── vapor_chamber_baseline.yml
│   └── biporous_wick_cht.yml
├── data/
│   └── cfd_reference/
├── docs/
│   └── images/
│       └── ThermalPINN_Validation.png
├── examples/
├── models/
│   ├── pinn_core.py
│   └── network_arch.py
├── physics/
│   ├── navier_stokes.py
│   ├── cht_interface.py
│   └── darcy_porous.py
├── utils/
│   ├── matlab_exporter.py
│   ├── vtk_exporter.py
│   └── plot_engine.py
├── scripts/
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib, PyVista

### Quick Start

```bash
git clone https://github.com/Farzadmdz/ThermalPINN.git
cd ThermalPINN
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running a Baseline Simulation

```bash
python -m models.pinn_core \
    --config configs/vapor_chamber_baseline.yml \
    --epochs 10000 \
    --export_matlab True
```

---

## Validation & Benchmarking Results

The figure below presents the high-fidelity validation of **ThermalPINN** against traditional Finite Volume Method (FVM) reference data for a **bi-porous vapor chamber** operating under conjugate heat transfer (CHT) conditions. The neural network prediction exhibits excellent agreement with the CFD reference, with a maximum absolute temperature error of only **0.92°C** localized at the challenging interface coupling and boundary constraint regions.

![ThermalPINN vs. Traditional CFD - Bi-Porous Vapor Chamber Heat Spreading and CHT Field Validation](docs/images/ThermalPINN_Validation.png)

**Figure 1.** Comprehensive CHT field validation: (a) FVM Temperature Field (Reference), (b) ThermalPINN Temperature Field (Prediction), (c) PINN-Predicted Flow Velocity Field illustrating capillary liquid return and vapor core flow, and (d) Prediction Error Distribution (PINN – FVM) highlighting error localization at the interface coupling and boundary constraints. The inset in (a) shows the magnified bi-porous wick microstructure, while the inset in (d) displays the error distribution along the $x = 25$ mm cut-line and the training convergence history (MSE Loss vs. Epoch).

---

## Validation & Benchmarking

ThermalPINN includes canonical validation cases against CFD reference simulations, analytical benchmark solutions, thermal conduction benchmarks, and conjugate heat-transfer reference problems. Validation metrics encompass relative $L^2$ error, pressure-drop deviation, temperature-field reconstruction accuracy, residual convergence history, and Nusselt-number comparison.

---

## Engineering Applications

### Aerospace Thermal Management

ThermalPINN enables rapid optimization of vapor chambers and thermal spreaders for avionics and satellite systems where weight and thermal performance must be simultaneously optimized.

### Power Electronics Cooling

The framework models high-power-density cooling systems for IGBT, MOSFET, and AI accelerator hardware, providing real-time thermal insight during the design cycle.

### Scientific Machine Learning

ThermalPINN serves as a development platform for differentiable CFD surrogates, reduced-order modeling, and real-time digital twins in scientific computing environments.

---

## Visualization Pipeline

ThermalPINN automatically generates 600 DPI journal-ready figures, thermal contour maps, velocity vector fields, residual convergence curves, streamline visualizations, and ParaView-compatible datasets. The visualization stack is specifically engineered for high-impact journal publications and international conference presentations.

---

## Future Work

Planned research extensions include turbulence modeling via hybrid RANS/PINN methods, Fourier Neural Operators (FNOs), LES-informed surrogate models, multiphase boiling and evaporation physics, and GPU-distributed scientific training pipelines.

---

## Contribution Guidelines

Contributions from researchers, CFD engineers, and scientific machine-learning developers are warmly welcomed. Please ensure scientifically validated implementations, clear numerical documentation, reproducible benchmark results, high-quality visualization outputs, and consistent coding standards.

---

## License

This project is distributed under the MIT License.

---

## Author & Citation

**Developer:** Farzad MehdiZadeh (Farzad Mahdizadeh Farsangi)  
**Affiliation:** Sharif University of Technology, Tehran, Iran  
**Background:** B.Sc. and M.Sc. in Mechanical Engineering (Thermal-Fluids Sciences), Sharif University of Technology. Currently pursuing advanced master's research projects in computational heat transfer, physics-informed machine learning, and aerospace thermal management systems.

If you utilize ThermalPINN in your research, please cite:

```bibtex
@misc{thermalpinn2026,
  author       = {Mahdizadeh Farsangi, Farzad},
  title        = {ThermalPINN: A Physics-Informed Neural CFD Framework for Conjugate Heat Transfer and Advanced Vapor Chambers},
  year         = {2026},
  howpublished = {\url{https://github.com/Farzadmdz/ThermalPINN}},
  note         = {Version 1.0.0}
}
```

---

## Research Statement

ThermalPINN is maintained as an open scientific-computing initiative focused on accelerating next-generation thermal-management research through the integration of computational physics, machine learning, and high-performance engineering simulation workflows.
