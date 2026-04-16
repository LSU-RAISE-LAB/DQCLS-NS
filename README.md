# DQCLS: Dynamic Quantum-Assisted Co-Design of Control Tuning and Lyapunov Stability Synthesis for Nonlinear Systems

![License](https://img.shields.io/badge/license-Academic-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-compatible-purple)

> Official code, models, and simulation results accompanying the paper:  
> *"Dynamic Quantum-Assisted Co-Design of Control Tuning and Lyapunov Stability Synthesis for Nonlinear Systems"*

---

## 🔬 Overview

This repository contains implementations and simulation results for a **general two-step dynamic quantum-assisted co-design framework** for nonlinear closed-loop systems.  
The proposed methodology jointly redesigns **controller parameters** and **Lyapunov-certificate parameters** online, so that control performance and stability-oriented synthesis are handled within a unified dynamic optimization loop.

The framework combines:
- a **Black-Hole (BH)** stage for online search-space calibration
- an **encoded optimization** stage for finite binary representation of the co-design variables
- a **QITE-based quantum solution layer** for solving the encoded problem
- exact nonlinear closed-loop re-evaluation for final candidate selection

The framework is developed as a **general methodology for nonlinear control**, rather than for one specific application.  
Its applicability is demonstrated on:
- first-order nonlinear consensus
- second-order nonlinear consensus
- induction motor drive control

All simulation studies in the paper can be reproduced from the codes provided in this repository.

---

## 🌟 Features

- **Two-step dynamic co-design framework** for nonlinear systems  
- **Joint online redesign** of controller gains and Lyapunov-certificate parameters  
- **Black-Hole-based search-space calibration** before encoding  
- **Finite binary encoding and QUBO/Ising-like surrogate construction**  
- **QITE-based encoded quantum optimization**  
- **Support for alternative encoded solvers**, including brute force and classical local search  
- **Validation on multiple nonlinear closed-loop systems**  
- **Application-agnostic structure** that can be adapted to other nonlinear control problems  

---

## 📚 Citation

If you use the content of this project in your research, please cite:

> M. Hasanzadeh and A. Kargarian, "Dynamic Quantum-Assisted Co-Design of Control Tuning and Lyapunov Stability Synthesis for Nonlinear Systems," to appear.

---

## 👤 Author

**Milad Hasanzadeh**  
Department of Electrical and Computer Engineering  
Louisiana State University  
📧 e.mhasanzadeh1377@yahoo.com  

📅 **Release Date:** 2026  
📄 **License:** Academic and Research Use Only  

