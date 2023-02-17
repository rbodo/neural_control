# Neural Control

Collection of tools to study the controllability of neural dynamical systems 
using neural networks, optimal control theory, and reinforcement learning.

## Installation
For basic functionality:
```
pip install -r requirements.txt
```
To use nonlinear MUJOCO systems such as the inverted pendulum, follow the 
respective instructions here: https://www.gymlibrary.dev/environments/mujoco/. 
Note: Gym requires the old mujoco 1.5.0 version for its python bindings 
(download here: https://www.roboti.us/download.html).

## Contents

- `examples/`: End-to-end use cases of RNNs learning to control linear and nonlinear systems using optimal control or RL. A file called `linear_rnn_lqg.py` refers to a `linear` dynamical system environment (e.g. Double Integrator) in which an `rnn` neural system solves some task (e.g. regression) using `lqg` learning signals.  
- `scratch/`: Collection of scripts to explore various aspects of controllability of neural systems. 
- `src/`: Core logic, common objects and utility functions.

## Getting started
Run `linear_rnn_lqr.py`, and afterwards `visualize_linear_rnn_lqr.py`.

## Citation
If you find this software useful in your research, please cite the associated [paper](https://www.biorxiv.org/content/10.1101/2022.12.24.521852v1.abstract).
