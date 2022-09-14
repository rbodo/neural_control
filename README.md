# Neural Control

Collection of tools to study the controllability of neural dynamical systems 
using neural networks, optimal control theory, and reinforcement learning.

## Installation
For basic functionality:
```
pip install -r requirements.txt
```
To use nonlinear MUJOCO systems such as the inverted pendulum, follow the 
respective instructions here: https://www.gymlibrary.dev/environments/mujoco/

## Contents

- `examples/`: End-to-end use cases of RNNs learning to control linear and nonlinear systems using optimal control or RL. A file called `linear_rnn_lqg.py` refers to a `linear` dynamical system environment (e.g. Double Integrator) in which an `rnn` neural system solves some task (e.g. regression) using `lqg` learning signals.  
- `scratch/`: Collection of scripts to explore various aspects of controllability of neural systems. 
- `src/`: Core logic, common objects and utility functions.
