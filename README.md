# HoloSwarm

**A PyBullet-based multi-agent simulation framework for coordinated holographic drone swarms.**

> *"With great power comes great responsibility... and really convincing illusions."*  
> — Inspired by Mysterio & EDITH drones from *Spider-Man: Far From Home*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.x-orange.svg)](https://pybullet.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Early%20Development-yellow)](https://github.com/philipmk42/holoswarm)

---

## Overview

**HoloSwarm** is an open-source simulation platform built on top of [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) for researching and prototyping **coordinated holographic drone swarms**.

The goal is to simulate small quadcopter drones that fly in formation while projecting dynamic, viewer-dependent illusions — exactly like the iconic holographic drone swarms used by Mysterio in *Spider-Man: Far From Home*.

This project focuses on **physics-accurate simulation** (no hardware required) to explore:
- Multi-agent formation control
- Dynamic visual projection / illusion generation
- Reinforcement learning for coordinated "deception" behaviors
- Scalability of large swarms
- Metrics for visual fidelity and illusion effectiveness

**Current Status**: Early development. Core drone models, RL policies, and basic examples have been added.

---

## Motivation & Inspiration

Real-world drone swarms are already used for entertainment (light shows), delivery, inspection, and increasingly in defense (decoy & deception operations). Adding **holographic projection** capabilities opens exciting new possibilities:

- **Entertainment & VFX**: Physics-accurate pre-visualization for films and games
- **Research**: Novel multi-agent RL tasks involving visual coordination and deception
- **Defense Training**: Simulating visual/illusory swarm tactics and countermeasures
- **Education**: Accessible platform to teach swarm robotics, MARL, and PyBullet

This work extends the excellent `gym-pybullet-drones` ecosystem with new visual and illusion-focused tasks while maintaining compatibility with existing RL pipelines (Stable-Baselines3, Ray RLlib, etc.).

---

## Repository Structure

```
holoswarm/
├── holoswarm/
│   ├── drones/
│   │   └── holodrone_mysterio.py      # Core holographic drone model
│   └── rl/
│       ├── holodrone_rl.py            # RL training for holographic drones
│       └── illusion_rl.py             # Illusion generation with RL
├── examples/
│   ├── mysterio_tricks.py
│   └── run_train.csv                  # Training scripts / logs
├── policies/
│   └── policy.npz                     # Trained RL policy
├── docs/
│   └── train_trick_readings.csv
├── .gitignore
└── README.md
```

---

## Key Files

| File                                      | Description                                      |
|-------------------------------------------|--------------------------------------------------|
| `holoswarm/drones/holodrone_mysterio.py`  | Main holographic drone implementation            |
| `holoswarm/rl/holodrone_rl.py`            | Reinforcement learning setup for drone control   |
| `holoswarm/rl/illusion_rl.py`             | RL for generating and coordinating illusions     |
| `policies/policy.npz`                     | Trained RL policy weights                        |
| `examples/mysterio_tricks.py`             | Example scripts demonstrating Mysterio-style tricks |
| `examples/run_train.csv`                  | Training run data / logs                         |

---

## Installation (Coming Soon)

```bash
git clone https://github.com/philipmk42/holoswarm.git
cd holoswarm

# Recommended: create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

> Full installation instructions and `requirements.txt` will be added shortly.

---

## Usage (Planned)

```python
# Example usage will be added here once environments are registered
import gymnasium as gym
# import holoswarm

# env = gym.make("HoloSwarm-v0", num_drones=20)
```

Detailed examples using the files in `examples/` and the trained policy in `policies/` will be added soon.

---

## Roadmap

- [x] Initial repository structure + first code push
- [x] Core holographic drone model (`holodrone_mysterio.py`)
- [x] RL training scripts (`holodrone_rl.py`, `illusion_rl.py`)
- [x] Trained policy (`policy.npz`)
- [ ] Custom Gymnasium environments with holographic visuals
- [ ] Dynamic projection / billboard system in PyBullet
- [ ] Viewer-dependent illusion metrics
- [ ] Full documentation and example notebooks
- [ ] Paper / preprint (planned)

---

## Contributing

Contributions are welcome! Whether it's improving the drone models, adding new RL environments, fixing bugs, or enhancing documentation.

Please open an issue or pull request.

---

## License

This project is licensed under the **MIT License**.

---

## Citation

If you use HoloSwarm in your research, please cite:

```bibtex
@software{holoswarm2026,
  author = {Philip and Contributors},
  title = {HoloSwarm: A PyBullet Multi-Agent Simulation Framework for Coordinated Holographic Drone Swarms},
  year = {2026},
  url = {https://github.com/philipmk42/holoswarm}
}
```

A formal paper is planned.

---

## Acknowledgments

- Built on [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) by Panerati et al. (IROS 2021)
- Inspired by the visual effects in *Spider-Man: Far From Home*
- Thanks to the PyBullet and multi-agent RL communities

---

**Let's build some convincing illusions together.** 🚁✨

*Repository is in active early development. Star it and follow along for updates!*
