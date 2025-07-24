# Multi-Agent Reinforcement Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen)](https://www.python.org/dev/peps/pep-0008/)

A comprehensive framework for training and evaluating multi-agent reinforcement learning algorithms in cooperative, competitive, and mixed environments. This implementation provides state-of-the-art MARL algorithms with robust training pipelines, visualization tools, and analysis capabilities.

## Key Features

- 🧩 **Multi-Environment Support**: Cooperative, competitive, and mixed interaction settings
- 🤖 **Advanced Algorithms**: IQL, MADDPG, QMIX, and COMA implementations
- 🧠 **Non-Stationarity Handling**: Fingerprinted experience replay for stable learning
- 🔄 **Agent Specialization**: Automatic role differentiation tracking
- 🛡️ **Robustness Testing**: Simulated agent failures and recovery metrics
- 📡 **Communication Protocols**: Attention-based learned communication
- 📊 **Comprehensive Visualization**: Training curves, action distributions, behavior traces
- 🧪 **Reproducible Experiments**: Configuration management and checkpointing

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Ermi1223/multi-agent-rl-framework.git
cd multi-agent-rl-framework
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training Agents
```bash
# Cooperative environment with QMIX
python -m scripts.train --config configs/cooperative.yaml

# Competitive environment with MADDPG
python -m scripts.train --config configs/competitive.yaml

# Mixed environment with COMA
python -m scripts.train --config configs/mixed.yaml
```

### Evaluating Trained Models
```bash
# Evaluate cooperative agents after 5000 episodes
python -m scripts.evaluate --config configs/cooperative.yaml --checkpoint 3000
```

### Visualizing Results
```bash
# Generate behavior traces and action distributions
python -m scripts.visualize --config configs/cooperative.yaml --checkpoint 3000
```

### Live Demo
```bash
# Interactive demo of trained agents
python -m scripts.demo --config configs/cooperative.yaml --checkpoint 5000
```

## Configuration

The framework uses YAML configuration files for experiment management. Key configuration files:

- `configs/base.yaml`: Base configuration with common parameters
- `configs/cooperative.yaml`: Cooperative environment (simple_spread)
- `configs/competitive.yaml`: Competitive environment (simple_tag)
- `configs/mixed.yaml`: Mixed environment (simple_adversary)

### Key Configuration Parameters
```yaml
env_name: "cooperative"   # Environment type
n_agents: 3               # Number of agents
algorithm: "QMIX"         # MARL algorithm (IQL, MADDPG, QMIX, COMA)
total_episodes: 3000     # Training episodes
use_communication: true   # Enable inter-agent communication
failure_simulation: true  # Simulate agent failures
```

## Project Structure

```
multi-agent-rl-framework/
├── agents/               # Agent implementations
│   ├── base_agent.py     # Base agent class
│   ├── iql_agent.py      # Independent Q-Learning
│   ├── maddpg_agent.py   # MADDPG algorithm
│   ├── qmix_agent.py     # QMIX with value decomposition
│   └── coma_agent.py     # COMA with counterfactual baselines
├── configs/              # Experiment configurations
│   ├── base.yaml         # Base configuration
│   ├── cooperative.yaml  # Cooperative task config
│   ├── competitive.yaml  # Competitive task config
│   └── mixed.yaml        # Mixed task config
├── environments/         # Environment wrappers
│   ├── multi_agent_env.py # PettingZoo integration
│   └── failure_simulator.py # Agent failure simulation
├── models/               # Neural network architectures
│   ├── actor_critic.py   # Actor-Critic networks
│   ├── coma_critic.py    # COMA-specific critic
│   ├── mixer.py          # QMIX mixing network
│   └── q_networks.py     # Q-network implementations
├── scripts/              # Executable scripts
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   ├── visualize.py      # Visualization tools
│   └── demo.py           # Live demo
├── utils/                # Utility modules
│   ├── replay_buffer.py  # Fingerprinted experience replay
│   ├── logger.py         # Training logger and visualization
│   ├── communication.py  # Attention-based communication
│   ├── coordinator.py    # Coordination metrics
│   └── analysis_tools.py # Advanced analysis
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## Key Algorithms Implemented

| Algorithm | Type | Key Features | Best For |
|-----------|------|--------------|----------|
| **IQL** | Independent | Simple implementation, Baseline | Competitive tasks |
| **MADDPG** | Centralized Critic | CTDE, Continuous actions | Mixed environments |
| **QMIX** | Value Decomposition | Monotonic mixing, Credit assignment | Cooperative tasks |
| **COMA** | Counterfactual | Centralized critic, Individual baselines | Cooperative tasks with credit assignment |

## Analysis and Visualization

The framework provides comprehensive analysis tools:

1. **Learning Curves**: Track reward progress during training
   ```python
   from utils.logger import Logger
   logger = Logger(config)
   logger.generate_plots()  # Saves reward/specialization curves
   ```

2. **Behavior Analysis**:
   - Action distribution histograms
   - Agent trajectory visualization
   - Communication pattern analysis

3. **Specialization Metrics**:
   - Role consistency scores
   - Skill diversity indices
   - Task coordination efficiency

4. **Robustness Reports**:
   - Failure recovery rates
   - Performance degradation metrics
   - Stability indices

## Customization

### Adding New Environments
1. Extend `environments/multi_agent_env.py`
2. Implement environment-specific wrapper
3. Add configuration in `configs/`

### Implementing New Algorithms
1. Create new agent class in `agents/`
2. Inherit from `BaseAgent`
3. Implement `select_action()` and `update()` methods
4. Add to `agent_factory` in `agents/__init__.py`

## Troubleshooting

### Common Issues
- **Import errors**: Ensure PYTHONPATH is set correctly:
  ```bash
  PYTHONPATH=. python scripts/train.py
  ```
- **Environment dependencies**: Install PettingZoo MPE environments:
  ```bash
  pip install pettingzoo[mpe]
  ```
- **Visualization issues**: Install required dependencies:
  ```bash
  pip install pygame imageio
  ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new feature branch
3. Implement your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Cite This Work

If you use this framework in your research, please cite:

```bibtex
@misc{marl_framework,
  title = {Multi-Agent Reinforcement Learning Framework},
  author = {Ermiyas},
  year = {2025},
  url = {https://github.com/Ermi1223/multi-agent-rl-framework}
}
```