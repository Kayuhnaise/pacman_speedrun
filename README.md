# Pac-Man Speedrun AI (RL + Search Comparison)

A custom Pac-Man environment built to compare **rule-based search policies** and **reinforcement learning (DQN)** for efficient pellet collection and survival.

---

## Features

* Custom Pac-Man environment (grid-based, Atari-style board)
* Multiple AI policies:

  * Random Policy
  * Greedy Policy
  * BFS Policy
  * Smart BFS Policy
  * Deep Q-Network (DQN)
* Deep Reinforcement Learning:

  * CNN-based state representation
  * Double DQN implementation
  * Experience replay buffer
* Evaluation + Visualization:

  * Reward, loss, and win-rate plots
  * Policy comparison bar charts
  * Pellet reduction trends
* Optional video recording of gameplay

---

## Methods

### Rule-Based Policies

* **Greedy**: moves toward nearest pellet
* **BFS**: finds shortest path to pellets
* **Smart BFS**: balances path planning with safety

### Reinforcement Learning

* **DQN (Deep Q-Network)**:

  * Input: grid-based state encoding
  * Model: Convolutional Neural Network (CNN)
  * Output: Q-values for actions (up, down, left, right, stay)

* **Double DQN**:

  * Reduces overestimation bias
  * Uses separate networks for action selection and evaluation

---

## Reward Function

| Event             | Reward |
| ----------------- | ------ |
| Step              | -0.5   |
| Pellet            | +20    |
| Power Pellet      | +50    |
| Eat Ghost         | +200   |
| Death             | -200   |
| Win (clear board) | +1000  |

---

## Project Structure

```
pacman_speedrun/
│
├── env.py                # Pac-Man environment
├── boards.py             # Board layouts
├── main.py               # Run rule-based policies
├── train_dqn.py          # Train DQN agent
├── eval_dqn.py           # Evaluate trained DQN
├── plot_results.py       # Generate plots
├── dqn_model.py          # CNN model
├── replay_buffer.py      # Experience replay
├── state_encoder.py      # State representation
│
├── policies/
│   ├── bfs_policy.py
│   ├── greedy_policy.py
│   ├── random_policy.py
│   └── smart_bfs_policy.py
│
├── outputs/
│   ├── logs/
│   ├── models/
│   └── plots/
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Kayuhnaise/pacman_speedrun.git
cd pacman_speedrun

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## Running Policies

### Run rule-based policies:

```powershell
$env:POLICY="bfs_policy"; $env:BOARD="atari"; python main.py
```

Available policies:

* random_policy
* greedy_policy
* bfs_policy
* smart_bfs_policy
* mdp_policy

### Multiprocessing:
Policies can be run with multiprocessing to speed up their execution with parallel_runner.py

```bash
RENDER=0 SAVE_VIDEOS=0 EPISODES=2000 BOARD=pac POLICY=mdp_policy python parallel_runner.py
```

---

## Train DQN

```powershell
$env:BOARD="atari"; $env:TRAIN_EPISODES="100"; $env:MAX_STEPS="200"; python train_dqn.py
```

---

## Evaluate DQN

```powershell
$env:BOARD="atari"; $env:EPISODES="10"; $env:RENDER="0"; python eval_dqn.py
```

---

## Generate Plots

```powershell
python plot_results.py
```

Outputs:

* Reward curve
* Loss curve
* Win rate
* Pellet reduction
* Policy comparison charts

---

## Results Summary

* Search-based methods (BFS, Smart BFS) achieve near-optimal performance
* DQN improves over time but requires more training
* Reward shaping significantly improves learning speed
* Double DQN stabilizes training

