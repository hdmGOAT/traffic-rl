## AI System Specification: Reinforcement Learning–Based Traffic Signal Control

### System Overview

The proposed AI system implements a **reinforcement learning (RL)–based traffic signal controller** using the **CityFlow** traffic simulation environment.
The system learns an adaptive signal control policy through repeated interaction with a simulated urban traffic network, aiming to reduce congestion and improve traffic efficiency.

---

## Environment

* **Simulation Platform:** CityFlow
* **Environment Type:** Microscopic, discrete-time traffic simulation
* **Controlled Entities:** Signalized intersections
* **Simulation Horizon:** Fixed-duration episodes (e.g., 3600 seconds)

CityFlow provides a fully observable environment where traffic dynamics, vehicle movements, and signal phase transitions are simulated deterministically given a traffic demand configuration.

---

## Agent Definition

Each signalized intersection is controlled by a reinforcement learning agent.

* **Agent Type:** Model-free RL agent
* **Control Granularity:** Intersection-level
* **Decision Interval:** Fixed time step (e.g., every 5–10 seconds)

---

## State Space (Observations)

At each decision step ( t ), the agent observes traffic conditions from incoming lanes.

Typical state features include:

* Queue length per incoming lane
* Number of waiting vehicles
* Current signal phase (encoded)
* Optional elapsed green time

Example state vector:
[
s_t = [q_1, q_2, q_3, q_4, p_t]
]
where ( q_i ) represents lane queue lengths and ( p_t ) denotes the active signal phase.

---

## Action Space

The action space consists of **discrete traffic signal phase selections**.

* **Action Type:** Discrete
* **Actions:**
  [
  a_t \in {0, 1, \dots, P-1}
  ]
  where ( P ) is the number of valid signal phases at an intersection.

CityFlow enforces:

* Legal phase transitions
* Minimum green time constraints

This ensures all actions are physically valid.

---

## Reward Function

The reward function is designed to minimize traffic congestion.

Common formulation:
[
r_t = -\sum_{i} q_i
]

Where:

* ( q_i ) is the queue length of lane ( i )

Alternative reward variants may include:

* Negative total waiting time
* Vehicle delay penalties
* Throughput-based rewards

The reward is computed at every decision step.

---

## Learning Algorithm

The system uses a **model-free reinforcement learning algorithm**, such as:

* Deep Q-Network (DQN), or
* Proximal Policy Optimization (PPO)

The agent learns a policy:
[
\pi(a_t | s_t)
]
that maps observed traffic states to signal control actions.

---

## Training Data

Training data is generated **online** through simulation.

* **Data Type:** Experience tuples
  [
  (s_t, a_t, r_t, s_{t+1})
  ]
* **Episodes:** Multiple simulation runs with fixed traffic demand
* **Replay:** Optional experience replay (for value-based methods)

No external datasets are required at this stage.

---

## Evaluation Metrics

Performance is evaluated using:

* Average queue length
* Average vehicle waiting time
* Intersection throughput
* Reward convergence over episodes

Baselines may include:

* Fixed-time signal control
* Rule-based or max-pressure controllers

---

## Scope and Limitations

* The system is evaluated **only in simulation**
* Traffic demand is synthetic and pre-defined
* Results do not imply direct real-world deployability

This implementation focuses on **algorithmic feasibility and learning effectiveness**, serving as a foundation for future integration with real-world traffic data.

---

## Implementation Notes (Current Repo)

The current implementation includes command-line tooling to make RL behavior observable and reproducible:

- `traffic_rl.cli.train` prints per-episode terminal progress with reward, running average, and a small ASCII trend bar.
- `traffic_rl.cli.evaluate` can now emit CityFlow chart files from replay logs via:
  - `--chart-file`
  - `--chart-title`
- `traffic_rl.cli.visualize` builds an evidence report (`.html` + `.json`) that compares trained vs untrained policies with:
  - mean reward difference
  - confidence interval, p-value, and Cohen's d
  - queue and throughput summary metrics
  - episode reward traces

These additions are reporting/observability features and do not change the underlying RL objective (queue-length minimization).