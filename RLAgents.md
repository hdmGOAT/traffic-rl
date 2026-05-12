# RL Agent Implementation Guide

This document provides a technical overview of how the reinforcement learning agents are implemented, how they interact with the traffic environment, and how they learn.

## 1. Agent Architecture & Inheritance

To keep the system modular and interchangeable, every agent (DQN, Double DQN, Dueling DQN, Tabular Q) must inherit from the `RLAgent` Abstract Base Class (located in `src/traffic_rl/agents/base.py`).

By inheriting from `RLAgent`, every model is forced to implement two core methods:

*   **`act(self, state_vector, train=True)`**: Given the current state of the intersection (e.g., how many cars are waiting in each lane), the agent uses its neural network or internal table to predict the best traffic light phase.
*   **`observe(self, state, action, reward, next_state, done)`**: After taking an action, the agent receives feedback. This is the exact function where the memory storage and neural network weight updates (learning) are triggered.

Because they all share this interface, the main training loop (`src/traffic_rl/training.py`) doesn't need to know *which* specific agent is running.

## 2. The Reward Loop

The reward calculation happens inside the Environment, specifically in the `step()` function (e.g., inside `src/traffic_rl/envs/cityflow_env.py`):

1.  **The Agent Acts**: The agent predicts the optimal phase (`action = agent.act(state)`).
2.  **The Environment Advances**: The environment applies the chosen phase and advances the simulation by a set number of seconds (`decision_interval`).
3.  **The Reward is Calculated**: The environment observes the new state of the intersection and calls the reward function (`src/traffic_rl/reward.py`) to calculate the penalty. For example, if there are 20 cars waiting, the reward is `-20.0`.
4.  **Feedback Loop**: The environment returns this reward back to the training loop (`next_obs, reward, done, _ = env.step(action)`).

## 3. The Learning Process

The actual "learning" happens entirely inside the agent's `observe()` method:

1.  **Store the Memory**: The agent takes the specific sequence (`state`, `action`, `reward`, `next_state`) and saves it into its **Replay Buffer** (its memory bank).
2.  **Sample Past Memories**: Every few steps (dictated by `train_frequency`), the agent grabs a random batch of memories from the buffer.
3.  **Calculate the Error (Loss)**: It looks at a memory and asks its Neural Network, *"What did you THINK the total future reward was going to be for this action?"* It then compares that guess to the *actual* reward it received plus the predicted value of the next state.
4.  **Update Weights**: It calculates the mathematical difference between its guess and reality (the Mean Squared Error). Using backpropagation (Gradient Descent), it tweaks the weights of the neural network so that its next guess will be slightly closer to the truth.

---

## Agent Hyperparameters

The deep reinforcement learning agents (DQN, Double DQN, Dueling DQN) share a common set of hyperparameters defined in the `TrainingConfig`. 

| Parameter | Description |
| :--- | :--- |
| **`gamma`** | **Discount Factor.** Determines how much the agent cares about rewards in the distant future versus immediate rewards. `0` means it only cares about the next step; `0.99` means it plans far ahead. |
| **`learning_rate`** | The step size for the neural network's gradient descent. A higher value makes the agent learn faster but can cause instability. |
| **`epsilon_start`** | The starting exploration rate. At `1.0`, the agent ignores its network and takes 100% random actions to discover how the environment works. |
| **`epsilon_end`** | The minimum exploration rate. At `0.01`, the agent takes random actions 1% of the time to ensure it never completely stops exploring. |
| **`epsilon_decay`** | How fast epsilon drops from the start value to the end value. At `0.995`, it multiplies epsilon by 0.995 every step, slowly transitioning from pure exploration to pure exploitation. |
| **`hidden_dim`** | The number of neurons in the hidden layers of the neural network. Larger values allow the network to learn more complex patterns but take longer to train. |
| **`batch_size`** | The number of past memories (experiences) the agent samples from the Replay Buffer during each training update. `32` is a standard default. |
| **`replay_capacity`** | The maximum number of memories the Replay Buffer can hold. Once full, the oldest memories are deleted to make room for new ones. |
| **`learning_starts`** | The number of steps the agent must take (purely random exploration) to fill up the Replay Buffer *before* it is allowed to start updating its neural network. |
| **`target_update_interval`** | How often the agent syncs its stable "Target Network" with its actively learning "Online Network". Larger values mean more stable, but slower, learning. |
| **`train_frequency`** | How often the agent performs a gradient descent update. A value of `4` means the network updates its weights once every 4 steps. |

---

## 4. Glossary of Machine Learning Terms

To help explain the model to audiences unfamiliar with reinforcement learning, here are definitions for key concepts used in this project:

*   **Gradient Descent:** The mathematical process the neural network uses to learn. It calculates how "wrong" its predictions were (the gradient of the error) and adjusts its internal weights slightly in the opposite direction (descent) to make fewer mistakes next time.
*   **Backpropagation:** The algorithm that efficiently calculates the exact gradient for every single weight in the network. It propagates the error backwards from the output prediction all the way through the hidden layers.
*   **Mean Squared Error (MSE):** The mathematical formula used to calculate how "wrong" the agent was. It takes the difference between the predicted reward and actual reward, and squares it (which harshly penalizes larger errors).
*   **Replay Buffer:** A memory bank (usually an array or queue) where the agent stores its past experiences `(State, Action, Reward, Next State)`. Learning directly from sequential steps causes instability (because consecutive steps are highly correlated), so the agent learns by randomly sampling scattered memories from this buffer.
*   **Epsilon-Greedy:** A strategy for balancing exploration and exploitation. With probability *epsilon* ($\epsilon$), the agent takes a random action to explore the environment. Otherwise, it greedily exploits its neural network to pick the best known action. Epsilon "decays" over time as the agent becomes more confident.
*   **Target Network vs. Online Network:** Deep Reinforcement Learning often uses two networks. The **Online Network** is actively learning and predicting actions. The **Target Network** is a frozen, older copy of the Online Network used to evaluate future rewards. Syncing them periodically stops the agent from chasing a "moving target", drastically improving learning stability.
*   **Overestimation Bias:** A common failure in standard DQN where the agent becomes irrationally optimistic about the value of certain actions, inflating its expected rewards. Double DQN fixes this.
*   **Bellman Equation:** The foundational math equation behind Q-learning. It states that the value of a state is the immediate reward you get, plus the discounted expected value of the *next* state.
