# Reinforcement Learning for Autonomous Highway Driving

**Course:** Reinforcement Learning -- BDML 2025-2026

**School:** EFREI Paris

**Authors:** Akram BOUDOUAOUR & Nassim ISSAD

**Repository:** [github.com/Accrame/Highway-env-project](https://github.com/Accrame/Highway-env-project)

---

## 1. Introduction

This report presents our approach to training autonomous driving agents in the Highway-env simulator using reinforcement learning. The objective is to teach an agent to navigate a three-lane highway among 40 aggressive vehicles without crashing.

We implemented and compared three algorithms:

1. **DQN** (Deep Q-Network) using Stable-Baselines3
2. **PPO** (Proximal Policy Optimization) using Stable-Baselines3
3. **Hand-coded DQN** implemented from scratch with PyTorch

All agents are evaluated on the same challenging configuration: 3 lanes, 40 aggressive vehicles, tight initial spacing (0.1), and a 40-second episode duration.

## 2. Environment

### 2.1 Highway-env

Highway-env (Leurent, 2018) is a collection of driving environments for reinforcement learning built on top of Gymnasium. The `highway-v0` environment simulates a multi-lane highway where the ego vehicle must navigate among other vehicles.

### 2.2 Action Space

The action space is **discrete** with 5 meta-actions:

| Action | Description             |
|--------|-------------------------|
| 0      | Change lane to the left |
| 1      | Idle (maintain course)  |
| 2      | Change lane to the right|
| 3      | Accelerate              |
| 4      | Decelerate              |

### 2.3 Observation Space

The observation is a **kinematics matrix** of shape (5, 5), where each row represents a nearby vehicle (the ego vehicle + 4 closest neighbours). The columns encode:

- **presence**: whether the vehicle exists (0 or 1)
- **x**: longitudinal position (normalised)
- **y**: lateral position (normalised)
- **vx**: longitudinal velocity (normalised)
- **vy**: lateral velocity (normalised)

### 2.4 Reward Function

The default reward at each timestep is:

$$R = \text{collision\_reward} \times \mathbb{1}_{\text{crashed}} + \text{high\_speed\_reward} \times \frac{v - v_{\min}}{v_{\max} - v_{\min}}$$

By default, the agent receives approximately +0.4 per timestep for driving at high speed, and -1 upon collision (which also terminates the episode). This incentivises both speed and survival.

### 2.5 Evaluation Configuration

The professor's evaluation configuration is deliberately challenging:

- **40 vehicles** with aggressive driving behaviour (`AggressiveVehicle`)
- **Tight initial spacing** (0.1), creating a dense traffic scenario
- **Duration**: 40 seconds

## 3. Algorithms

### 3.1 DQN (Stable-Baselines3)

Deep Q-Network (Mnih et al., 2015) is a value-based algorithm that learns the optimal action-value function $Q^*(s, a)$ using a neural network. Key components:

- **Experience Replay**: transitions $(s, a, r, s', d)$ are stored in a buffer and sampled in random mini-batches to break temporal correlations.
- **Target Network**: a periodically-synced copy of the Q-network provides stable TD targets: $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$.
- **Epsilon-greedy exploration**: with probability $\epsilon$, the agent selects a random action; otherwise it selects $\arg\max_a Q(s, a)$.

**Hyperparameters:**

| Parameter              | Value   |
|------------------------|---------|
| Learning rate          | 5e-4    |
| Buffer size            | 50,000  |
| Batch size             | 64      |
| Discount factor (gamma)| 0.99    |
| Target update interval | 500     |
| Exploration fraction   | 0.3     |
| Final epsilon          | 0.05    |
| Network architecture   | [256, 256] |
| Total timesteps        | 100,000 |

### 3.2 PPO (Stable-Baselines3)

Proximal Policy Optimization (Schulman et al., 2017) is a policy-gradient algorithm that directly optimises the policy $\pi_\theta(a|s)$. Key features:

- **Clipped surrogate objective**: the policy update is constrained by a clipping ratio to prevent destructively large updates:
$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)\right]$$
- **Generalised Advantage Estimation (GAE)**: reduces variance in advantage estimates using an exponentially-weighted sum of TD residuals.
- **On-policy**: collects fresh rollouts each iteration (no replay buffer).

**Hyperparameters:**

| Parameter              | Value   |
|------------------------|---------|
| Learning rate          | 3e-4    |
| Rollout steps (n_steps)| 256     |
| Batch size             | 64      |
| Number of epochs       | 10      |
| Discount factor (gamma)| 0.99    |
| GAE lambda             | 0.95    |
| Clip range             | 0.2     |
| Entropy coefficient    | 0.01    |
| Network architecture   | [256, 256] (shared) |
| Total timesteps        | 100,000 |

### 3.3 Hand-coded DQN (from scratch)

To demonstrate a deeper understanding of DQN, we implemented the algorithm entirely from scratch using PyTorch. Our implementation includes:

1. **QNetwork**: a feedforward neural network with two hidden layers of 256 units and ReLU activations, mapping the flattened observation (25 features) to Q-values for each of the 5 actions.

2. **ReplayBuffer**: a circular buffer (capacity 50,000) storing transitions and supporting uniform random sampling.

3. **Training loop**: at each environment step, the agent:
   - Selects an action via epsilon-greedy policy
   - Stores the transition in the replay buffer
   - Samples a mini-batch and computes the TD loss:
   $$L = \frac{1}{N}\sum_{i}\left(Q(s_i, a_i) - \left[r_i + \gamma \max_{a'} Q_{\text{target}}(s_i', a')\right]\right)^2$$
   - Backpropagates and updates the Q-network weights
   - Periodically syncs the target network (every 500 steps)

4. **Epsilon schedule**: linear decay from 1.0 to 0.05 over 40,000 steps.

**Hyperparameters:**

| Parameter              | Value   |
|------------------------|---------|
| Learning rate          | 5e-4    |
| Discount factor (gamma)| 0.99    |
| Epsilon decay steps    | 40,000  |
| Target update frequency| 500     |
| Batch size             | 64      |
| Training episodes      | 800     |

## 4. Results and Comparison

### 4.1 Training

All three models were trained on a slightly simplified configuration (30 vehicles, default behaviour) to accelerate training, then evaluated on the full challenging evaluation configuration.

The hand-coded DQN training curves show clear learning: the smoothed episode reward rises from ~15 to ~50 over 800 episodes, while the epsilon schedule drives the exploration rate from 1.0 to 0.05.

### 4.2 Evaluation Results

All models were evaluated over 30 episodes on the challenging evaluation configuration:

| Algorithm       | Mean Reward | Mean Episode Length |
|-----------------|-------------|---------------------|
| DQN (SB3)       | 6.57        | 7.73                |
| PPO (SB3)       | 6.74        | 8.70                |
| **Hand-coded DQN** | **27.58**   | **38.93**           |

### 4.3 Analysis

**Hand-coded DQN significantly outperforms both SB3 models.** Several factors explain this gap:

1. **Training duration**: The hand-coded DQN trained for 800 episodes (~40,000+ environment steps with full episode rollouts), while SB3 models were limited to 100,000 timesteps shared across 4 parallel environments. The hand-coded model accumulated more diverse experience.

2. **Single-environment training**: The hand-coded DQN trained on a single environment sequentially, which may have provided more coherent learning signals compared to the vectorised environments used by SB3. In SB3, the replay buffer and policy updates mix experience from 4 concurrent environments, which can introduce noise for this specific task.

3. **Gamma sensitivity**: We used $\gamma = 0.99$ for all models. In hindsight, the high discount factor may have been more suitable for the hand-coded implementation where episodes ran longer during training, while the SB3 models with shorter effective episodes may have benefited from a lower gamma (e.g., 0.8).

4. **Observation handling**: The hand-coded DQN flattens the (5,5) observation matrix into a 25-dimensional vector, which the fully-connected network processes as a single feature vector. This simple approach proved effective, as the network can learn to weight the relative positions and velocities of nearby vehicles.

**DQN vs PPO (SB3):** Both library-based models performed similarly poorly (6.57 vs 6.74 mean reward). PPO's slight edge may come from its on-policy nature allowing faster adaptation, though neither had enough training to properly handle the aggressive evaluation environment.

## 5. Conclusion

Our experiments demonstrate that a carefully implemented DQN from scratch can outperform library-based implementations when given sufficient training time. The hand-coded DQN achieved a mean reward of 27.58 on the challenging evaluation configuration (40 aggressive vehicles), surviving an average of 39 steps compared to fewer than 9 for the SB3 models.

Key takeaways:

- **Hyperparameter tuning matters more than algorithm choice** for this specific environment. The training configuration and duration had a larger impact on performance than the choice between DQN and PPO.
- **Reward shaping** (modifying collision and speed reward weights) is a promising avenue for further improvement, encouraging defensive driving strategies.
- **Implementation from scratch** provides deeper understanding and more control over the training process, which can lead to better results through careful tuning.

Future work could explore Double DQN, Dueling DQN architectures, or curriculum learning strategies where the agent progressively faces more aggressive traffic.

## References

- Leurent, E. (2018). *An Environment for Autonomous Driving Decision-Making*. GitHub: highway-env.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540).
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
