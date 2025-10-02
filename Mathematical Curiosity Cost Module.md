Maple Brain Healthcare Inc.
Renee M Gagnon Author
2025

# Mathematical Curiosity Cost Module

A sophisticated intrinsic motivation system designed specifically for mathematical reasoning tasks in deep reinforcement learning. This module encourages **meaningful mathematical exploration** rather than superficial novelty-seeking.

## 🎯 Key Features

- **📐 Structure-Aware Exploration**: Understands mathematical concepts (problem types, difficulty, reasoning modes) rather than just pixel-level differences
- **🧠 Learning Progress Tracking**: Rewards genuine understanding improvement, not just surprise
- **🎮 Flow State Optimization**: Implements Zone of Proximal Development (ZPD) to keep agents in optimal challenge zones
- **🚫 Anti-Pathology Mechanisms**: Prevents boredom loops and stuckness behaviors
- **🔬 Ensemble Uncertainty**: Uses multiple prediction heads for principled epistemic uncertainty estimation

## 🔬 Theoretical Foundation

The module computes step-wise intrinsic cost:

```
C_int(t) = β₁·Boredom + β₂·Stuckness - α₁·InfoGain - α₂·LearnProgress - α₃·Novelty - α₄·ZPD
```

### Components Explained

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Information Gain (IG)** | Reward epistemic uncertainty reduction | Ensemble prediction variance |
| **Learning Progress (LP)** | Reward model improvement, not just surprise | EMA loss reduction per bucket |
| **Novelty (N)** | Encourage diverse mathematical exploration | Structure-aware pseudo-counts |
| **ZPD Shaping** | Stay in optimal challenge zone (~60% success) | Gaussian preference around target |
| **Boredom** | Prevent policy entropy collapse | Policy entropy or action repetition |
| **Stuckness** | Prevent progress plateaus | Progress delta history |

## 📦 Installation

### Prerequisites

```bash
pip install torch numpy collections
```

### Basic Installation

Simply copy the `mathematical_curiosity_cost.py` file into your project and import:

```python
from mathematical_curiosity_cost import MathematicalCuriosityCost, create_curiosity_module
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from mathematical_curiosity_cost import create_curiosity_module

# Create curiosity module with defaults
curiosity = create_curiosity_module()

# Example observation and action (adapt to your environment)
obs = {
    "problem_type": "algebra",
    "difficulty": "medium", 
    "question_length": 120,
    "progress": 0.3,
    "features": [1.0, 0.8, 0.6, 0.0, 0.2, 0.9, 0.1, 0.4]
}

action = {
    "strategic": 2,
    "tactical": 7
}

next_obs = {
    **obs,
    "progress": 0.4  # Progress increased
}

# Compute curiosity cost and reward
cost, reward, components = curiosity(obs, action, next_obs, solved=False)

print(f"Curiosity Cost: {cost:.3f}")
print(f"Curiosity Reward: {reward:.3f}")
print(f"Components: {components}")
```

### Integration with Training Loop

```python
# In your training loop
def train_episode():
    obs = env.reset()
    episode_reward = 0.0
    
    while not done:
        # Get action from your policy
        action = agent.select_action(obs)
        
        # Environment step
        next_obs, env_reward, done, info = env.step(action)
        
        # Compute curiosity
        curiosity_cost, curiosity_reward, components = curiosity(
            obs, action, next_obs, 
            solved=info.get('solved', False),
            policy_logits=agent.last_policy_logits  # Optional
        )
        
        # Combine rewards (tune lambda in [0.1, 0.5])
        lambda_curiosity = 0.3
        total_reward = env_reward + lambda_curiosity * curiosity_reward
        
        # Store experience with combined reward
        agent.store_experience(obs, action, total_reward, next_obs, done)
        
        # Train agent
        if len(agent.replay_buffer) > batch_size:
            agent.train_step()
        
        obs = next_obs
        episode_reward += total_reward
    
    # Reset curiosity state for next episode
    curiosity.reset_episode()
    
    return episode_reward, components
```

## ⚙️ Configuration

### Default Configuration

```python
default_config = {
    'feature_dim': 32,                    # Feature vector size for ensemble
    'action_dim_strategic': 6,            # Number of strategic actions
    'action_dim_tactical': 12,            # Number of tactical actions
    'zpd_tau': 0.6,                      # Target success rate (flow state)
    'zpd_sigma': 0.2,                    # ZPD Gaussian width
    'alpha': (0.35, 0.25, 0.20, 0.20),  # Curiosity weights (IG, LP, N, ZPD)
    'beta': (0.40, 0.30),               # Penalty weights (Boredom, Stuckness)
    'k_ensemble': 4,                     # Number of ensemble heads
    'stuck_k': 4,                        # Steps to check for stuckness
    'progress_eps': 1e-3,                # Minimum progress threshold
    'ema_beta': 0.9,                     # EMA decay rate
    'learning_rate': 1e-3,               # Ensemble learning rate
    'device': 'cuda' if available else 'cpu'
}
```

### Custom Configuration

```python
# For more exploration-heavy tasks
exploration_config = {
    'alpha': (0.45, 0.25, 0.25, 0.15),  # Higher IG and Novelty
    'beta': (0.30, 0.25),               # Lower penalties
    'zpd_tau': 0.5,                     # Slightly easier problems
}

curiosity = create_curiosity_module(exploration_config)
```

```python
# For more focused, less exploratory learning
focused_config = {
    'alpha': (0.25, 0.35, 0.15, 0.25),  # Higher Learning Progress
    'beta': (0.50, 0.35),               # Higher penalties for wandering
    'zpd_tau': 0.7,                     # Harder problems
}

curiosity = create_curiosity_module(focused_config)
```

## 🔧 Advanced Usage

### Custom Feature Extraction

Override feature extraction for your specific environment:

```python
class CustomCuriosity(MathematicalCuriosityCost):
    def extract_features(self, obs, action):
        # Custom feature extraction logic
        problem_embedding = self.embed_problem(obs['question_text'])
        action_embedding = self.embed_action(action)
        features = torch.cat([problem_embedding, action_embedding], dim=1)
        return features
```

### Monitoring and Debugging

```python
# Log curiosity components for analysis
components_history = []

def log_curiosity_components(components):
    components_history.append(components)
    
    # Log every 100 steps
    if len(components_history) % 100 == 0:
        recent = components_history[-100:]
        avg_components = {
            key: np.mean([c[key] for c in recent])
            for key in recent[0].keys()
        }
        print(f"Average curiosity components: {avg_components}")

# In training loop
cost, reward, components = curiosity(obs, action, next_obs, solved)
log_curiosity_components(components)
```

### Adaptive Configuration

```python
class AdaptiveCuriosity:
    def __init__(self, curiosity_module):
        self.curiosity = curiosity_module
        self.performance_history = []
    
    def adapt_weights(self, recent_performance):
        """Adapt curiosity weights based on learning progress"""
        if recent_performance < 0.3:  # Struggling
            # Increase exploration
            self.curiosity.alpha = (0.4, 0.2, 0.3, 0.1)
        elif recent_performance > 0.7:  # Doing well
            # Focus more on learning progress
            self.curiosity.alpha = (0.2, 0.4, 0.1, 0.3)
```

## 🎛️ Hyperparameter Tuning Guide

### α Weights (Curiosity Terms)

- **Information Gain (α₁)**: 
  - Increase (0.4-0.5) for more exploration of uncertain states
  - Decrease (0.2-0.3) if agent is too distractible
  
- **Learning Progress (α₂)**:
  - Increase (0.3-0.4) for more focus on understanding improvement
  - Decrease (0.1-0.2) in very noisy environments
  
- **Novelty (α₃)**:
  - Increase (0.3-0.4) for broader exploration across problem types
  - Decrease (0.1-0.2) if agent avoids difficult problems
  
- **ZPD Shaping (α₄)**:
  - Increase (0.3-0.4) to strongly enforce optimal challenge level
  - Decrease (0.1-0.2) for more flexible difficulty selection

### β Weights (Penalty Terms)

- **Boredom (β₁)**:
  - Increase (0.5-0.6) if agent gets stuck in repetitive behaviors
  - Decrease (0.2-0.3) if agent seems too restless
  
- **Stuckness (β₂)**:
  - Increase (0.4-0.5) if agent plateaus frequently
  - Decrease (0.1-0.2) for more persistent problem-solving

### ZPD Parameters

- **τ (Target Success Rate)**:
  - 0.5-0.6: Easier problems, more confidence building
  - 0.6-0.7: Standard flow state range
  - 0.7-0.8: Harder problems, less exploration
  
- **σ (ZPD Width)**:
  - 0.1-0.15: Narrow preference, strong ZPD enforcement
  - 0.2-0.3: Standard range
  - 0.3-0.4: Wide preference, flexible difficulty

## 🐛 Troubleshooting

### Common Issues

**1. Agent Not Exploring Enough**
```python
# Increase exploration weights
config['alpha'] = (0.45, 0.25, 0.25, 0.15)  # Higher IG, Novelty
config['beta'] = (0.30, 0.25)               # Lower penalties
```

**2. Agent Too Distractible**
```python
# Focus on learning progress
config['alpha'] = (0.25, 0.35, 0.15, 0.25)  # Higher LP
config['beta'] = (0.45, 0.35)               # Higher penalties
```

**3. Curiosity Rewards Too Small/Large**
```python
# Adjust the combination weight in training
lambda_curiosity = 0.5  # Increase for more curiosity influence
total_reward = env_reward + lambda_curiosity * curiosity_reward
```

**4. Ensemble Not Learning**
```python
# Check ensemble training
stats = curiosity.get_statistics()
print(f"Variance max: {stats['variance_max']}")  # Should increase over time

# Increase learning rate if stuck
config['learning_rate'] = 5e-3
```

### Performance Optimization

**For Large Action Spaces:**
```python
# Reduce ensemble size and feature dimension
config['k_ensemble'] = 2
config['feature_dim'] = 16
```

**For Fast Training:**
```python
# Simplify components
config['stuck_k'] = 2  # Shorter history
config['ema_beta'] = 0.95  # Faster adaptation
```

## 📊 Evaluation Metrics

Track these metrics to evaluate curiosity effectiveness:

```python
def evaluate_curiosity_effectiveness(components_history, performance_history):
    metrics = {}
    
    # Exploration diversity
    buckets_explored = len(set([c.get('bucket', '') for c in components_history]))
    metrics['exploration_diversity'] = buckets_explored
    
    # Learning efficiency  
    novelty_vs_performance = np.corrcoef(
        [c['novelty'] for c in components_history],
        performance_history
    )[0, 1]
    metrics['novelty_performance_correlation'] = novelty_vs_performance
    
    # ZPD adherence
    zpd_values = [c['zpd_shaping'] for c in components_history]
    metrics['avg_zpd_adherence'] = np.mean(zpd_values)
    
    # Curiosity-performance alignment
    curiosity_rewards = [c['curiosity_reward'] for c in components_history]
    curiosity_performance_corr = np.corrcoef(curiosity_rewards, performance_history)[0, 1]
    metrics['curiosity_performance_alignment'] = curiosity_performance_corr
    
    return metrics
```

## 🧪 Experimental Validation

To validate the module's effectiveness, run these experiments:

### 1. Ablation Study
```python
configs = {
    'no_curiosity': {'alpha': (0, 0, 0, 0), 'beta': (0, 0)},
    'only_novelty': {'alpha': (0, 0, 1, 0), 'beta': (0, 0)},
    'only_zpd': {'alpha': (0, 0, 0, 1), 'beta': (0, 0)},
    'full_system': {'alpha': (0.35, 0.25, 0.20, 0.20), 'beta': (0.40, 0.30)}
}

for name, config in configs.items():
    agent = train_with_curiosity(config)
    results[name] = evaluate_agent(agent)
```

### 2. Transfer Learning Test
```python
# Train on algebra problems
algebra_curiosity = create_curiosity_module({'zpd_tau': 0.6})
algebra_agent = train_agent(algebra_problems, algebra_curiosity)

# Test on geometry problems
geometry_performance = evaluate_agent(algebra_agent, geometry_problems)
print(f"Transfer performance: {geometry_performance}")
```

## 🔗 Integration Examples

### With OpenAI Gym Environment

```python
import gym
from mathematical_curiosity_cost import create_curiosity_module

class MathGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curiosity = create_curiosity_module()
        
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        
        # Convert gym action to curiosity format
        curiosity_action = {
            'strategic': action // 10,
            'tactical': action % 10
        }
        
        # Compute curiosity
        _, curiosity_reward, _ = self.curiosity(
            self.last_obs, curiosity_action, next_obs,
            solved=info.get('solved', False)
        )
        
        # Combine rewards
        total_reward = reward + 0.3 * curiosity_reward
        
        self.last_obs = next_obs
        return next_obs, total_reward, done, info
```

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class CuriosityCallback(BaseCallback):
    def __init__(self, curiosity_module, curiosity_weight=0.3):
        super().__init__()
        self.curiosity = curiosity_module
        self.curiosity_weight = curiosity_weight
        
    def _on_step(self) -> bool:
        # Get last transition
        obs = self.locals['obs_tensor']
        actions = self.locals['actions']
        rewards = self.locals['rewards'] 
        next_obs = self.locals['new_obs']
        
        # Compute curiosity (batch processing)
        for i in range(len(obs)):
            _, curiosity_reward, _ = self.curiosity(
                obs[i], actions[i], next_obs[i], solved=False
            )
            rewards[i] += self.curiosity_weight * curiosity_reward
        
        return True

# Usage
curiosity = create_curiosity_module()
callback = CuriosityCallback(curiosity)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, callback=callback)
```

## 📚 Research Background

This module implements ideas from several research areas:

- **Intrinsic Motivation**: [Oudeyer & Kaplan, 2007](https://www.pyoudeyer.com/ims.pdf)
- **Curiosity-Driven Learning**: [Pathak et al., 2017](https://arxiv.org/abs/1705.05363)
- **Flow Theory**: [Csikszentmihalyi, 1990](https://www.amazon.com/Flow-Psychology-Experience-Perennial-Classics/dp/0061339202)
- **Zone of Proximal Development**: [Vygotsky, 1978](https://en.wikipedia.org/wiki/Zone_of_proximal_development)
- **Information Gain**: [Lindley, 1956](https://en.wikipedia.org/wiki/Information_gain_(decision_tree))

## 📄 License

MIT License - feel free to use in your research and projects!

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the example implementations

---

**Happy Mathematical Learning! 🧮✨**