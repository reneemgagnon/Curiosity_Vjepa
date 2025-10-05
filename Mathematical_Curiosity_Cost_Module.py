"""
Mathematical Curiosity Cost Module for Deep RL

A sophisticated intrinsic motivation system designed specifically for mathematical 
reasoning tasks. Rewards "useful surprise" and penalizes boredom or wheel-spinning,
with structure-aware exploration that understands mathematical concepts rather than
just pixel-level novelty.

Key Features:
- Structure-aware novelty using mathematical semantics
- Learning progress tracking (not just raw surprise)
- Zone of Proximal Development (ZPD) shaping for optimal challenge
- Anti-boredom and anti-stuckness mechanisms
- Ensemble-based uncertainty estimation

Author: Maple Brain Healthcare Inc.
Renee M Gagnon
License: MIT
Version: 1.0.0
"""

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional, List
import warnings


class MathematicalCuriosityCost(nn.Module):
    """
    Intrinsic Mathematical Curiosity Cost Module
    
    Computes step-wise intrinsic cost C_int(t) that encourages mathematically
    meaningful exploration:
    
    C_int(t) = β₁·B_t + β₂·S_t - α₁·IG_t - α₂·LP_t - α₃·N_t - α₄·ZPD_t
    
    Where:
    - B_t: Boredom (policy entropy collapse)
    - S_t: Stuckness (no progress for k steps)  
    - IG_t: Information Gain (epistemic uncertainty reduction)
    - LP_t: Learning Progress (improvement in prediction error)
    - N_t: Novelty (structure-aware pseudo-counts)
    - ZPD_t: Zone of Proximal Development shaping (optimal challenge)
    """
    
    def __init__(self,
                 feature_dim: int = 32,
                 action_dim_strategic: int = 6,
                 action_dim_tactical: int = 12,
                 zpd_tau: float = 0.6,
                 zpd_sigma: float = 0.2,
                 alpha: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
                 beta: Tuple[float, float] = (0.40, 0.30),
                 k_ensemble: int = 4,
                 stuck_k: int = 4,
                 progress_eps: float = 1e-3,
                 ema_beta: float = 0.9,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        """
        Initialize Mathematical Curiosity Cost module.
        
        Args:
            feature_dim: Dimension of feature vectors for ensemble heads
            action_dim_strategic: Number of strategic actions
            action_dim_tactical: Number of tactical actions  
            zpd_tau: Target success rate for ZPD (typically 0.6 for flow state)
            zpd_sigma: Width of ZPD Gaussian preference
            alpha: Weights for curiosity terms (IG, LP, N, ZPD)
            beta: Weights for penalty terms (Boredom, Stuckness)
            k_ensemble: Number of ensemble heads for uncertainty estimation
            stuck_k: Number of steps to look back for stuckness detection
            progress_eps: Minimum progress delta to avoid stuckness
            ema_beta: EMA decay rate for statistics tracking
            learning_rate: Learning rate for ensemble heads
            device: Device for computations ('cpu' or 'cuda')
        """
        super().__init__()
        
        # Store configuration
        self.feature_dim = feature_dim
        self.action_dim_strategic = action_dim_strategic
        self.action_dim_tactical = action_dim_tactical
        self.zpd_tau = zpd_tau
        self.zpd_sigma = zpd_sigma
        self.alpha = alpha  # (IG, LP, N, ZPD)
        self.beta = beta    # (Boredom, Stuckness)
        self.stuck_k = stuck_k
        self.progress_eps = progress_eps
        self.ema_beta = ema_beta
        self.device = device
        
        # Tracking data structures
        self.counts = collections.Counter()
        self.success_ema = {}  # bucket -> (ema_value, ema_count)
        self.loss_ema = {}     # bucket -> (ema_value, ema_count)
        self.progress_history = collections.deque(maxlen=stuck_k)
        self.var_max = 1e-6    # Running max for variance normalization

        assert action_dim_strategic > 0, "action_dim_strategic must be positive"
        assert action_dim_tactical > 0, "action_dim_tactical must be positive"
        
        # Tiny ensemble predictors for progress delta
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, 1)
            ) for _ in range(k_ensemble)
        ])
        
        # Optimizer for ensemble heads
        self.optimizer = optim.Adam(self.ensemble_heads.parameters(), lr=learning_rate)
        
        # Move to device
        self.to(device)
        
        print(f"✅ Mathematical Curiosity Cost initialized:")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Ensemble size: {k_ensemble}")
        print(f"  ZPD target: {zpd_tau:.2f} ± {zpd_sigma:.2f}")
        print(f"  Alpha weights: {alpha}")
        print(f"  Beta weights: {beta}")
    
    def create_structure_signature(self, obs: Dict[str, Any], action: Dict[str, Any]) -> Tuple:
        """
        Create structure-aware signature for mathematical novelty.
        
        This captures mathematical semantics rather than surface features:
        - Problem type (algebra, geometry, etc.)
        - Difficulty level 
        - Problem size (question length buckets)
        - Current reasoning mode (tactical action)
        - Mathematical structure (has equations, variables, etc.)
        
        Args:
            obs: Environment observation
            action: Agent action
            
        Returns:
            Tuple representing mathematical structure signature
        """
        # Extract mathematical semantics
        problem_type = obs.get("problem_type", "unknown")
        difficulty = obs.get("difficulty", "medium")
        
        # Bucket question length (coarse granularity)
        question_length = obs.get("question_length", 0)
        length_bucket = min(question_length // 40, 5)
        
        # Coarsen tactical action to reasoning mode
        tactical_action = action.get("tactical", 0)
        reasoning_mode = tactical_action // 2  # Group similar tactics
        
        # Mathematical structure indicators
        features = obs.get("features", [0] * 10)
        has_equations = int(len(features) > 0 and features[0] > 0.5)
        has_numbers = int(len(features) > 1 and features[1] > 0.5) 
        has_variables = int(len(features) > 2 and features[2] > 0.5)
        
        return (problem_type, difficulty, length_bucket, reasoning_mode, 
                has_equations, has_numbers, has_variables)
    
    def update_ema(self, ema_dict: Dict, key: Any, value: float) -> float:
        """Update exponential moving average for a given key."""
        if key in ema_dict:
            ema_val, ema_count = ema_dict[key]
            ema_val = self.ema_beta * ema_val + (1 - self.ema_beta) * value
            ema_count = self.ema_beta * ema_count + (1 - self.ema_beta) * 1.0
        else:
            ema_val, ema_count = value, 1.0
        
        ema_dict[key] = (ema_val, ema_count)
        return ema_val / max(ema_count, 1e-6)
    
    def extract_features(self, obs: Dict[str, Any], action: Dict[str, Any]) -> torch.Tensor:
        """
        Extract feature vector for ensemble heads.
        
        Combines observation features with action information to create
        a fixed-size feature vector for progress prediction.
        
        Args:
            obs: Environment observation
            action: Agent action
            
        Returns:
            Feature tensor of shape [1, feature_dim]
        """
        # Calculate target dimension first
        target_obs_dim = self.feature_dim - 4  # Reserve 4 dims for action info
        
        # Get observation features (pad/truncate to fit)
        obs_features = obs.get("features", [])
        
        # Convert to numpy array based on type
        if not obs_features:  # Handle empty case
            obs_features = np.zeros(target_obs_dim, dtype=np.float32)
        elif isinstance(obs_features, torch.Tensor):
            obs_features = obs_features.detach().cpu().numpy().astype(np.float32)
        elif isinstance(obs_features, (list, tuple)):
            obs_features = np.array(obs_features, dtype=np.float32)
        else:
            # Handle scalar or other types
            obs_features = np.array([obs_features], dtype=np.float32)
        
        # Ensure we have the right number of observation features
        if len(obs_features) >= target_obs_dim:
            obs_features = obs_features[:target_obs_dim]
        else:
            obs_features = np.pad(obs_features, (0, target_obs_dim - len(obs_features)))
        
        # Add action information
        strategic_action = action.get("strategic", 0)
        tactical_action = action.get("tactical", 0)
        
        action_features = np.array([
            strategic_action / max(self.action_dim_strategic, 1),
            tactical_action / max(self.action_dim_tactical, 1),
            float(tactical_action % 3 == 0),  # Pattern indicator
            float(strategic_action in [0, 1, 2])  # Early-stage indicator
        ], dtype=np.float32)
        
        # Combine features
        features = np.concatenate([obs_features, action_features])
        
        # Convert to tensor
        return torch.tensor(features[None, :], dtype=torch.float32, device=self.device)
    
    def compute_information_gain(self, obs: Dict[str, Any], action: Dict[str, Any], 
                               progress_delta: float) -> Tuple[float, float]:
        """
        Compute information gain using ensemble uncertainty.
        
        Trains ensemble heads to predict progress delta and uses prediction
        variance as uncertainty estimate. High variance = high information gain.
        
        Args:
            obs: Environment observation
            action: Agent action  
            progress_delta: Actual progress change
            
        Returns:
            Tuple of (information_gain, training_loss)
        """
        features = self.extract_features(obs, action)
        target = torch.tensor([[progress_delta]], dtype=torch.float32, device=self.device)
        
        # Forward pass through ensemble
        predictions = []
        total_loss = 0.0
        
        for head in self.ensemble_heads:
            pred = head(features)
            predictions.append(pred)
            loss = nn.MSELoss()(pred, target)
            total_loss += loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ensemble_heads.parameters(), 1.0)
        self.optimizer.step()
        
        # Compute variance across predictions
        predictions = torch.cat(predictions, dim=1)  # [1, k_ensemble]
        variance = torch.var(predictions).item()
        
        # Update running max for normalization
        self.var_max = max(self.var_max, variance + 1e-9)
        
        # Normalize information gain
        information_gain = min(variance / self.var_max, 1.0)
        
        return information_gain, total_loss.item()
    
    def compute_learning_progress(self, bucket: Tuple, loss: float) -> float:
        """
        Compute learning progress as improvement in prediction error.
        
        Learning progress rewards steps that improve the agent's internal
        model, not just steps that create surprise.
        
        Args:
            bucket: Structure signature for this state-action
            loss: Current prediction loss
            
        Returns:
            Learning progress value in [0, 1]
        """
        # Get previous EMA loss for this bucket
        if bucket in self.loss_ema:
            prev_ema = self.loss_ema[bucket][0]
        else:
            prev_ema = loss  # First time seeing this bucket
        
        # Update EMA
        current_ema = self.update_ema(self.loss_ema, bucket, loss)
        
        # Learning progress = reduction in loss
        if prev_ema > 1e-6:
            progress = max(0.0, (prev_ema - current_ema) / prev_ema)
        else:
            progress = 0.0
        
        return min(progress, 1.0)
    
    def compute_novelty(self, bucket: Tuple) -> float:
        """
        Compute structure-aware novelty using pseudo-counts.
        
        Uses mathematical structure signature rather than raw state
        to encourage exploration across problem types and reasoning modes.
        
        Args:
            bucket: Structure signature
            
        Returns:
            Novelty value in [0, 1]
        """
        count = self.counts[bucket]
        self.counts[bucket] += 1
        
        # Pseudo-count based novelty (decreases with visits)
        novelty = 1.0 / (1.0 + np.sqrt(count))
        return novelty
    
    def compute_zpd_shaping(self, bucket: Tuple, solved: bool) -> float:
        """
        Compute Zone of Proximal Development shaping.
        
        Encourages the agent to seek problems where success probability
        is near the target rate (typically 0.6 for flow state).
        
        Args:
            bucket: Structure signature
            solved: Whether the current episode was solved
            
        Returns:
            ZPD shaping value in [0, 1]
        """
        # Update success rate for this bucket
        success_rate = self.update_ema(self.success_ema, bucket, float(solved))
        
        # Gaussian preference around target success rate
        zpd_value = np.exp(-((success_rate - self.zpd_tau) ** 2) / (2 * self.zpd_sigma ** 2))
        
        return zpd_value
    
    def compute_boredom(self, action: Dict[str, Any], 
                       policy_logits: Optional[torch.Tensor] = None) -> float:
        """
        Compute boredom from policy entropy collapse or action repetition.
        
        Args:
            action: Current action
            policy_logits: Policy logits if available
            
        Returns:
            Boredom value in [0, 1]
        """
        if policy_logits is not None:
            # Use policy entropy if logits available
            with torch.no_grad():
                probs = torch.softmax(policy_logits, dim=-1)
                entropy = -(probs * torch.clamp(probs, 1e-9, 1.0).log()).sum().item()
                max_entropy = np.log(probs.numel())
                boredom = 1.0 - (entropy / max(max_entropy, 1e-6))
        else:
            # Fallback: detect action repetition
            tactical = action.get("tactical", 0)
            if hasattr(self, '_last_tactical') and tactical == self._last_tactical:
                boredom = 0.8  # High boredom for repetition
            else:
                boredom = 0.0
            self._last_tactical = tactical
        
        return min(max(boredom, 0.0), 1.0)
    
    def compute_stuckness(self, current_progress: float) -> float:
        """
        Compute stuckness from lack of progress over recent steps.
        
        Args:
            current_progress: Current progress value
            
        Returns:
            Stuckness value in [0, 1]
        """
        self.progress_history.append(current_progress)
        
        if len(self.progress_history) < self.stuck_k:
            return 0.0  # Not enough history
        
        # Check if any recent step made meaningful progress
        progress_deltas = np.diff(list(self.progress_history))
        max_recent_progress = np.max(progress_deltas) if len(progress_deltas) > 0 else 0.0
        
        stuck = 1.0 if max_recent_progress < self.progress_eps else 0.0
        return stuck
    
    def forward(self, obs: Dict[str, Any], action: Dict[str, Any], 
                next_obs: Dict[str, Any], solved: bool,
                policy_logits: Optional[torch.Tensor] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute intrinsic curiosity cost and reward.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            solved: Whether episode was solved
            policy_logits: Policy logits for entropy calculation (optional)
            
        Returns:
            Tuple of (curiosity_cost, curiosity_reward, component_values)
        """
        # Extract progress information
        current_progress = obs.get("progress", 0.0)
        next_progress = next_obs.get("progress", 0.0)
        progress_delta = next_progress - current_progress
        
        # Create structure signature
        bucket = self.create_structure_signature(obs, action)
        
        # Compute all components
        information_gain, prediction_loss = self.compute_information_gain(obs, action, progress_delta)
        learning_progress = self.compute_learning_progress(bucket, prediction_loss)
        novelty = self.compute_novelty(bucket)
        zpd_shaping = self.compute_zpd_shaping(bucket, solved)
        boredom = self.compute_boredom(action, policy_logits)
        stuckness = self.compute_stuckness(next_progress)
        
        # Unpack weights
        alpha1, alpha2, alpha3, alpha4 = self.alpha  # IG, LP, N, ZPD
        beta1, beta2 = self.beta  # Boredom, Stuckness
        
        # Compute intrinsic cost
        curiosity_terms = alpha1 * information_gain + alpha2 * learning_progress + \
                         alpha3 * novelty + alpha4 * zpd_shaping
        penalty_terms = beta1 * boredom + beta2 * stuckness
        
        # Cost equation: C_int = penalties - curiosity_benefits
        raw_cost = penalty_terms - curiosity_terms
        
        # Normalize to [0, 1] with 0.5 as neutral
        curiosity_cost = np.clip(0.5 + 0.5 * raw_cost, 0.0, 1.0)
        curiosity_reward = 1.0 - curiosity_cost
        
        # Component breakdown for logging
        components = {
            "information_gain": information_gain,
            "learning_progress": learning_progress, 
            "novelty": novelty,
            "zpd_shaping": zpd_shaping,
            "boredom": boredom,
            "stuckness": stuckness,
            "curiosity_cost": curiosity_cost,
            "curiosity_reward": curiosity_reward
        }
        
        return curiosity_cost, curiosity_reward, components
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics for monitoring."""
        return {
            "total_buckets_seen": len(self.counts),
            "most_common_buckets": self.counts.most_common(5),
            "variance_max": self.var_max,
            "progress_history_length": len(self.progress_history),
            "success_rates_tracked": len(self.success_ema),
            "loss_emas_tracked": len(self.loss_ema)
        }
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.progress_history.clear()
        if hasattr(self, '_last_tactical'):
            delattr(self, '_last_tactical')
    
    def save_state(self, filepath: str):
        """Save the module state."""
        state = {
            'model_state_dict': self.state_dict(),
            'counts': dict(self.counts),
            'success_ema': self.success_ema,
            'loss_ema': self.loss_ema,
            'var_max': self.var_max,
            'config': {
                'feature_dim': self.feature_dim,
                'action_dim_strategic': self.action_dim_strategic,
                'action_dim_tactical': self.action_dim_tactical,
                'zpd_tau': self.zpd_tau,
                'zpd_sigma': self.zpd_sigma,
                'alpha': self.alpha,
                'beta': self.beta,
                'stuck_k': self.stuck_k,
                'progress_eps': self.progress_eps,
                'ema_beta': self.ema_beta
            }
        }
        torch.save(state, filepath)
        print(f"✅ Curiosity module state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the module state."""
        state = torch.load(filepath, map_location=self.device)
        self.load_state_dict(state['model_state_dict'])
        self.counts = collections.Counter(state['counts'])
        self.success_ema = state['success_ema']
        self.loss_ema = state['loss_ema']
        self.var_max = state['var_max']
        print(f"✅ Curiosity module state loaded from {filepath}")


def create_curiosity_module(config: Optional[Dict[str, Any]] = None) -> MathematicalCuriosityCost:
    """
    Factory function to create a Mathematical Curiosity Cost module with sensible defaults.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured MathematicalCuriosityCost instance
    """
    default_config = {
        'feature_dim': 32,
        'action_dim_strategic': 6,
        'action_dim_tactical': 12,
        'zpd_tau': 0.6,
        'zpd_sigma': 0.2,
        'alpha': (0.35, 0.25, 0.20, 0.20),
        'beta': (0.40, 0.30),
        'k_ensemble': 4,
        'stuck_k': 4,
        'progress_eps': 1e-3,
        'ema_beta': 0.9,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if config:
        default_config.update(config)
    
    return MathematicalCuriosityCost(**default_config)