# custom_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from gymnasium import spaces
from typing import List, Type, Optional, Tuple, Dict, Any

class DuelingMlpFeaturesExtractor(BaseFeaturesExtractor):
    """
    MLP features extractor for Dueling DQN with 3 hidden layers.
    Increased capacity: Input -> 512 -> 256 -> features_dim
    It outputs the final layer features before splitting into Value and Advantage streams.
    """
    # --- Increased Capacity ---
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128): # Increased default features_dim
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        # Three fully connected layers: Input -> 512 -> 256 -> features_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), # 1st Hidden Layer (Wider)
            nn.ReLU(),
            nn.Linear(512, 512),      # 2nd Hidden Layer
            nn.ReLU(),
            nn.Linear(512, 512),      # 3rd Hidden Layer
            nn.ReLU(),
            nn.Linear(512, features_dim), # Output Layer (features_dim=128)
            nn.ReLU()
        )
        # ------------------------

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations.float())

class DuelingQNetwork(BasePolicy):
    """
    Dueling Action-Value (Q-Value) network.
    Splits the output from a features extractor into separate value and advantage streams.
    Uses increased default net_arch.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        # --- Updated Default Architecture for Value/Advantage Streams ---
        net_arch: Optional[List[int]] = None,
        # --------------------------------------------------------------
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # --- Updated Default Architecture ---
        if net_arch is None:
            # Default hidden layer size(s) for the value/advantage streams
            net_arch = [256, 128] # Increased default: 2 layers
        # ---------------------------------

        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError("DuelingQNetwork currently only supports Discrete action spaces.")

        # Build Value stream
        value_net = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.value_net = nn.Sequential(*value_net)

        # Build Advantage stream
        advantage_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, self.activation_fn)
        self.advantage_net = nn.Sequential(*advantage_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        value = self.value_net(features)
        advantages = self.advantage_net(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def set_training_mode(self, mode: bool) -> None:
        self.features_extractor.train(mode)
        self.advantage_net.train(mode)
        self.value_net.train(mode)
        self.training = mode


class DuelingMlpPolicy(DQNPolicy):
    """
    Policy class with Dueling Q-Network using MLP architecture.
    Uses the updated DuelingMlpFeaturesExtractor and DuelingQNetwork defaults.
    """
    q_net: DuelingQNetwork
    q_net_target: DuelingQNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[List[int]] = None, # For value/advantage streams
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = DuelingMlpFeaturesExtractor, # Use updated extractor
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        # Default features_dim for the updated extractor is 128
        if "features_dim" not in features_extractor_kwargs:
             features_extractor_kwargs["features_dim"] = 128

        self.net_arch_q_value = net_arch # Store arch for Q-network streams

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None, # Pass None here
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> DuelingQNetwork:
        """Creates the Dueling Q-Network using updated defaults."""
        features_extractor = self.features_extractor
        features_dim = self.features_extractor.features_dim

        # DuelingQNetwork will use its updated default net_arch ([256, 128]) if self.net_arch_q_value is None
        return DuelingQNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=self.net_arch_q_value, # Pass the specific arch if provided, otherwise DuelingQNetwork uses its default
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
        ).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

# --- Example Usage Guard ---
if __name__ == "__main__":
    print("Testing DuelingMlpPolicy structure (requires dummy spaces)...")
    from gymnasium.spaces import Box, Discrete
    import numpy as np

    # Example obs space: 5 vehicles * 7 features = 35
    # The input layer of the feature extractor will adapt to this size.
    obs_space = Box(low=-1, high=1, shape=(35,), dtype=np.float32)
    act_space = Discrete(5) # Example discrete actions

    def constant_lr_schedule(progress_remaining): return 3e-4

    try:
        # Instantiate the policy. It will use the updated feature extractor
        # and the updated default net_arch ([256, 128]) for value/adv streams
        # unless overridden here. features_dim will default to 128.
        policy = DuelingMlpPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=constant_lr_schedule,
            # features_extractor_kwargs={"features_dim": 128}, # Explicitly set if needed
            # net_arch=[512, 256] # Explicitly override value/adv stream arch if needed
        )
        print("Policy Instantiated Successfully:")
        print(policy)

        dummy_obs = torch.tensor(obs_space.sample(), dtype=torch.float32).unsqueeze(0)
        print("\nDummy Observation shape:", dummy_obs.shape)
        with torch.no_grad():
            action = policy(dummy_obs)
            q_values = policy.q_net(dummy_obs)
        print("Predicted Action:", action.item())
        print("Predicted Q-Values:", q_values)

    except Exception as e:
        print(f"Error during policy instantiation or test: {e}")
        import traceback
        traceback.print_exc()
