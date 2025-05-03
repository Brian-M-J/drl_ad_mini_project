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
    Simple MLP features extractor for Dueling DQN.
    It outputs the final layer features before splitting into Value and Advantage streams.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # Assuming observation_space is a flattened 1D Box
        input_dim = observation_space.shape[0]
        # Example MLP: Adjust layer sizes as needed
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), # 1st Hidden Layer
            nn.ReLU(),
            nn.Linear(256, 256),      # 2nd Hidden Layer
            nn.ReLU(),
            nn.Linear(256, 256),      # 3rd Hidden Layer
            nn.ReLU(),
            nn.Linear(256, features_dim), # Output Layer (features_dim)
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure input is float32
        return self.net(observations.float())

class DuelingQNetwork(BasePolicy):
    """
    Dueling Action-Value (Q-Value) network.
    Splits the output from a features extractor into separate value and advantage streams.

    :param observation_space: Observation space
    :param action_space: Action space
    :param features_extractor: Network to extract features
        (must be compatible with DuelingMlpFeaturesExtractor or similar)
    :param features_dim: Number of features extracted.
    :param net_arch: The specification of the network architecture for value and advantage streams.
                     Default is [64].
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not, defaults to True.
                               (Not relevant for MLP but kept for compatibility)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True, # Keep for SB3 compatibility
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images, # Pass along
        )

        if net_arch is None:
            net_arch = [64] # Default hidden layer size for value/advantage streams

        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.action_dim = action_space.n # type: ignore

        # Build Value stream
        value_net = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.value_net = nn.Sequential(*value_net)

        # Build Advantage stream
        advantage_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, self.activation_fn)
        self.advantage_net = nn.Sequential(*advantage_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # Extract features (assumes observations are already preprocessed/normalized if needed)
        features = self.extract_features(obs) # Uses the passed features_extractor

        # Calculate value and advantage
        value = self.value_net(features)
        advantages = self.advantage_net(features)

        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        Used internally by SB3 agents.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        q_values = self.forward(observation)
        # Greedy action selection
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    # Required by SB3 DQNPolicy structure, even if features_extractor handles it
    def set_training_mode(self, mode: bool) -> None:
        self.features_extractor.train(mode)
        self.advantage_net.train(mode)
        self.value_net.train(mode)
        self.training = mode


class DuelingMlpPolicy(DQNPolicy):
    """
    Policy class with Dueling Q-Network using MLP architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the network architecture. If None, default DQN arch is used.
                     This is primarily for the value/advantage streams AFTER feature extraction.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor constructor.
    :param normalize_images: Whether to normalize images or not, defaults to True.
    :param optimizer_class: The optimizer to use, ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: DuelingQNetwork
    q_net_target: DuelingQNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule, # Function from float -> float
        net_arch: Optional[List[int]] = None, # For value/advantage streams
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = DuelingMlpFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True, # Keep for compatibility, less relevant for MLP
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Set features_dim in kwargs if not provided, default to 64
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        if "features_dim" not in features_extractor_kwargs:
             features_extractor_kwargs["features_dim"] = 64 # Default feature dim

        # Call DQNPolicy constructor AFTER setting up kwargs
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch, # Passed to DuelingQNetwork constructor later
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        # Note: _build() is called within the super().__init__()

    def make_q_net(self) -> DuelingQNetwork:
        """
        Creates the Dueling Q-Network.
        This is called by the superclass's _setup_model method.
        """
        # features_extractor is already initialized by BasePolicy's constructor
        features_extractor = self.features_extractor
        features_dim = self.features_extractor.features_dim

        # Pass the features extractor instance and its output dim to the Q network
        return DuelingQNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=self.net_arch, # Architecture for value/advantage streams
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
        ).to(self.device)

    # _build method is implicitly handled by DQNPolicy calling make_q_net
    # and setting up optimizer etc. in its _setup_model.

    # Forward pass delegates to the q_net
    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    # set_training_mode is inherited from DQNPolicy which calls it on q_net and q_net_target

# --- Example Usage Guard ---
if __name__ == "__main__":
    print("Testing DuelingMlpPolicy structure (requires dummy spaces)...")
    from gymnasium.spaces import Box, Discrete
    import numpy as np

    obs_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32) # Example 1D observation
    act_space = Discrete(5) # Example discrete actions

    # Dummy learning rate schedule
    def constant_lr_schedule(progress_remaining):
        return 3e-4 # Example constant learning rate

    try:
        # Instantiate the policy
        policy = DuelingMlpPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=constant_lr_schedule,
            features_extractor_kwargs={"features_dim": 128}, # Example feature dim
            net_arch=[256, 128] # Example arch for value/adv streams
        )
        print("Policy Instantiated Successfully:")
        print(policy)

        # Test forward pass with dummy data
        dummy_obs = torch.tensor(obs_space.sample(), dtype=torch.float32).unsqueeze(0) # Add batch dim
        print("\nDummy Observation shape:", dummy_obs.shape)
        with torch.no_grad():
            action = policy(dummy_obs)
            q_values = policy.q_net(dummy_obs) # Get Q-values directly
        print("Predicted Action:", action.item())
        print("Predicted Q-Values:", q_values)

    except Exception as e:
        print(f"Error during policy instantiation or test: {e}")

