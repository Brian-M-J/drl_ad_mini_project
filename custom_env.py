# custom_env.py
import numpy as np
import gymnasium as gym
from gymnasium import register

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env import utils

class CustomEnv(AbstractEnv):
    """
    A custom HighwayEnv environment with a 2-lane circular road where all vehicles
    move counter-clockwise, using KinematicsObservation.
    """

    @classmethod
    def default_config(cls) -> dict:
        """Define default configuration settings."""
        config = super().default_config()
        # --- Updated Configuration ---
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,  # Observe ego + 4 nearest vehicles
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": { # Normalization ranges
                    "x": [-100, 100],
                    "y": [-15, 15],
                    "vx": [-30, 30],
                    "vy": [-30, 30],
                },
                "absolute": False, # Use relative features for other vehicles
                "normalize": True, # Normalize observation features
                "flatten": True,   # Flatten observation into a 1D vector
                "see_behind": True # Allow seeing vehicles behind
            },
            "action": {
                "type": "DiscreteMetaAction", # Discrete actions (e.g., lane change, speed change)
                "target_speeds": np.linspace(0, 30, 5) # Example: 5 target speeds [0, 7.5, 15, 22.5, 30] m/s
            },
            "lanes_count": 2,
            "vehicles_count": 10, # Number of background vehicles in the environment
            "controlled_vehicles": 1,
            "initial_lane_id": None, # Random initial lane
            "duration": 40,  # seconds, episode length
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -5.0, # Increased penalty for collision
            "right_lane_reward": 0.1, # Reward for being in the right lane (usually safer)
            "high_speed_reward": 1.0, # Reward for high speed (scaled by speed)
            "low_speed_reward": -1.0, # Penalty for low speed
            "lane_change_reward": -0.1, # Slight penalty for changing lanes
            "on_road_reward": 1.0, # Reward for staying on road
            "reward_speed_range": [10, 30], # Speed range for scaling high_speed_reward [m/s]
            "normalize_reward": True, # Scale reward to [-1, 1] (helps training stability)
            "offroad_terminal": False, # Episode doesn't end if agent goes off-road
            "offscreen_terminal": False, # Episode doesn't end if agent goes off-screen
            "simulation_frequency": 15, # Simulation physics frequency [Hz]
            "policy_frequency": 5,      # Agent decision frequency [Hz]
            "screen_width": 800,
            "screen_height": 800, # Adjusted for better circular view
            "centering_position": [0.5, 0.5],
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
        })
        # ---------------------------
        return config

    def _reward(self, action: int) -> float:
        """Calculate the total reward based on individual reward components."""
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward_value
                     for name, reward_value in rewards.items())

        if self.config["normalize_reward"]:
            # Normalize reward to [-1, 1] based on max/min possible rewards
            # Adjust these bounds based on your config weights
            min_reward = self.config["collision_reward"] + self.config["lane_change_reward"] + self.config["low_speed_reward"]
            max_reward = self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["on_road_reward"]
            reward = utils.lmap(reward, [min_reward, max_reward], [-1, 1])

        # Penalize heavily for collisions
        reward = reward if not self.vehicle.crashed else -1.0
        # Ensure reward is zero if off-road (unless offroad_terminal is True)
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        """Define individual reward components."""
        # Get current speed and lane index
        current_speed = self.vehicle.speed
        scaled_speed = utils.lmap(current_speed, self.config["reward_speed_range"], [0, 1])
        lane_index = self.vehicle.lane_index if self.vehicle.lane_index is not None else ('a', 'b', self.config["lanes_count"] - 1) # Default to rightmost if unknown
        on_road = self.vehicle.on_road

        # Calculate individual rewards
        collision_reward = float(self.vehicle.crashed)
        high_speed_reward = np.clip(scaled_speed, 0, 1) # Reward proportional to speed in range
        low_speed_reward = float(current_speed < self.config["reward_speed_range"][0]) # Penalty if below min speed
        right_lane_reward = float(lane_index[-1] == self.config["lanes_count"] - 1) if on_road else 0.0
        lane_change_action = action in [0, 2] # Assuming action 0 is LEFT, 2 is RIGHT in DiscreteMetaAction
        lane_change_reward = float(lane_change_action)
        on_road_reward = float(on_road)

        return {
            "collision_reward": collision_reward,
            "high_speed_reward": high_speed_reward,
            "low_speed_reward": low_speed_reward,
            "right_lane_reward": right_lane_reward,
            "lane_change_reward": lane_change_reward,
            "on_road_reward": on_road_reward,
        }

    def _is_terminated(self) -> bool:
        """Check if the episode terminates (e.g., due to a crash)."""
        return self.vehicle.crashed or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road) or \
               (self.config["offscreen_terminal"] and not self.vehicle.is_in_fov())


    def _is_truncated(self) -> bool:
        """Check if the episode is truncated (e.g., time limit reached)."""
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        """Reset the environment by creating the road and vehicles."""
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """Create a 2-lane circular road with counter-clockwise movement."""
        center = [0, 0]  # [m]
        radius = 50  # [m] Adjusted radius
        lanes_count = self.config["lanes_count"]
        lane_width = 4 # [m]

        net = RoadNetwork()
        radii = [radius + i * lane_width for i in range(lanes_count)]
        # line types: [outer_most ..., inner_most]
        line_types = [[LineType.CONTINUOUS, LineType.STRIPED]] + \
                     [[LineType.STRIPED, LineType.STRIPED]] * (lanes_count - 2) + \
                     [[LineType.STRIPED, LineType.NONE]] if lanes_count > 1 else [[LineType.CONTINUOUS, LineType.NONE]]

        # Create lanes moving counter-clockwise
        for lane_idx in range(lanes_count):
            lane_id_prefix = f"c{lane_idx}" # Unique prefix for each lane's segments
            lane = CircularLane(center, radii[lane_idx], 0, 2 * np.pi, clockwise=False,
                                line_types=line_types[lane_idx], speed_limit=30)
            net.add_lane(lane_id_prefix, lane_id_prefix, lane) # Lane connects back to itself

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _make_vehicles(self) -> None:
        """Populate the road with the ego vehicle and other vehicles."""
        rng = self.np_random

        # --- Ego Vehicle ---
        ego_lane_idx = rng.integers(self.config["lanes_count"]) if self.config["initial_lane_id"] is None else self.config["initial_lane_id"]
        ego_lane_id = f"c{ego_lane_idx}"
        ego_lane = self.road.network.get_lane((ego_lane_id, ego_lane_id, 0)) # Lane index within segment is 0 for circular
        ego_start_pos = ego_lane.length * rng.uniform(0, 1) # Random start position

        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(ego_start_pos, 0), # Position on the lane center
            heading=ego_lane.heading_at(ego_start_pos),
            speed=rng.uniform(15, 25) # Random initial speed
        )
        # Set a long route to loop around the circle
        ego_vehicle.plan_route_to(ego_lane_id) # Plan route to stay on the same circular lane
        ego_vehicle.speed_limit = 35 # Slightly higher limit for ego if desired
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle # Assign to AbstractEnv's vehicle attribute

        # --- Other Vehicles ---
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicles_count = self.config["vehicles_count"]

        for _ in range(vehicles_count):
            # Choose random lane and position, ensuring minimum spacing from ego and other vehicles
            while True:
                lane_idx = rng.integers(self.config["lanes_count"])
                lane_id = f"c{lane_idx}"
                lane = self.road.network.get_lane((lane_id, lane_id, 0))
                start_pos = ego_lane.length * rng.uniform(0, 1)

                # Check spacing with existing vehicles
                position = lane.position(start_pos, 0)
                too_close = any(
                    np.linalg.norm(np.array(position) - np.array(v.position)) < self.config["ego_spacing"] * v.LENGTH
                    for v in self.road.vehicles
                )

                if not too_close:
                    break # Found a suitable position

            vehicle = other_vehicles_type(
                self.road,
                position,
                heading=lane.heading_at(start_pos),
                speed=rng.uniform(10, 20), # Other vehicles are slower
                # target_speed=rng.uniform(10, 20) # For IDMVehicle
            )
            vehicle.plan_route_to(lane_id) # Stay on their circular lane
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

# --- Registration ---
# Make sure this runs when the module is imported
try:
    register(
        id='custom-kinematic-v0', # New ID for the kinematic version
        entry_point='custom_env:CustomEnv',
    )
    print("Successfully registered custom-kinematic-v0 environment.")
except gym.error.Error as e:
    print(f"Environment 'custom-kinematic-v0' might already be registered: {e}")


# --- Example Usage Guard ---
if __name__ == "__main__":
    print("Testing CustomEnv with KinematicsObservation...")
    # Use the new ID
    env = gym.make('custom-kinematic-v0', render_mode='human')

    # --- Print Env Details ---
    print("Observation Space:", env.observation_space)
    print("Sample Observation:", env.observation_space.sample())
    print("Action Space:", env.action_space)
    print("Sample Action:", env.action_space.sample())
    # ---

    obs, info = env.reset()
    print("Initial Observation:", obs)
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = env.action_space.sample()  # Random action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1
        if step % 20 == 0:
            print(f"Step: {step}, Action: {action}, Reward: {reward:.2f}")
            # print("Current Observation:", obs) # Can be verbose
        if terminated or truncated:
            print(f"Episode finished after {step} steps. Terminated={terminated}, Truncated={truncated}")
            # obs, info = env.reset() # Uncomment to run multiple episodes
            # terminated = truncated = False
            # step = 0

    env.close()
    print("Environment test finished.")

