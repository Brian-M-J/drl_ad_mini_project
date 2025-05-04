# custom_env.py
import numpy as np
import gymnasium as gym
from gymnasium import register
import warnings # Import warnings

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env import utils

class CustomEnv(AbstractEnv):
    """
    A custom HighwayEnv environment with a 2-lane circular road where all vehicles
    move counter-clockwise, using KinematicsObservation. (Reward calculation fixed)
    """

    @classmethod
    def default_config(cls) -> dict:
        """Define default configuration settings."""
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics", "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"x": [-100, 100], "y": [-15, 15], "vx": [-30, 30], "vy": [-30, 30]},
                "absolute": False, "normalize": True, "flatten": True, "see_behind": True
            },
            "action": {"type": "DiscreteMetaAction", "target_speeds": np.linspace(0, 30, 5)},
            "lanes_count": 2, "vehicles_count": 10, "controlled_vehicles": 1,
            "initial_lane_id": None, "duration": 40, "ego_spacing": 2, "vehicles_density": 1,
            "collision_reward": -5.0, # Weight for collision penalty
            "right_lane_reward": 0.1, # Weight for right lane reward
            "high_speed_reward": 1.0, # Weight for high speed reward
            "low_speed_penalty": 0.5, # Weight for low speed penalty (changed name)
            "lane_change_penalty": 0.05, # Weight for lane change penalty (changed name)
            # "on_road_reward": 1.0, # Removed direct weight, handled differently
            "reward_speed_range": [10, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "offscreen_terminal": False,
            "simulation_frequency": 15, "policy_frequency": 5,
            "screen_width": 800, "screen_height": 800, "centering_position": [0.5, 0.5],
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
        })
        return config

    def _reward(self, action: int) -> float:
        """Calculate the total reward based on individual reward components."""
        rewards = self._rewards(action) # Get dictionary of component values [-1, 1] or [0, 1]

        # Calculate raw weighted reward
        raw_reward = 0
        raw_reward += self.config["collision_reward"] * rewards["collision"]
        raw_reward += self.config["high_speed_reward"] * rewards["high_speed"]
        raw_reward -= self.config["low_speed_penalty"] * rewards["low_speed"] # Subtract penalty
        raw_reward += self.config["right_lane_reward"] * rewards["right_lane"]
        raw_reward -= self.config["lane_change_penalty"] * rewards["lane_change"] # Subtract penalty

        # --- Normalization Fix ---
        # Define theoretical min/max *possible* raw reward based on weights
        # Assumes component rewards are in [0, 1] range (except collision which is 0 or 1)
        min_raw_reward = self.config["collision_reward"] * 1 - self.config["low_speed_penalty"] * 1 - self.config["lane_change_penalty"] * 1
        max_raw_reward = self.config["high_speed_reward"] * 1 + self.config["right_lane_reward"] * 1

        if self.config["normalize_reward"]:
            if max_raw_reward > min_raw_reward: # Avoid division by zero
                 # Scale raw reward to [-1, 1]
                 reward = utils.lmap(raw_reward, [min_raw_reward, max_raw_reward], [-1, 1])
            else:
                 # Handle case where min/max are the same (e.g., all weights zero)
                 reward = 0.0
                 warnings.warn("Reward normalization skipped: min_raw_reward >= max_raw_reward", UserWarning)
        else:
            reward = raw_reward # Use raw reward if not normalizing

        # --- Off-road Handling Fix ---
        # Apply heavy penalty if off-road, instead of just multiplying by zero
        if not rewards["on_road"]:
            reward = -1.0 # Assign maximum penalty if off-road

        # Ensure collision results in the most negative reward (-1 after normalization)
        if rewards["collision"]:
             reward = -1.0

        # Clip final reward to be safe
        reward = np.clip(reward, -1.0, 1.0)

        # --- Debug Print (Optional) ---
        # print(f"Step: {self.time:.2f}, Raw: {raw_reward:.2f}, Norm: {reward:.2f}, Speed: {self.vehicle.speed:.1f}, OnRoad: {rewards['on_road']}, Collision: {rewards['collision']}")
        # ---

        return float(reward)


    def _rewards(self, action: int) -> dict[str, float]:
        """
        Define individual reward components, scaled primarily between 0 and 1.
        Returns: Dictionary with component values.
        """
        current_speed = self.vehicle.speed
        scaled_speed = utils.lmap(current_speed, self.config["reward_speed_range"], [0, 1])
        # Use lane_index from vehicle, check if it's valid before accessing index -1
        lane_idx = self.vehicle.lane_index[-1] if self.vehicle.lane_index is not None else self.config["lanes_count"] - 1
        on_road = self.vehicle.on_road

        # Calculate individual components (mostly 0 or 1)
        collision = float(self.vehicle.crashed)
        high_speed = np.clip(scaled_speed, 0, 1) # Speed component (0 to 1)
        low_speed = float(current_speed < self.config["reward_speed_range"][0] and not self.vehicle.crashed) # Penalty active if slow and not crashed
        right_lane = float(lane_idx == self.config["lanes_count"] - 1 and on_road) # 1 if on rightmost lane and on road
        lane_change_action = action in [0, 2] # Assuming 0=LEFT, 2=RIGHT
        lane_change = float(lane_change_action)

        return {
            "collision": collision,       # 1 if collision, 0 otherwise
            "high_speed": high_speed,     # Scaled speed [0, 1]
            "low_speed": low_speed,       # 1 if too slow, 0 otherwise
            "right_lane": right_lane,     # 1 if on right lane, 0 otherwise
            "lane_change": lane_change,   # 1 if lane change action, 0 otherwise
            "on_road": float(on_road),    # 1 if on road, 0 otherwise
        }

    def _is_terminated(self) -> bool:
        """Check if the episode terminates (e.g., due to a crash)."""
        # Terminate on collision or if offroad_terminal is True and vehicle is off-road
        return self.vehicle.crashed or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """Check if the episode is truncated (e.g., time limit reached or offscreen)."""
        return self.time >= self.config["duration"] or \
               (self.config["offscreen_terminal"] and not self.vehicle.is_in_fov())


    def _reset(self) -> None:
        """Reset the environment by creating the road and vehicles."""
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """Create a 2-lane circular road with counter-clockwise movement."""
        center = [0, 0]; radius = 50; lanes_count = self.config["lanes_count"]; lane_width = 4
        net = RoadNetwork()
        radii = [radius + i * lane_width for i in range(lanes_count)]
        line_types = [[LineType.CONTINUOUS, LineType.STRIPED]] + \
                     [[LineType.STRIPED, LineType.STRIPED]] * (lanes_count - 2) + \
                     [[LineType.STRIPED, LineType.NONE]] if lanes_count > 1 else [[LineType.CONTINUOUS, LineType.NONE]]
        for lane_idx in range(lanes_count):
            lane_id_prefix = f"c{lane_idx}"
            lane = CircularLane(center, radii[lane_idx], 0, 2 * np.pi, clockwise=False,
                                line_types=line_types[lane_idx], speed_limit=30)
            net.add_lane(lane_id_prefix, lane_id_prefix, lane)
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_vehicles(self) -> None:
        """Populate the road with the ego vehicle and other vehicles."""
        rng = self.np_random
        ego_lane_idx = rng.integers(self.config["lanes_count"]) if self.config["initial_lane_id"] is None else self.config["initial_lane_id"]
        ego_lane_id = f"c{ego_lane_idx}"
        ego_lane = self.road.network.get_lane((ego_lane_id, ego_lane_id, 0))
        ego_start_pos = ego_lane.length * rng.uniform(0, 1)
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_lane.position(ego_start_pos, 0),
            heading=ego_lane.heading_at(ego_start_pos), speed=rng.uniform(15, 25)
        )
        ego_vehicle.plan_route_to(ego_lane_id)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            while True:
                lane_idx = rng.integers(self.config["lanes_count"])
                lane_id = f"c{lane_idx}"
                lane = self.road.network.get_lane((lane_id, lane_id, 0))
                start_pos = ego_lane.length * rng.uniform(0, 1)
                position = lane.position(start_pos, 0)
                if not any(np.linalg.norm(np.array(position) - np.array(v.position)) < self.config["ego_spacing"] * v.LENGTH for v in self.road.vehicles):
                    break
            vehicle = other_vehicles_type(
                self.road, position, heading=lane.heading_at(start_pos), speed=rng.uniform(10, 20)
            )
            vehicle.plan_route_to(lane_id)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

# --- Registration ---
try:
    register(id='custom-kinematic-v0', entry_point='custom_env:CustomEnv')
    # print("Successfully registered custom-kinematic-v0 environment.") # Already printed in train.py etc.
except gym.error.Error as e:
    # print(f"Environment 'custom-kinematic-v0' might already be registered: {e}")
    pass # Ignore if already registered

# --- Example Usage Guard ---
if __name__ == "__main__":
    print("Testing CustomEnv with KinematicsObservation (Reward Fix)...")
    env = gym.make('custom-kinematic-v0', render_mode='human')
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    obs, info = env.reset()
    terminated = truncated = False
    step = 0
    total_reward = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1
        total_reward += reward
        if step % 20 == 0 or terminated or truncated:
            print(f"Step: {step}, Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}")
        if terminated or truncated:
            print(f"Episode finished. Terminated={terminated}, Truncated={truncated}")
    env.close()
    print("Environment test finished.")
