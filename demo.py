# custom_env.py
import numpy as np
import gymnasium as gym
from gymnasium import register
import warnings # Import warnings

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env import utils

class CustomEnv(AbstractEnv):
    """
    A custom HighwayEnv environment with a 2-lane circular road where all vehicles
    move counter-clockwise, using KinematicsObservation.
    (Road rendering fixed by reverting to segment-based road construction)
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
            "reward_speed_range": [10, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "offscreen_terminal": False,
            "simulation_frequency": 15, "policy_frequency": 5,
            "screen_width": 800, "screen_height": 800, "centering_position": [0.5, 0.5],
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            # Road configuration (used in _make_road)
            "road_radius": 50, # Radius of the inner lane edge
            "road_lane_width": 4,
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

        min_raw_reward = self.config["collision_reward"] * 1 - self.config["low_speed_penalty"] * 1 - self.config["lane_change_penalty"] * 1
        max_raw_reward = self.config["high_speed_reward"] * 1 + self.config["right_lane_reward"] * 1

        if self.config["normalize_reward"]:
            if max_raw_reward > min_raw_reward: # Avoid division by zero
                 reward = utils.lmap(raw_reward, [min_raw_reward, max_raw_reward], [-1, 1])
            else:
                 reward = 0.0
                 warnings.warn("Reward normalization skipped: min_raw_reward >= max_raw_reward", UserWarning)
        else:
            reward = raw_reward

        if not rewards["on_road"]: reward = -1.0 # Heavy penalty if off-road
        if rewards["collision"]: reward = -1.0 # Ensure collision is max penalty

        reward = np.clip(reward, -1.0, 1.0)
        return float(reward)


    def _rewards(self, action: int) -> dict[str, float]:
        """Define individual reward components, scaled primarily between 0 and 1."""
        current_speed = self.vehicle.speed
        scaled_speed = utils.lmap(current_speed, self.config["reward_speed_range"], [0, 1])
        lane_idx = self.vehicle.lane_index[-1] if self.vehicle.lane_index is not None else self.config["lanes_count"] - 1
        on_road = self.vehicle.on_road

        collision = float(self.vehicle.crashed)
        high_speed = np.clip(scaled_speed, 0, 1)
        low_speed = float(current_speed < self.config["reward_speed_range"][0] and not self.vehicle.crashed)
        right_lane = float(lane_idx == self.config["lanes_count"] - 1 and on_road)
        lane_change_action = action in [0, 2] # Assuming 0=LEFT, 2=RIGHT
        lane_change = float(lane_change_action)

        return {
            "collision": collision, "high_speed": high_speed, "low_speed": low_speed,
            "right_lane": right_lane, "lane_change": lane_change, "on_road": float(on_road),
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"] or (self.config["offscreen_terminal"] and not self.vehicle.is_in_fov())

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Create a circular road using connected segments (similar to the old method).
        This version should render correctly.
        """
        lanes_count = self.config["lanes_count"]
        radius = self.config["road_radius"]
        lane_width = self.config["road_lane_width"]
        center = [0, 0]  # Road center

        net = RoadNetwork()
        radii = [radius + i * lane_width for i in range(lanes_count)]

        # Define line types: [outer_most(0) ..., inner_most(N-1)]
        line_types = []
        if lanes_count == 1:
            line_types = [[LineType.CONTINUOUS, LineType.CONTINUOUS]] # Treat single lane as bounded
        else:
            line_types.append([LineType.CONTINUOUS, LineType.STRIPED]) # Outer edge
            for _ in range(lanes_count - 2):
                line_types.append([LineType.STRIPED, LineType.STRIPED]) # Middle lanes
            line_types.append([LineType.STRIPED, LineType.CONTINUOUS]) # Inner edge

        # Define segments (e.g., 8 segments for a smoother circle appearance)
        segment_count = 8
        segment_angle = 2 * np.pi / segment_count
        nodes = [f"c{i}" for i in range(segment_count)] # Node names c0, c1, ...

        # Create lane segments for each lane index
        for lane_idx in range(lanes_count):
            current_radius = radii[lane_idx]
            current_line_types = line_types[lane_idx]
            for i in range(segment_count):
                start_node = nodes[i]
                end_node = nodes[(i + 1) % segment_count] # Loop back to c0
                start_angle = i * segment_angle
                end_angle = (i + 1) * segment_angle
                # Add the circular lane segment to the network
                net.add_lane(
                    start_node,
                    end_node,
                    CircularLane(
                        center=center,
                        radius=current_radius,
                        start_phase=start_angle,
                        end_phase=end_angle,
                        clockwise=False, # Counter-clockwise
                        line_types=current_line_types,
                        speed_limit=30,
                    ),
                    lane_id=lane_idx # Assign lane index explicitly
                )

        # --- IMPORTANT: Assign self.road AFTER the loop ---
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        # --------------------------------------------------


    def _make_vehicles(self) -> None:
        """Populate the road with the ego vehicle and other vehicles."""
        rng = self.np_random

        # --- Ego Vehicle ---
        # Choose a random starting segment and lane
        start_node_idx = rng.integers(8) # 8 segments
        start_node = f"c{start_node_idx}"
        end_node = f"c{(start_node_idx + 1) % 8}"
        ego_lane_idx = rng.integers(self.config["lanes_count"]) if self.config["initial_lane_id"] is None else self.config["initial_lane_id"]

        # Get the lane object
        ego_lane = self.road.network.get_lane((start_node, end_node, ego_lane_idx))
        ego_start_pos = ego_lane.length * rng.uniform(0.1, 0.9) # Start somewhere in the middle of the segment

        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(ego_start_pos, 0), # Position on the lane center
            heading=ego_lane.heading_at(ego_start_pos),
            speed=rng.uniform(15, 25)
        )
        # Plan a route that goes around the circle multiple times
        route = [(f"c{i}", f"c{(i + 1) % 8}", ego_lane_idx) for i in range(start_node_idx, start_node_idx + segment_count * 5)] # 5 laps
        ego_vehicle.plan_route(route)
        ego_vehicle.speed_limit = 35
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # --- Other Vehicles ---
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        segment_count = 8 # Match the road definition

        for _ in range(self.config["vehicles_count"]):
            while True:
                # Choose random lane, segment, and position
                lane_idx = rng.integers(self.config["lanes_count"])
                start_idx = rng.integers(segment_count)
                start_node = f"c{start_idx}"
                end_node = f"c{(start_idx + 1) % segment_count}"
                lane = self.road.network.get_lane((start_node, end_node, lane_idx))
                start_pos = lane.length * rng.uniform(0, 1)
                position = lane.position(start_pos, 0)

                # Check spacing
                if not any(np.linalg.norm(np.array(position) - np.array(v.position)) < self.config["ego_spacing"] * v.LENGTH for v in self.road.vehicles):
                    break # Found a suitable position

            vehicle = other_vehicles_type(
                self.road,
                position,
                heading=lane.heading_at(start_pos),
                speed=rng.uniform(10, 20),
            )
            # Plan route for other vehicles
            route = [(f"c{i}", f"c{(i + 1) % segment_count}", lane_idx) for i in range(start_idx, start_idx + segment_count * 5)] # 5 laps
            vehicle.plan_route(route)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

# --- Registration ---
try:
    register(id='custom-kinematic-v0', entry_point='custom_env:CustomEnv')
except gym.error.Error as e:
    pass # Ignore if already registered

# --- Example Usage Guard ---
if __name__ == "__main__":
    print("Testing CustomEnv with KinematicsObservation (Road Rendering Fix)...")
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
