# evaluate.py
import os
import argparse
import time
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

# Import custom environment - ensures registration
import custom_env

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv # For potential single-threaded eval

# Define the common config here, MUST match training config
common_config = {
    "observation": {
        "type": "Kinematics", "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {"x": [-100, 100], "y": [-15, 15], "vx": [-30, 30], "vy": [-30, 30]},
        "absolute": False, "normalize": True, "flatten": True, "see_behind": True
    },
    "action": {"type": "DiscreteMetaAction", "target_speeds": np.linspace(0, 30, 5)},
    "duration": 40, "collision_reward": -5.0, "high_speed_reward": 1.0,
    "normalize_reward": True, "reward_speed_range": [10, 30],
}

def calculate_jerk(accelerations):
    """Calculates jerk (change in acceleration) magnitude."""
    if len(accelerations) < 2:
        return 0.0, 0.0 # Cannot calculate jerk with less than 2 points
    jerks = np.diff(accelerations)
    total_jerk_mag = np.sum(np.abs(jerks))
    peak_jerk_mag = np.max(np.abs(jerks)) if len(jerks) > 0 else 0.0
    return total_jerk_mag, peak_jerk_mag

def evaluate_single_environment(env_id, model_path, n_eval_episodes, seed, config):
    """
    Evaluates the model on a single environment type for n_eval_episodes.
    Designed to be run in a separate process.

    :param env_id: ID of the environment to evaluate.
    :param model_path: Path to the saved model (.zip).
    :param n_eval_episodes: Number of episodes to run.
    :param seed: Random seed for the environment.
    :param config: Environment configuration dictionary.
    :return: List of dictionaries, each containing metrics for one episode.
    """
    print(f"Process {os.getpid()} evaluating: {env_id}")
    # Create environment
    # Use Monitor to capture episode stats like reward, length, time
    try:
        eval_env = Monitor(gym.make(env_id, config=config))
    except Exception as e:
        print(f"Error creating env {env_id} in process {os.getpid()}: {e}")
        return []

    # Load model (determine device automatically)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = DQN.load(model_path, device=device)
    except Exception as e:
        print(f"Error loading model in process {os.getpid()}: {e}")
        eval_env.close()
        return []

    episode_results = []
    for episode in range(n_eval_episodes):
        obs, info = eval_env.reset(seed=seed + episode) # Vary seed per episode
        terminated = truncated = False
        step = 0
        total_reward = 0.0
        ep_speeds = []
        ep_accelerations = [] # Store acceleration values (vx changes)
        ep_steering_changes = [] # Store steering angle changes
        ep_on_road_steps = 0
        start_time = time.time()
        last_action = None # To calculate steering/acceleration changes

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            step += 1

            # Collect data for metrics (access underlying env if Monitor hides vehicle)
            vehicle = eval_env.unwrapped.vehicle
            ep_speeds.append(vehicle.speed)
            ep_on_road_steps += 1 if vehicle.on_road else 0

            # --- Calculate Jerk and Steering Change ---
            # Need access to the actual low-level action (steering, throttle)
            # This depends on how DiscreteMetaAction translates to low-level control
            # For simplicity, approximate acceleration change from speed change
            # and assume steering change is related to LANE_LEFT/RIGHT actions
            if step > 1:
                 accel = (ep_speeds[-1] - ep_speeds[-2]) / (1.0 / config.get("policy_frequency", 5)) # dt = 1/policy_freq
                 ep_accelerations.append(accel)

            # Steering change approximation (crude)
            current_discrete_action = action.item() # Get the integer action
            if last_action is not None:
                # Penalize lane change actions (assuming 0=LEFT, 2=RIGHT)
                if current_discrete_action in [0, 2] and last_action not in [0, 2]:
                    ep_steering_changes.append(1.0) # Arbitrary value for change
                elif current_discrete_action not in [0, 2] and last_action in [0, 2]:
                     ep_steering_changes.append(1.0) # Arbitrary value for change back
                else:
                    ep_steering_changes.append(0.0)
            last_action = current_discrete_action
            # ---

            if terminated or truncated:
                break

        # Episode finished, calculate metrics
        end_time = time.time()
        ep_duration = end_time - start_time
        ep_len = step # From Monitor info or step counter
        ep_collision = info.get("crashed", terminated and not truncated) # Check if termination was due to crash

        avg_speed = np.mean(ep_speeds) if ep_speeds else 0
        total_dist = avg_speed * (ep_len / config.get("policy_frequency", 5)) # Approx distance
        onlane_rate = ep_on_road_steps / ep_len if ep_len > 0 else 0
        total_steering = np.sum(ep_steering_changes) if ep_steering_changes else 0
        total_jerk, peak_jerk = calculate_jerk(ep_accelerations)

        episode_data = {
            "env_id": env_id,
            "episode": episode,
            "reward": total_reward,
            "length": ep_len,
            "duration_sec": ep_duration,
            "avg_speed_mps": avg_speed,
            "total_distance_m": total_dist, # Approximate
            "onlane_rate": onlane_rate,
            "collision": ep_collision,
            "total_steering_change": total_steering, # Approximation
            "total_jerk": total_jerk,
            "peak_jerk": peak_jerk,
        }
        episode_results.append(episode_data)
        # print(f"  Env: {env_id}, Ep: {episode+1}/{n_eval_episodes}, Reward: {total_reward:.2f}, Len: {ep_len}, Collision: {ep_collision}")

    eval_env.close()
    print(f"Process {os.getpid()} finished evaluating: {env_id}")
    return episode_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DQN on HighwayEnv environments")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model (.zip file)")
    parser.add_argument("--env-ids", nargs='+', default=['highway-v0', 'merge-v0', 'roundabout-v0', 'custom-kinematic-v0'],
                        help="List of environment IDs to evaluate on")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of episodes per environment")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed for evaluation")
    parser.add_argument("--n-workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of parallel workers (processes)")
    parser.add_argument("--output-csv", type=str, default="./evaluation_results.csv",
                        help="Path to save the aggregated results CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Evaluating model: {args.model_path}")
    print(f"Environments: {args.env_ids}")
    print(f"Episodes per env: {args.n_eval_episodes}")
    print(f"Number of workers: {args.n_workers}")

    start_time = time.time()

    # Prepare arguments for parallel execution
    # Use partial to fix the arguments that are the same for all calls
    eval_func_partial = partial(
        evaluate_single_environment,
        model_path=args.model_path,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
        config=common_config # Pass the common config
    )

    all_results = []
    # Use multiprocessing Pool for parallelism
    with Pool(processes=args.n_workers) as pool:
        # Map the evaluation function across the environment IDs
        results_list = pool.map(eval_func_partial, args.env_ids)
        # Flatten the list of lists into a single list of results
        for res in results_list:
            all_results.extend(res)

    total_time = time.time() - start_time
    print(f"\nEvaluation finished in {total_time:.2f} seconds.")

    if not all_results:
        print("No evaluation results collected.")
        return

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate aggregated statistics per environment
    agg_metrics = results_df.groupby("env_id").agg(
        mean_reward=('reward', 'mean'),
        std_reward=('reward', 'std'),
        mean_length=('length', 'mean'),
        mean_duration=('duration_sec', 'mean'),
        mean_avg_speed=('avg_speed_mps', 'mean'),
        mean_distance=('total_distance_m', 'mean'),
        mean_onlane_rate=('onlane_rate', 'mean'),
        collision_rate=('collision', lambda x: x.sum() / len(x)), # Proportion of collisions
        mean_total_steering=('total_steering_change', 'mean'),
        mean_total_jerk=('total_jerk', 'mean'),
        mean_peak_jerk=('peak_jerk', 'mean'),
        n_episodes=('episode', 'count') # Should match n_eval_episodes
    ).reset_index()

    print("\nAggregated Results per Environment:")
    print(agg_metrics.to_string())

    # Save detailed and aggregated results
    detailed_csv_path = args.output_csv.replace(".csv", "_detailed.csv")
    agg_csv_path = args.output_csv
    print(f"\nSaving detailed results to: {detailed_csv_path}")
    results_df.to_csv(detailed_csv_path, index=False)
    print(f"Saving aggregated results to: {agg_csv_path}")
    agg_metrics.to_csv(agg_csv_path, index=False)

if __name__ == "__main__":
    main()
