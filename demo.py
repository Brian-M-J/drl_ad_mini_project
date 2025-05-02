# demo.py
import gymnasium as gym
import torch
import argparse
import time
import numpy as np
import os
import sys

# Import custom environment - ensures registration
import custom_env

from stable_baselines3 import DQN

# --- Colab Rendering Setup (Comment out if running locally) ---
# IN_COLAB = 'google.colab' in sys.modules
# if IN_COLAB:
#     print("Running in Colab - Setting up virtual display...")
#     !pip install pyvirtualdisplay -q
#     !apt-get install -y xvfb python-opengl ffmpeg -q
#     from pyvirtualdisplay import Display
#     # Start virtual display
#     virtual_display = Display(visible=0, size=(800, 800))
#     virtual_display.start()
#     # Suppress pygame welcome message
#     os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# else:
#     print("Not running in Colab.")
# --- End Colab Setup ---


# Define the common config here, MUST match training config
common_config = {
    "observation": {
        "type": "Kinematics", "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {"x": [-100, 100], "y": [-15, 15], "vx": [-30, 30], "vy": [-30, 30]},
        "absolute": False, "normalize": True, "flatten": True, "see_behind": True
    },
    "action": {"type": "DiscreteMetaAction", "target_speeds": np.linspace(0, 30, 5)},
    "duration": 60, # Longer duration for demo
    # Add other relevant config keys if they affect behavior/rendering
    "screen_width": 800,
    "screen_height": 800,
}


def main():
    parser = argparse.ArgumentParser(description="Run demo of trained DQN agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model (.zip file)")
    parser.add_argument("--env-id", type=str, default="highway-v0", help="Environment ID to run demo on")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for the environment")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to demonstrate")
    parser.add_argument("--log-folder", type=str, default="./drl_demo_logs/",
                        help="Folder to save demo logs (observations, actions)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # --- Setup Log Path ---
    os.makedirs(args.log_folder, exist_ok=True)
    log_filename = os.path.join(args.log_folder, f"demo_{args.env_id}_{int(time.time())}.csv")
    log_data = []
    print(f"Logging demo data to: {log_filename}")

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from: {args.model_path}")
    try:
        model = DQN.load(args.model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Create Environment ---
    render_mode = 'human' # Default for local execution
    # --- Uncomment for Colab Video Recording ---
    # if IN_COLAB:
    #     from gymnasium.wrappers.record_video import RecordVideo
    #     render_mode = 'rgb_array' # Necessary for video recording
    #     print("Setting render_mode to 'rgb_array' for Colab video.")
    # ---

    try:
        # Pass the *exact same config* used during training
        env = gym.make(args.env_id, config=common_config, render_mode=render_mode)
        print(f"Environment '{args.env_id}' created.")
    except Exception as e:
        print(f"Error creating environment '{args.env_id}': {e}")
        return

    # --- Colab Video Wrapper (Uncomment if needed) ---
    # if IN_COLAB:
    #     video_folder = os.path.join(args.log_folder, "videos")
    #     env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    #     print(f"Recording videos to: {video_folder}")
    # ---

    # --- Run Demo Loop ---
    for episode in range(args.num_episodes):
        print(f"\n--- Starting Episode {episode + 1}/{args.num_episodes} ---")
        seed_to_use = args.seed + episode if args.seed is not None else None
        obs, info = env.reset(seed=seed_to_use)
        terminated = truncated = False
        step = 0
        ep_reward = 0

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            action_int = action.item() # Get integer action for logging

            # Store log data BEFORE stepping
            log_entry = {
                "episode": episode, "step": step,
                "action": action_int, "reward": 0, # Reward comes after step
                "terminated": False, "truncated": False,
                "crashed": False, "on_road": False,
                "speed": 0.0
            }
            # Add observation features to log (flattened)
            for i, val in enumerate(obs):
                 log_entry[f"obs_{i}"] = val

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            # Update log entry with results from the step
            log_entry["reward"] = reward
            log_entry["terminated"] = terminated
            log_entry["truncated"] = truncated
            if isinstance(env.unwrapped, gym.Wrapper): # Handle Monitor wrapper etc.
                 vehicle = env.unwrapped.vehicle
            else:
                 vehicle = env.vehicle
            log_entry["crashed"] = getattr(vehicle, 'crashed', False)
            log_entry["on_road"] = getattr(vehicle, 'on_road', False)
            log_entry["speed"] = getattr(vehicle, 'speed', 0.0)

            log_data.append(log_entry)

            # Render (handled by RecordVideo wrapper in Colab if enabled)
            if not ('google.colab' in sys.modules and 'RecordVideo' in str(type(env))):
                 try:
                     env.render()
                 except Exception as e:
                     print(f"Error during rendering: {e}")
                     # break # Optional: stop demo if rendering fails

            # Optional: Add a small delay for better visualization locally
            # time.sleep(0.05)

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step} steps. Reward: {ep_reward:.2f}")
                print(f"Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
                break # Exit inner loop

    # --- Cleanup ---
    env.close()
    print("\nDemo finished.")

    # --- Save Log Data ---
    if log_data:
        import pandas as pd
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(log_filename, index=False)
        print(f"Demo log saved to {log_filename}")

if __name__ == "__main__":
    main()

