# train.py
import os
import argparse
import time
import gymnasium as gym
import torch
import numpy as np

# Import custom environment - make sure custom_env.py is in the Python path
# This will also run the registration code inside custom_env.py
import custom_env

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import custom policy (optional, comment out if using 'MlpPolicy')
# from custom_policy import DuelingMlpPolicy

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN on HighwayEnv environments")
    parser.add_argument("--env-ids", nargs='+', default=['highway-v0', 'merge-v0', 'roundabout-v0', 'custom-kinematic-v0'],
                        help="List of environment IDs to train on")
    parser.add_argument("--custom-env-id", type=str, default="custom-kinematic-v0",
                        help="ID of the custom environment to use")
    parser.add_argument("--policy", type=str, default="MlpPolicy", choices=["MlpPolicy", "DuelingMlpPolicy"],
                        help="Policy architecture ('MlpPolicy' or 'DuelingMlpPolicy' from custom_policy.py)")
    parser.add_argument("--total-timesteps", type=int, default=200_000,
                        help="Total number of training timesteps")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=50_000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, # Increased batch size for stability
                        help="Batch size for training")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="How many steps of random actions before learning starts")
    parser.add_argument("--gamma", type=float, default=0.95, # Discount factor
                        help="Discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0, # Target network update rate
                        help="Soft update coefficient (tau)")
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Update the model every train_freq steps")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="How many gradient steps to perform at each update")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
                        help="Fraction of entire training period over which the exploration rate is reduced")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05,
                        help="Final value of random action probability")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (adjust based on CPU cores)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-folder", type=str, default="./drl_logs/",
                        help="Folder to save logs and models")
    parser.add_argument("--tensorboard-log", type=str, default="./drl_tensorboard/",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                        help="Save model checkpoint every N steps")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Evaluate model every N steps")
    parser.add_argument("--vec-env-type", type=str, default="subproc", choices=["subproc", "dummy"],
                        help="Vectorized environment type: 'subproc' for parallel, 'dummy' for sequential")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load and continue training")
    return parser.parse_args()

def main():
    """Main training loop."""
    args = get_args()

    # --- Setup Paths ---
    run_name = f"DQN_{args.policy}_MultiEnv_{int(time.time())}"
    log_path = os.path.join(args.log_folder, run_name)
    model_save_path = os.path.join(log_path, "models")
    tensorboard_log_path = os.path.join(args.tensorboard_log, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)

    print(f"Run Name: {run_name}")
    print(f"Log Path: {log_path}")
    print(f"TensorBoard Log Path: {tensorboard_log_path}")
    print(f"Using Environments: {args.env_ids}")

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment Configuration ---
    # IMPORTANT: Ensure this config matches EXACTLY across all environments
    # used for training, evaluation, and demo.
    common_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {"x": [-100, 100], "y": [-15, 15], "vx": [-30, 30], "vy": [-30, 30]},
            "absolute": False,
            "normalize": True,
            "flatten": True,
            "see_behind": True
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(0, 30, 5) # Consistent target speeds
        },
        # Add other common config keys if needed, e.g., duration, reward weights
        "duration": 40,
        "collision_reward": -5.0,
        "high_speed_reward": 1.0,
        "normalize_reward": True,
        "reward_speed_range": [10, 30],
    }

    # --- Vectorized Environment ---
    vec_env_cls = SubprocVecEnv if args.vec_env_type == "subproc" else DummyVecEnv
    print(f"Using VecEnv type: {args.vec_env_type}")

    # Create a list of functions, each returning a configured & monitored env
    env_fns = []
    for env_id in args.env_ids:
        # Create a specific config if needed (e.g., different rewards for one env)
        # Otherwise, use common_config
        current_config = common_config.copy()
        # Example: Modify config slightly for a specific env if necessary
        # if env_id == 'merge-v0':
        #     current_config['reward_speed_range'] = [5, 20]

        # Lambda function captures the current env_id and config
        env_fns.append(lambda id=env_id, cfg=current_config: Monitor(gym.make(id, config=cfg)))

    # Create the vectorized environment
    vec_env = vec_env_cls(env_fns, start_method='fork') # 'fork' often needed for SubprocVecEnv

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1), # Adjust freq for vec env
        save_path=model_save_path,
        name_prefix="dqn_highway",
        save_replay_buffer=True,
        save_vecnormalize=True, # Save normalization stats if VecNormalize wrapper is used
    )

    # Evaluation environment (use the first env type for simplicity in eval callback)
    # For more rigorous evaluation across all types, use the evaluate.py script
    eval_env_id = args.env_ids[0]
    eval_env = Monitor(gym.make(eval_env_id, config=common_config))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=os.path.join(log_path, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1), # Adjust freq
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # --- Model Definition ---
    policy_choice = args.policy
    # if policy_choice == "DuelingMlpPolicy":
    #     policy_choice = DuelingMlpPolicy # Use the imported custom policy class

    model_params = {
        "policy": policy_choice,
        "env": vec_env,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "gamma": args.gamma,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "target_update_interval": 1000, # How often to update target network (steps)
        "seed": args.seed,
        "device": device,
        "verbose": 1,
        "tensorboard_log": args.tensorboard_log,
    }

    # --- Load or Create Model ---
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading model from checkpoint: {args.load_checkpoint}")
        # Ensure the .zip file exists
        if not args.load_checkpoint.endswith('.zip'):
             checkpoint_zip = args.load_checkpoint + ".zip"
        else:
             checkpoint_zip = args.load_checkpoint

        if os.path.exists(checkpoint_zip):
            model = DQN.load(
                checkpoint_zip,
                env=vec_env,
                device=device,
                # Pass parameters that might have changed or are needed for continuing
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                seed=args.seed,
                # custom_objects={"policy_class": DuelingMlpPolicy} # Needed if loading custom policy
            )
            print("Model loaded successfully.")
            # Reset timesteps to False if you want TB logging to continue from where it left off
            reset_num_timesteps = False
            # Load replay buffer if saved with the checkpoint
            if os.path.exists(args.load_checkpoint + "_replay_buffer.pkl"):
                 print("Loading replay buffer...")
                 model.load_replay_buffer(args.load_checkpoint + "_replay_buffer.pkl")
            else:
                 print("Replay buffer not found, starting fresh.")

        else:
            print(f"Error: Checkpoint zip file not found at {checkpoint_zip}")
            print("Creating a new model.")
            model = DQN(**model_params)
            reset_num_timesteps = True

    else:
        print("No checkpoint specified or found. Creating a new model.")
        model = DQN(**model_params)
        reset_num_timesteps = True


    # --- Training ---
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback], # List of callbacks
            log_interval=10, # Log TensorBoard scalars every 10 episodes
            tb_log_name=run_name, # Use the unique run name for TensorBoard logging
            reset_num_timesteps=reset_num_timesteps, # Continue timestep count if loading
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # --- Save Final Model ---
        final_model_path = os.path.join(model_save_path, "dqn_highway_final.zip")
        print(f"Saving final model to: {final_model_path}")
        model.save(final_model_path)
        # Optionally save replay buffer with the final model
        # model.save_replay_buffer(os.path.join(model_save_path, "dqn_highway_final_replay_buffer.pkl"))

        print("Closing environments...")
        vec_env.close()
        eval_env.close()
        print("Training finished.")

if __name__ == "__main__":
    main()
