# interpret.py
import gymnasium as gym
import torch
import argparse
import os
import numpy as np
import pandas as pd
import shap # Make sure shap is installed: pip install shap
import matplotlib.pyplot as plt

# Import custom environment - ensures registration
import custom_env

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the common config here, MUST match training config
common_config = {
    "observation": {
        "type": "Kinematics", "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {"x": [-100, 100], "y": [-15, 15], "vx": [-30, 30], "vy": [-30, 30]},
        "absolute": False, "normalize": True, "flatten": True, "see_behind": True
    },
    "action": {"type": "DiscreteMetaAction", "target_speeds": np.linspace(0, 30, 5)},
    "duration": 40,
}

# Define feature names for SHAP plots (MUST match the order in common_config["features"])
# Example for vehicles_count = 5 (Ego + 4 others) -> 5 vehicles * 7 features = 35 features
N_VEHICLES = common_config["observation"]["vehicles_count"]
FEATURES_PER_VEHICLE = len(common_config["observation"]["features"])
FEATURE_NAMES = []
for i in range(N_VEHICLES):
    prefix = f"v{i}_" if i > 0 else "ego_"
    for feature_name in common_config["observation"]["features"]:
        FEATURE_NAMES.append(prefix + feature_name)

print(f"Generated {len(FEATURE_NAMES)} feature names for SHAP.")
# print(FEATURE_NAMES) # Uncomment to verify names

def get_background_data(env_id, config, n_samples=100):
    """Collects sample observations to serve as background data for SHAP."""
    print(f"Collecting {n_samples} background samples from {env_id}...")
    env = DummyVecEnv([lambda: gym.make(env_id, config=config)])
    obs = env.reset()
    background = []
    for _ in range(n_samples):
        action = env.action_space.sample() # Random actions
        obs, _, _, _ = env.step(action)
        background.append(obs[0]) # Append the single observation from DummyVecEnv
    env.close()
    return np.array(background)


def main():
    parser = argparse.ArgumentParser(description="Interpret trained DQN agent using SHAP")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model (.zip file)")
    parser.add_argument("--env-id", type=str, default="highway-v0",
                        help="Environment ID to use for collecting samples and context")
    parser.add_argument("--n-background-samples", type=int, default=200,
                        help="Number of background samples for SHAP explainer")
    parser.add_argument("--n-explain-samples", type=int, default=50,
                        help="Number of samples/decisions to explain")
    parser.add_argument("--output-folder", type=str, default="./drl_interpret_output/",
                        help="Folder to save SHAP plots")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from: {args.model_path}")
    try:
        # Load the model. Note: We don't need the env for interpretation itself,
        # but we might need it to get sample data.
        model = DQN.load(args.model_path, device=device)
        # Access the Q-network (adjust if using a custom policy structure)
        q_network = model.policy.q_net
        q_network.eval() # Set to evaluation mode
        print("Model and Q-network loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Get Background Data ---
    background_data = get_background_data(args.env_id, common_config, args.n_background_samples)
    # Convert background data to tensor on the correct device
    background_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)
    print(f"Background data shape: {background_tensor.shape}")

    # --- Create SHAP Explainer ---
    # For DQN, we explain the Q-value outputs of the network.
    # DeepExplainer is suitable for PyTorch models.
    print("Creating SHAP DeepExplainer...")
    # The explainer needs the model (or function) and background data
    explainer = shap.DeepExplainer(q_network, background_tensor)
    print("Explainer created.")

    # --- Get Samples to Explain ---
    # Collect some new samples by running the agent for a few steps
    print(f"Collecting {args.n_explain_samples} samples to explain from {args.env_id}...")
    explain_env = DummyVecEnv([lambda: gym.make(args.env_id, config=common_config)])
    obs = explain_env.reset()
    explain_samples = []
    actions_taken = []
    for _ in range(args.n_explain_samples):
        action, _ = model.predict(obs, deterministic=True)
        explain_samples.append(obs[0]) # Store the observation
        actions_taken.append(action.item()) # Store the action taken
        obs, _, terminated, truncated = explain_env.step(action)
        if terminated or truncated:
            obs = explain_env.reset() # Reset if episode ends
    explain_env.close()
    explain_tensor = torch.tensor(np.array(explain_samples), dtype=torch.float32).to(device)
    print(f"Samples to explain shape: {explain_tensor.shape}")

    # --- Calculate SHAP Values ---
    print("Calculating SHAP values...")
    # shap_values will be a list (one element per output neuron/action)
    # Each element will have shape (n_explain_samples, n_features)
    shap_values = explainer.shap_values(explain_tensor)
    print(f"SHAP values calculated. Type: {type(shap_values)}, Len: {len(shap_values)}")
    if isinstance(shap_values, list) and len(shap_values) > 0:
        print(f"Shape of SHAP values for one action: {shap_values[0].shape}")
    else:
         print(f"Shape of SHAP values: {shap_values.shape}") # If not a list (e.g., single output)


    # --- Generate SHAP Plots ---

    # 1. Summary Plot (Feature Importance)
    # We need to decide which action's SHAP values to summarize, or combine them.
    # Option A: Summarize for the action taken most often (or a specific action index)
    # Option B: Summarize the average absolute SHAP value across all actions.
    # Let's do Option B for general importance:
    try:
        print("\nGenerating SHAP Summary Plot (Average Absolute)...")
        # Calculate mean absolute SHAP values across all action outputs
        mean_abs_shap = np.mean([np.abs(s) for s in shap_values], axis=0)
        # Create a SHAP values object for plotting
        shap_values_for_plot = shap.Explanation(
            values=mean_abs_shap,
            base_values=None, # Not applicable for summary
            data=explain_tensor.cpu().numpy(), # Use the data points explained
            feature_names=FEATURE_NAMES
        )

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_for_plot, plot_type="bar", show=False)
        plt.title("Mean Absolute SHAP Value (Across All Actions)")
        plt.tight_layout()
        summary_plot_path = os.path.join(args.output_folder, "shap_summary_bar.png")
        plt.savefig(summary_plot_path)
        plt.close()
        print(f"Summary plot saved to {summary_plot_path}")

        # Dot summary plot
        plt.figure(figsize=(10, 8))
        # Use shap_values for a specific action (e.g., action 1: IDLE or FASTER) for dot plot
        action_to_plot = 1 # Example: Choose an action index
        if action_to_plot < len(shap_values):
             shap.summary_plot(shap_values[action_to_plot], features=explain_tensor.cpu().numpy(), feature_names=FEATURE_NAMES, show=False)
             plt.title(f"SHAP Dot Summary Plot (for Action {action_to_plot})")
             plt.tight_layout()
             dot_plot_path = os.path.join(args.output_folder, f"shap_summary_dot_action_{action_to_plot}.png")
             plt.savefig(dot_plot_path)
             plt.close()
             print(f"Dot summary plot saved to {dot_plot_path}")
        else:
             print(f"Action index {action_to_plot} out of range for SHAP values.")


    except Exception as e:
        print(f"Error generating summary plot: {e}")


    # 2. Force Plot (Explanation for a single prediction)
    # Requires JS, might not work well outside notebooks directly.
    # We can plot for one sample and one action output.
    try:
        sample_idx = 0 # Explain the first sample
        action_idx = actions_taken[sample_idx] # Explain the action actually taken
        print(f"\nGenerating SHAP Force Plot for sample {sample_idx}, action {action_idx}...")

        # Need base value (expected output) for the specific action
        # The explainer might store this, or we approximate it
        # explainer.expected_value seems to be for the sum of outputs in some versions.
        # Let's try getting the base value for the specific action output
        base_value_action = explainer.expected_value[action_idx] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value


        force_plot = shap.force_plot(
            base_value_action,
            shap_values[action_idx][sample_idx,:],
            features=explain_tensor[sample_idx,:].cpu().numpy(),
            feature_names=FEATURE_NAMES,
            matplotlib=True, # Use matplotlib backend
            show=False
        )
        force_plot_path = os.path.join(args.output_folder, f"shap_force_plot_sample_{sample_idx}_action_{action_idx}.png")
        plt.savefig(force_plot_path, bbox_inches='tight') # Save the matplotlib figure
        plt.close()
        print(f"Force plot saved to {force_plot_path}")

    except Exception as e:
        print(f"Error generating force plot: {e}")


    # 3. Dependence Plots (How a feature's value affects its SHAP value)
    try:
        feature_to_plot = "v1_x" # Example: Distance to nearest vehicle
        if feature_to_plot in FEATURE_NAMES:
            print(f"\nGenerating SHAP Dependence Plot for feature '{feature_to_plot}'...")
            plt.figure()
            # Plot dependence for a specific action's SHAP values
            action_idx_dep = 1 # Example action index
            shap.dependence_plot(
                feature_to_plot,
                shap_values[action_idx_dep],
                features=explain_tensor.cpu().numpy(),
                feature_names=FEATURE_NAMES,
                interaction_index="auto", # Color by interacting feature
                show=False
            )
            plt.title(f"SHAP Dependence Plot (Feature: {feature_to_plot}, Action: {action_idx_dep})")
            plt.tight_layout()
            dep_plot_path = os.path.join(args.output_folder, f"shap_dependence_{feature_to_plot}_action_{action_idx_dep}.png")
            plt.savefig(dep_plot_path)
            plt.close()
            print(f"Dependence plot saved to {dep_plot_path}")
        else:
            print(f"Feature '{feature_to_plot}' not found in FEATURE_NAMES.")

    except Exception as e:
        print(f"Error generating dependence plot: {e}")


    print("\nInterpretation script finished.")


if __name__ == "__main__":
    main()
