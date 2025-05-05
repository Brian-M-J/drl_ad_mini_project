# Autonomous Driving using Deep Reinforcement Learning

## Overview

This project implements and trains a Deep Reinforcement Learning (DRL) agent to perform autonomous driving tasks in various simulated environments. It utilizes the `highway-env` simulator and the Stable Baselines3 library to train a single Dueling Deep Q-Network (DQN) model with a custom Multi-Layer Perceptron (MLP) policy across multiple driving scenarios. The project includes scripts for training, evaluation, demonstration, and model interpretation using SHAP.

## Features

* **DRL Agent:** Implements a Dueling DQN agent using Stable Baselines3.
* **Custom Policy:** Defines a custom Dueling MLP policy network (`custom_policy.py`).
* **Multi-Environment Training:** Trains a single agent concurrently on multiple `highway-env` scenarios using vectorized environments (`SubprocVecEnv`).
* **Custom Environment:** Includes a custom circular track environment (`custom_env.py`).
* **Kinematic Observations:** Uses low-dimensional kinematic state information as input to the agent.
* **Parallelism:** Leverages multiprocessing for faster training and evaluation.
* **Training Management:** Supports checkpointing for interrupting and resuming training.
* **Evaluation Framework:** Provides scripts for quantitative evaluation of the trained agent using standard metrics (speed, collision rate, jerk, etc.).
* **Demonstration:** Allows visual demonstration of the trained agent's behavior in selected environments.
* **Interpretability:** Includes tools (`interpret.py`) using SHAP to explain the agent's decisions based on feature importance.
* **Cross-Platform:** Designed to run on both local machines (Linux, macOS, Windows) and Google Colab, supporting CPU and GPU.

## Environments Trained On

The agent is configured to train on the following `highway-env` environments simultaneously:

* `highway-v0`
* `merge-v0`
* `roundabout-v0`
* `intersection-v0`
* `exit-v0`
* `custom-kinematic-v0` (Custom circular environment defined in `custom_env.py`)

*(Note: You can modify the environment list in `train.py` and `evaluate.py`)*

## Technology Stack

* Python 3.9+
* PyTorch
* Stable Baselines3 (`stable-baselines3[extra]`)
* Gymnasium (formerly OpenAI Gym)
* highway-env
* SHAP
* NumPy
* Pandas
* Matplotlib / Seaborn (for plotting, mainly in interpretation)
* (Optional for Colab Rendering): `pyvirtualdisplay`, `python-opengl`, `ffmpeg`

## File Structure

.├── custom_env.py             # Defines the custom circular track environment├── custom_policy.py          # Defines the custom Dueling MLP policy network├── train.py                  # Script for training the DRL agent├── evaluate.py               # Script for evaluating the trained agent├── demo.py                   # Script for running visual demonstrations├── interpret.py              # Script for interpreting the model using SHAP├── drl_logs/                 # Default folder for saving models and evaluation logs├── drl_tensorboard/          # Default folder for TensorBoard training logs├── drl_demo_logs/            # Default folder for demo step logs├── drl_interpret_output/     # Default folder for SHAP plot outputs└── README.md                 # This file
## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate (Linux/macOS)
    source .venv/bin/activate
    # Activate (Windows)
    .\.venv\Scripts\activate
    ```
    *Alternatively, use Conda or Pixi.*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is not provided, install manually:*
    ```bash
    pip install torch highway-env gymnasium "stable-baselines3[extra]" shap numpy pandas matplotlib seaborn
    ```
    * **PyTorch:** Ensure you install the correct PyTorch version for your system (CPU or specific CUDA version). See [PyTorch installation instructions](https://pytorch.org/).
    * **Windows:** `SubprocVecEnv` requires the code using it to be inside an `if __name__ == "__main__":` block (which is done in the scripts).

4.  **(Optional) Google Colab:** Upload the `.py` files or mount Google Drive. Install dependencies using `!pip install ...` in a code cell. Remember to handle paths for saving/loading models and logs (e.g., mount Drive).

## Usage

### 1. Training

* Run the training script. Adjust parameters as needed (e.g., number of environments, total steps).
    ```bash
    python train.py --total-timesteps 500000 --n-envs 4 --policy DuelingMlpPolicy
    ```
* **Key Arguments:**
    * `--total-timesteps`: Total training steps.
    * `--n-envs`: Number of parallel environments (adjust based on CPU cores).
    * `--policy`: Use `MlpPolicy` (SB3 default) or `DuelingMlpPolicy` (custom).
    * `--env-ids`: List of environment IDs to train on.
    * `--log-folder`: Where to save models and evaluation results during training.
    * `--tensorboard-log`: Where to save TensorBoard logs.
    * `--load-checkpoint`: Path to a model zip file (without `.zip`) to resume training.
* **Monitoring:** Use TensorBoard to monitor training progress:
    ```bash
    tensorboard --logdir ./drl_tensorboard/
    ```
    *(Navigate to `http://localhost:6006` in your browser)*

### 2. Evaluation

* Evaluate a trained model saved during training (e.g., the final model or the best model).
    ```bash
    python evaluate.py --model-path ./drl_logs/YOUR_RUN_NAME/models/dqn_highway_final.zip --n-eval-episodes 50 --n-workers 4
    ```
* **Key Arguments:**
    * `--model-path`: Path to the saved `.zip` model file.
    * `--n-eval-episodes`: Number of episodes to run per environment for evaluation.
    * `--n-workers`: Number of parallel processes for evaluation.
    * `--env-ids`: List of environment IDs to evaluate on (defaults match training).
    * `--output-csv`: Path to save the aggregated evaluation results. A detailed CSV is also saved.

### 3. Demonstration

* Run a visual demo of the trained agent in a specific environment.
    ```bash
    python demo.py --model-path ./drl_logs/YOUR_RUN_NAME/models/dqn_highway_final.zip --env-id highway-v0
    ```
* **Key Arguments:**
    * `--model-path`: Path to the saved `.zip` model file.
    * `--env-id`: The specific environment ID to run the demo in.
    * `--num-episodes`: How many episodes to run.
    * `--log-folder`: Where to save detailed step-by-step logs (including observations and Q-values).
* **(Colab):** Uncomment the virtual display and video recording setup in `demo.py` if needed.

### 4. Interpretation (SHAP)

* Analyze the decisions of a trained model using SHAP.
    ```bash
    python interpret.py --model-path ./drl_logs/YOUR_RUN_NAME/models/dqn_highway_final.zip --output-folder ./interpret_output/
    ```
* **Key Arguments:**
    * `--model-path`: Path to the saved `.zip` model file.
    * `--n-background-samples`: Number of samples to use for initializing the SHAP explainer.
    * `--n-explain-samples`: Number of agent decisions to collect and explain.
    * `--output-folder`: Where to save the generated SHAP plots (summary, force, dependence).

## Configuration

* **Environment Parameters:** Shared environment configurations (observation type, action type, common rewards) are defined in the `common_config` dictionary within `train.py`, `evaluate.py`, `demo.py`, and `interpret.py`. Ensure these are consistent. Specific parameters for the custom environment are in `custom_env.py`.
* **Model Architecture:** The Dueling MLP architecture (layer sizes, feature dimensions) is defined in `custom_policy.py`.
* **Hyperparameters:** Training hyperparameters (learning rate, buffer size, exploration, etc.) can be adjusted via command-line arguments in `train.py`.

## Logging and Results

* **Training:**
    * TensorBoard logs are saved to the directory specified by `--tensorboard-log`.
    * Model checkpoints and the best/final models (`.zip`) are saved within subdirectories under `--log-folder`.
    * `EvalCallback` results (`evaluations.npz`) are saved within the log folder.
* **Evaluation:**
    * Aggregated results are saved to the CSV file specified by `--output-csv`.
    * Detailed per-episode results are saved to `*_detailed.csv` in the same location.
* **Demonstration:**
    * Step-by-step logs including observations and Q-values are saved to CSV files in the directory specified by `--log-folder`.
* **Interpretation:**
    * SHAP plots are saved as image files in the directory specified by `--output-folder`.

