Kung Fu Master AI Training with A3C and PyTorch
Overview
This repository contains a reinforcement learning model that uses Asynchronous Advantage Actor-Critic (A3C) to train an AI agent to play the "KungFuMaster" game from the Atari Learning Environment (ALE). The training utilizes deep reinforcement learning techniques implemented in PyTorch and the gymnasium library. The model incorporates convolutional neural networks (CNN) for processing game frames and learning effective policies through both actor and critic networks. Additionally, the environment is processed and standardized using OpenAI Gym's observation wrappers.

Features
A3C Algorithm: Implements the Asynchronous Advantage Actor-Critic model for training an agent.
Atari Environment: Utilizes the KungFuMaster game from ALE as the training environment.
Deep Q-Learning (DQN) Integration: Features a convolutional neural network (CNN) for frame-based state representation.
Multi-Environment Training: Supports training in parallel across multiple environments to speed up the learning process.
Video Recording: The model can record gameplay video after training, allowing for visual verification of the agent's performance.
PyTorch-based Neural Network: Utilizes PyTorch for building and optimizing the neural network.
Setup Instructions
Prerequisites
Before running this code, make sure you have the following dependencies installed:

Python 3.11+ (Recommended)
PyTorch: For deep learning operations
Gymnasium: For interacting with the training environment
OpenCV: For image processing and frame resizing
ale-py: For Atari Learning Environment support
tqdm: For displaying progress during training
You can install the required packages using pip:

pip install torch gymnasium ale-py opencv-python tqdm
Hardware Requirements
GPU (NVIDIA CUDA support is recommended) for faster model training and evaluation.
CPU can be used, but training may be slower.
Environment Setup
To use the provided model, simply clone the repository and run the script. The script handles the creation of the KungFuMaster environment, training the agent, and recording gameplay videos.


git clone <repo-url>
cd <repo-directory>
python train_agent.py
Code Explanation
Classes and Functions
Network(nn.Module)
This class defines the neural network architecture used by the agent. It consists of:

Convolutional Layers: To process the raw pixel data from the Atari game.
Fully Connected Layers: To generate action values (actor network) and a state value (critic network).
Forward Pass: The network takes a state, passes it through convolutional layers, and then outputs action values and state value.
PreprocessAtari(ObservationWrapper)
A custom gym wrapper to preprocess the Atari environment's frames. This wrapper resizes and optionally converts frames to grayscale. It also stacks multiple frames together to capture temporal information about the environment.

reset(): Resets the environment and prepares the frame stack.
observation(): Processes each frame before passing it to the agent.
Agent
The agent class encapsulates the training and decision-making logic. It includes:

Actor-Critic Model: The agent maintains a network with both an actor and a critic to estimate action values and state values.
act() Method: This method returns the action the agent should take based on the current state. It uses a softmax policy.
step() Method: This method updates the agent's network by computing the loss and performing backpropagation.
EnvBatch
A class to manage multiple environments in parallel. It is used to speed up the training process by interacting with multiple environments at once. This class handles resetting the environments and collecting data from them in batches.

evaluate(agent, env)
This function evaluates the agent's performance in the environment by running multiple episodes and calculating the total reward.

show_video_of_model(agent, env)
After training, this function records a video of the agent interacting with the environment and stores it in a specified directory. It then displays the video in the notebook.

show_video()
This function displays a previously recorded video of the agent playing the game in a Jupyter notebook.

Training Loop
The model is trained using A3C, where the agent interacts with 10 parallel environments. The training loop runs for 3000 iterations and periodically prints the average reward. The agent is trained using batches of observations and rewards, updating the network using backpropagation.


with tqdm.trange(0, 3001) as progress_bar:
  for i in progress_bar:
    batch_actions = agent.act(state=batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(actions=batch_actions)
    
    batch_rewards *= 0.01  # Stabilizing training

    agent.step(state=batch_states, action=batch_actions, reward=batch_rewards, next_state=batch_next_states, done=batch_dones)
    batch_states = batch_next_states

    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent=agent, env=env, n_episodes=30)))
Hyperparameters
Learning Rate: 1e-4
Discount Factor: 0.99 (The agent prioritizes long-term rewards)
Number of Environments: 10 (Parallel environments for faster training)
Result Evaluation
After training, the agent's performance is evaluated by running a set of episodes and calculating the average reward.

Visualization
To visualize the trained agent's performance, a gameplay video is recorded and displayed.

show_video_of_model(agent, env)
You can view the recorded video in the output folder (KungFuVideo/).

Contributions
Feel free to fork this repository, open issues, or submit pull requests to improve the model or the environment!