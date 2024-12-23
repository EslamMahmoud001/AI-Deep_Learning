Pac-Man AI with Deep Convolutional Q-Learning

This project implements a Deep Convolutional Q-Learning (DQN) agent to play the classic Pac-Man game. By leveraging deep neural networks and reinforcement learning techniques, the agent learns to navigate the game environment efficiently, maximizing its score through trial and error.

Table of Contents

Introduction

Key Features

Project Structure

Requirements

Setup

How It Works

Training Details

Results

Usage

Contributions

License

Introduction

Deep Convolutional Q-Learning combines reinforcement learning with convolutional neural networks to solve complex decision-making tasks. In this project, the agent observes raw game frames, preprocesses them, and trains on the Q-learning principle to improve its performance over time. This approach allows the agent to make informed decisions based on visual cues and delayed rewards.

Key Features

Deep Convolutional Neural Network (CNN): Extracts meaningful features from game frames.

Replay Buffer: Stores past experiences to improve sample efficiency and reduce correlations between updates.

Target Network: Stabilizes learning by updating target Q-values at regular intervals.

Epsilon-Greedy Policy: Balances exploration of new actions and exploitation of learned strategies.

Video Recording: Captures gameplay for evaluation and visualization.

Dynamic Learning Rate and Decay: Adjusts the exploration strategy over time to focus on exploitation.

Project Structure

Pacman-DQN/
├── train_pacman.py       # Main training script
├── dqn_agent.py          # Agent implementation with CNN
├── replay_buffer.py      # Experience replay buffer
├── utils.py              # Helper functions for preprocessing
├── videos/               # Directory to store gameplay recordings
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

Requirements

To run this project, ensure you have the following installed:

Python 3.11.6

Gymnasium (with Atari environments)

PyTorch

NumPy

Pillow (PIL)

torchvision

ALE-py

Install the required dependencies using:

pip install -r requirements.txt

Setup

Clone the repository:

git clone https://github.com/your-repo/pacman-dqn.git
cd pacman-dqn

Install dependencies as specified in the requirements.txt file.

Update the video_folder path in train_pacman.py to store recorded gameplay videos.

How It Works

Neural Network Architecture

The deep convolutional neural network processes game frames and predicts Q-values for each action:

Input Layer: Preprocessed game frames (grayscale, resized to 128x128 pixels).

Convolutional Layers: Extract spatial features using multiple convolutional layers, followed by ReLU activations and batch normalization.

Fully Connected Layers: Process extracted features and output Q-values for all possible actions.

Training Process

Data Collection: The agent interacts with the game environment and records experiences: (state, action, reward, next_state, done).

Experience Replay: Samples mini-batches of past experiences to train the neural network, reducing correlation between consecutive updates.

Target Network: Periodically updates a separate target network to stabilize learning.

Optimization: Uses Mean Squared Error (MSE) loss to minimize the difference between predicted and target Q-values, with backpropagation and the Adam optimizer.

Training Details

Hyperparameters

Learning Rate: 5e-4

Discount Factor (gamma): 0.99

Epsilon Decay: 0.995

Replay Buffer Size: 100,000

Mini-Batch Size: 64

Target Network Update Frequency: Every 100 episodes

Maximum Steps per Episode: 200

Epsilon-Greedy Strategy

The agent starts with a high probability of random actions (epsilon = 1.0) to explore the environment. Over time, epsilon decays until it reaches a minimum value (epsilon_min = 0.01), focusing on exploiting learned strategies.

Loss Function

The agent minimizes the temporal difference (TD) error:

where  is the target Q-value computed as:

Here,  is the target network's output.

Results

The agent begins training with random behavior and gradually improves as it learns to maximize rewards. Performance metrics such as average score and total reward are logged over episodes. The gameplay recording highlights the agent's progress visually.

Usage

To train the agent and save gameplay recordings:

Ensure the required packages are installed and the video folder is correctly set.

Run the training script:

python train_pacman.py

Monitor the agent's performance metrics during training.

Gameplay recordings will be saved in the specified directory for evaluation.

Contributions

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request. Together, we can make this project even better.