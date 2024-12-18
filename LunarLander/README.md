Lunar Lander Reinforcement Learning Agent
This project implements a Deep Q-Network (DQN) agent to solve the LunarLander-v3 environment using reinforcement learning. The agent is trained to land a spacecraft on a lunar surface using a series of actions based on its current state. The DQN algorithm combines Q-learning with a deep neural network to approximate the optimal Q-value function, enabling the agent to learn effective policies without needing a model of the environment.

Project Overview
The project uses the following key components:

LunarLander-v3: The environment provided by the gymnasium library that simulates a lunar lander task. The goal is to land the spacecraft safely by controlling the thrust with limited fuel and avoiding obstacles.

Deep Q-Network (DQN): A reinforcement learning algorithm where a neural network is used to approximate the Q-values for state-action pairs. The agent uses this network to decide which action to take based on the current state.

Replay Memory: A mechanism to store and sample past experiences (state, action, reward, next state, done) to break the correlations between consecutive experiences. This improves learning stability.

Target Network: A copy of the Q-network used to compute the target Q-values for the update. The target network is updated periodically using a soft update mechanism.

Code Description
1. Network Class
The Network class defines the architecture of the neural network used by the agent. It is a fully connected feed-forward network with:

Input Layer: Size equal to the state size (observation space).
Hidden Layers: Two fully connected layers with 64 neurons each.
Output Layer: Size equal to the number of actions that the agent can take (action space).
The forward() method defines the forward pass through the network with ReLU activation functions for the hidden layers.

2. ReplayMemory Class
The ReplayMemory class is used to store experiences during training. It ensures that the memory buffer doesn't exceed the defined capacity and supports sampling random mini-batches of experiences for training.

push(): Adds an experience to the memory buffer.
sample(): Randomly selects a batch of experiences from the memory for training.
3. Agent Class
The Agent class contains the core logic of the agent, including:

Q-Networks: The agent maintains two Q-networks: the local_qnetwork (for selecting actions) and the target_qnetwork (for computing target Q-values).
Optimizer: Uses Adam optimizer for training the neural network.
Action Selection: The agent uses an epsilon-greedy policy to balance exploration and exploitation.
act(): Chooses an action based on the current state using the epsilon-greedy approach.
Training: The agent stores experiences and learns from them by updating the Q-values using the Bellman equation.
learn(): Updates the Q-network using the experiences sampled from the memory buffer.
Soft Update: Periodically updates the target network parameters by blending them with the local network parameters using a small interpolation factor (interpolation_parameter).
4. Training Loop
The training loop involves running episodes of the LunarLander environment, where the agent interacts with the environment, selects actions, stores experiences, and updates its policy. The epsilon value gradually decays from 1.0 to 0.01, allowing the agent to explore more at the start and focus on exploiting learned knowledge towards the end.

5. Saving the Model
After training, if the agent reaches an average score of 200 over the last 100 episodes, it is considered to have solved the environment. The trained model (Q-network) is saved to a file called checkpoint.pth.

6. Displaying the Agent's Performance
The show_video_of_model() function captures and saves the agent's actions in the LunarLander environment as a video file (video.mp4). This video is then displayed using HTML5 video embedding.

7. Dependencies
torch: For implementing the neural network and training using deep learning methods.
gymnasium: Provides the LunarLander-v3 environment and other reinforcement learning environments.
numpy: For numerical operations and array manipulations.
imageio: For saving the simulation video.
8. Usage
Install Dependencies: Make sure you have the required libraries installed.


pip install torch gymnasium numpy imageio
pip install "gymnasium[atari, accept-rom-license]"
apt-get install -y swig
pip install gymnasium[box2d]

Run the Training: To train the agent, simply run the script. It will train the agent using the DQN algorithm and save the trained model when the environment is solved.

View the Performance: After training, the agent's performance can be visualized in a video format by calling the show_video() function. It will display the saved video of the agent interacting with the LunarLander environment.

Adjust Parameters: The training parameters like learning rate, epsilon decay, and network architecture can be adjusted to experiment with different configurations.

Project Details
State Space: The state space consists of 8 continuous values representing the position, velocity, angle, and angular velocity of the lunar lander, as well as the amount of fuel available.
Action Space: The action space has 4 discrete actions: do nothing, fire the left orientation engine, fire the main engine, and fire the right orientation engine.
Conclusion
This project demonstrates the application of Deep Q-Networks (DQN) to solve the LunarLander-v3 environment. The agent uses reinforcement learning to learn an optimal policy for safely landing the spacecraft. This setup can be extended to other environments, and different architectures or algorithms can be tested to improve performance.