import os  # For interacting with the operating system
import random  # For generating random parameters and sampling
import numpy as np  # For numerical operations and array manipulations
import torch  # For PyTorch operations
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms
import torch.nn.functional as F  # For activation functions
import torch.autograd as autograd  # For automatic differentiation
from torch.autograd import Variable  # For handling autograd Variables
from collections import deque, namedtuple  # For data storage and manipulation structures
from torch.utils.data import DataLoader, TensorDataset
from gym.wrappers import RecordVideo
import gym


# Define the neural network class, inheriting from nn.Module
class Network(nn.Module): #Inherit from nn module
    def __init__(self, action_size, seed = 42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        # building the eyes (convolution layers network)

        #Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride = 4) # 3 input channels (rgb), 32 output channels (convolutional layer) for pacman game, 8x8 kernel size, stride = 4
        # Batch Normalization operation for layer 1
        self.bn1 = nn.BatchNorm2d(32) # 32 is the number of output channells of convonutional layer 1

        #Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride = 2) # 32 input channels (convolutional layer 1), 64 output channels (convolutional layer 2) for pacman game, 4x4 kernel size, stride = 2
        # Batch Normalization operation for layer 2
        self.bn2 = nn.BatchNorm2d(64) # 64 is the number of output channells of convonutional layer 2

        #Convolutional layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride = 1) # 64 input channels (convolutional layer 2), 64 output channels (convolutional layer 3) for pacman game, 3x3 kernel size, stride = 1
        # Batch Normalization operation for layer 3
        self.bn3 = nn.BatchNorm2d(64) # 64 is the number of output channells of convonutional layer 3

        #Convolutional layer 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride = 1) # 64 input channels (convolutional layer 3) , 128 output channels (convolutional layer 4) for pacman game, 3x3 kernel size, stride = 1
        # Batch Normalization operation for layer 4
        self.bn4 = nn.BatchNorm2d(128) # 128 is the number of convonutional layer 4

        # Now agent have eyes

        # building the brain (full connection layers neural network)

        #flattening formula for each convolutional layer
        # The Pacman game typically uses an input size of  210 × 160 × 3 210×160×3 (height  𝐻 = 210 H=210, width  𝑊 = 160 W=160, and 3 color channels for RGB). Assuming no padding is applied ( Padding = 0 Padding=0):
        # Recalculate the output size of each convolutional layer based on the input size (128x128),
        # kernel size, stride, and padding using the formulas:
        #
        # H_out = floor((H_in - kernel_size + 2 * padding) / stride) + 1
        # W_out = floor((W_in - kernel_size + 2 * padding) / stride) + 1
        #
        # Convolutional Layers:
        # 1. Conv1:
        #    H_in = 128, W_in = 128, kernel_size = 8, stride = 4, padding = 0
        #    H_out = floor((128 - 8 + 2 * 0) / 4) + 1 = 31
        #    W_out = floor((128 - 8 + 2 * 0) / 4) + 1 = 31
        #    Resulting size: 31x31, output channels = 32
        #
        # 2. Conv2:
        #    H_in = 31, W_in = 31, kernel_size = 4, stride = 2, padding = 0
        #    H_out = floor((31 - 4 + 2 * 0) / 2) + 1 = 14
        #    W_out = floor((31 - 4 + 2 * 0) / 2) + 1 = 14
        #    Resulting size: 14x14, output channels = 64
        #
        # 3. Conv3:
        #    H_in = 14, W_in = 14, kernel_size = 3, stride = 1, padding = 0
        #    H_out = floor((14 - 3 + 2 * 0) / 1) + 1 = 12
        #    W_out = floor((14 - 3 + 2 * 0) / 1) + 1 = 12
        #    Resulting size: 12x12, output channels = 64
        #
        # 4. Conv4:
        #    H_in = 12, W_in = 12, kernel_size = 3, stride = 1, padding = 0
        #    H_out = floor((12 - 3 + 2 * 0) / 1) + 1 = 10
        #    W_out = floor((12 - 3 + 2 * 0) / 1) + 1 = 10
        #    Resulting size: 10x10, output channels = 128
        #
        # Final Flattened Size:
        # 10 * 10 * 128 = 12,800



        self.fc1 = nn.Linear(10*10*128, 512) # 512 neurons for the first fully connected layer by experiment
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

        # Now the agent have brain


        # Implementing the forward propagation

    def forward(self, state): #input is the state because its gonna propagate the state from the input layer to the output layer

        # *****propagating from the image to the convolutional layer

        # Signal from Images to 1st conv layer then from from 1st conv to 1st batch normalization layer, then activate it using rectifier func
        x = F.relu(self.bn1(self.conv1(state)))

        # Signal from 1st conv layer to 2nd conv layer then from  2nd conv to 2nd batch normalization layer, then activate it using rectifier func
        x = F.relu(self.bn2(self.conv2(x)))

        # Signal from 2nd conv layer to 3rd conv layer then from  3rd conv to 3rd batch normalization layer, then activate it using rectifier func
        x = F.relu(self.bn3(self.conv3(x)))

        # Signal from 3rd conv layer to 4th conv layer then from  4th conv to 4th batch normalization layer, then activate it using rectifier func
        x = F.relu(self.bn4(self.conv4(x)))

        # reshape for flattening
        x = x.view(x.size(0), -1)

        # ******* propagating from the conv layers to ANN 
        
        # propagate the signal from the input layer to first fully connected layer with rectfier activation function
        x = self.fc1(x) # take the state as the input to the first fully connected layer
        x = F.relu(x) # assigning it to rectifier activation function

        # propagate the signal from the  first fully connected layer to the second fully connected layer with rectfier activation function
        x = self.fc2(x) # take the first fully connected layer output as the input to the second fully connected layer
        x = F.relu(x) # assigning it to rectifier activation function

        # propagate the signal from the  second fully connected layer to the output with rectfier activation function
        return self.fc3(x) # take the second fully connected layer output as the input of the output layer


import ale_py
import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v4', render_mode='rgb_array')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)



# Define hyperparameters for the training
learning_rate = 5e-4 # From expermintation for Training Ai to play pacman
minibatch_size = 64 # Number of observations used in one step of the training to update the weights
discount_factor = 0.99 # Close to one to make the agent look for accumlated future reward (not being short sighted)
# REPLAY MEMORY NOT NEEDED FOR CONVOLUTIONAL DEPP Q - LEARNING
# Soft update not needed for this specific enviroment

# To make the input images converted into Pytorch tensors
# So they can be fed into the ANN

from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
  #convert the numpy array into PIL image
  frame = Image.fromarray(frame)
  # Do pre-processing (making the dimesnsions smaller and reshape it into squares 128x128 pixel)
  preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
  return preprocess(frame).unsqueeze(0) # .unsqueeze(0) to keep track of which batch each frame belongs to, and set it to the first dimension


# Define the agent class
class Agent():
    # state size not needed for images input

    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size

        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)

        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate) # Intiallizing the optimizer
        self.memory = deque(maxlen= 10000) # instead of the replay memory

    # Store experiences and decide when to learn from them
    def step(self, state, action, reward, next_state, done):
        
        #pre-process the state and the next state
        state = preprocess_frame(state).to(self.device)
        next_state = preprocess_frame(next_state).to(self.device)

        # append experience to the memory as tuple
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) > minibatch_size: #there are at least 64 of observations
            experiences = random.sample(self.memory, k = minibatch_size) # take random 64 sample (minibatch) of the observations from the memory 

            # then learn
            self.learn(experiences, discount_factor)

    #Act method thatt will select an action based on a given state and certain epsilon value for an epsilon greedy action selection policy
    def act(self, state, epsilon = 0.):
        
        state = preprocess_frame(state).to(self.device) # preprocess the state 
        self.local_qnetwork.eval # putting the local q network in evaluation mode

        # do check that we are in predection (inference) mode not training mode
        with torch.no_grad():
            # Now we making prediction
            action_values = self.local_qnetwork(state)

        #return back to training mode
        self.local_qnetwork.train()

        # Now use the epsilon, generate random number, if the random number > epsilon, then select the action number with the highest q value, else select random action
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # Make the learn method
    def learn(self, experiences, discount_factor):

        # Implementing Eligibility Trace (stacking the experience elements)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device) #stacking all the states from the sampled experienced together, # conver states into pytorch tensors, # Convert them to float, # Make sure this functions whether CPU or GPU
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device) #stacking all the actions from the sampled experienced together, # conver states into pytorch tensors, # Convert them to long integers, # Make sure this functions whether CPU or GPU
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device) #stacking all the rewards from the sampled experienced together, # conver states into pytorch tensors, # Convert them to float, # Make sure this functions whether CPU or GPU
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device) #stacking all the next states from the sampled experienced together, # conver states into pytorch tensors, # Convert them to float, # Make sure this functions whether CPU or GPU
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device) #stacking all the dones from the sampled experienced together, # conver states into pytorch tensors, # Convert them to boolean, # Make sure this functions whether CPU or GPU

        
        
        # prepare to compute Cross-Entropy Function
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) # forward propagate next state from our target q network, this gives the action values of our target q network propagating the next state, detatch the action values in the tensror, since we want to take the maximum q values, we need the maximum value along dimension 1, square bracket zero is because we dont want its indices
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))

        q_expected = self.local_qnetwork(states).gather(1, actions)

        # Compute the loss function (Cross-Entropy)
        loss = F.mse_loss(q_expected, q_targets)

        # Intialize the optimizer (reset it)
        self.optimizer.zero_grad()

        # Back Propagate the loss
        loss.backward()

        # single optimization step
        self.optimizer.step()

    # Self update not needed

        # # Update the target network parameters with thios of local network parameters
        # self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    # # Method that will update the parameters
    # def soft_update(self, local_model, target_model, interpolation_parameter):
    #   for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #     target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

# Initialize the agent
agent = Agent(number_actions)

# Training parameters
number_episodes = 4000
max_number_timesteps_per_episode = 10000
epsilon_start = 1.0  # Initial epsilon value
epsilon_end = 0.01  # Minimum epsilon value
epsilon_decay = 0.995  # Epsilon decay rate

epsilon = epsilon_start  # Initialize epsilon
scores_window = deque(maxlen=100)  # Store scores of the last 100 episodes

# Recording
# Wrap the environment with video recording, specifying a folder
video_folder = "C:/Users/HP/OneDrive/Desktop/AI/PacManvideos"
recording_episodes = []

# Training loop
for episode in range(1, number_episodes + 1):
    state, info = env.reset()  # Reset the environment
    score = 0  # Initialize score for the episode

    # Check if we should record this episode
    if episode % 50 == 0 or (np.mean(scores_window) >= 500.0 and episode == number_episodes):
        # Wrap the environment with video recording for this specific episode
        env = gym.wrappers.RecordVideo(env, video_folder)
        recording_episodes.append(episode)

    # Start the episode
    for t in range(max_number_timesteps_per_episode):
        action = agent.act(state, epsilon)  # Select an action
        next_state, reward, done, _, _ = env.step(action)  # Execute action

        # Store the experience and train
        agent.step(state, action, reward, next_state, done)

        state = next_state  # Transition to the next state
        score += reward  # Accumulate the reward

        if done:
            break  # End the episode if done

    scores_window.append(score)  # Add the score to the window
    epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Decay epsilon

    # Print progress only if scores_window has valid values
    if len(scores_window) > 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
    else:
        print(f'\rEpisode {episode}\tAverage Score: 0.00', end="")

    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}')

    # Check if environment is solved
    if np.mean(scores_window) >= 700.0 and episode not in recording_episodes:
        # Wrap the environment with video recording when the agent solves it
        env = gym.wrappers.RecordVideo(env, video_folder)
        print(f'\nEnvironment solved in episode {episode}!\tAverage Score: {np.mean(scores_window):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break  # End the training once the environment is solved

env.close()  # Close the environment after training