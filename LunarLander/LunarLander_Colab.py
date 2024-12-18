import os  # For the operating system
import random  # For random parameters
import numpy as np  # For arrays mathematics
import torch  # For pytourch
import torch.nn as nn # for nueral network
import torch.optim as optim # For the optimizer
import torch.nn.functional as F # for the activation function
import torch.autograd as autograd # for strochastic gradient descent
from torch.autograd import Variable
from collections import deque, namedtuple

class Network(nn.Module): # creating the netowrk class and inherting it from tourch.nn library
  def __init__(self, state_size, action_size, seed = 42):  # state size is observation space (vectors), action_size (number of actions the agent can make), seed for the randomness
      super(Network, self).__init__()
      self.seed = torch.manual_seed(seed) #activating the seed

      # Architicture of the Neural Network start
      self.fc1 = nn.Linear(state_size, 64)  # fc1 is the first full connection layer between the input layer and the first full connected layer, 64 is experimental number for this type of AI
      self.fc2 = nn.Linear(64, 64) # from experiments we need two fully connected layer
      self.fc3 = nn.Linear(64, action_size) # connection between second fully connected layer and the output (actions)
      # Architicture of the Neural Network end

  # Making forward propagation function
  def forward(self, state): #input is the state because its gonna propagate the state from the input layer to the output layer

    # propagate the signal from the input layer to first fully connected layer with rectfier activation function
    x = self.fc1(state) # take the state as the input to the first fully connected layer
    x = F.relu(x) # assigning it to rectifier activation function

    # propagate the signal from the  first fully connected layer to the second fully connected layer with rectfier activation function
    x = self.fc2(x) # take the first fully connected layer output as the input to the second fully connected layer
    x = F.relu(x) # assigning it to rectifier activation function

    # propagate the signal from the  second fully connected layer to the output with rectfier activation function
    return self.fc3(x) # take the second fully connected layer output as the input of the output layer

import gymnasium as gym  # Contains the game enviroment
env = gym.make('LunarLander-v3') # The Lunar Lander environment was upgraded to v3 (the game)
state_shape = env.observation_space.shape  # assigning the enviroment shape to state_shape (vector of the elements)
state_size = env.observation_space.shape[0] # assigning the state size (inputs)
number_actions = env.action_space.n #assigning number of actions the agent can make (outputs)
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

learning_rate = 5e-4 # From expermintation for Training Ai to land on the moon
minibatch_size = 100 # Number of observations used in one step of the training to update the weights
discount_factor = 0.99 # Close to one to make the agent look for accumlated future reward (not being short sighted)
replay_buffer_size = int(1e5) # Size of memory of the AI, how many experiences including state, action, reward, next state wheteher done or not in the memory of the agent, to break the coolerations
interpolation_parameter = 1e-3 # (tau) the number will be used in updating the parameters, from expermentation as well

class ReplayMemory(object):
  def __init__(self, capacity): # capacity of the memory
    # If you want to execute this code outside of collab (Jupyter notebook)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.capacity = capacity # Maximum size of the memory buffer
    self.memory = [] # the list that will store the experiences, each one contating the state, the action, the reward, the next state and whether we are done or not

  # The function that will add the experience (event) to the memory buffer
  def push(self, event):
    self.memory.append(event) # appending the event to the memory
    # Ensure the memory doesn't exceed it's capacity
    if len(self.memory) > self.capacity:
      # Delete the oldest event
      del self.memory[0]

  # Randomly select batch of experiences(events) from the memory buffer
  def sample(self, batch_size): # size of the batch of the events that will be taken for training
    experiences = random.sample(self.memory, k = batch_size) # take the random sample
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None ])).float().to(self.device) #stacking all the states from the sampled experienced together, # conver states into pytorch tensors, # Convert them to float, # Make sure this functions whether CPU or GPU
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None ])).long().to(self.device) #stacking all the actions from the sampled experienced together, # conver states into pytorch tensors, # Convert them to long integers, # Make sure this functions whether CPU or GPU
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None ])).float().to(self.device) #stacking all the rewards from the sampled experienced together, # conver states into pytorch tensors, # Convert them to float, # Make sure this functions whether CPU or GPU
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None ])).float().to(self.device) #stacking all the next states from the sampled experienced together, # conver states into pytorch tensors, # Convert them to float, # Make sure this functions whether CPU or GPU
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device) #stacking all the dones from the sampled experienced together, # conver states into pytorch tensors, # Convert them to boolean, # Make sure this functions whether CPU or GPU
    return states, next_states, actions, rewards, dones

class Agent():

  def __init__(self, state_size, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.state_size = state_size
    self.action_size = action_size

    self.local_qnetwork = Network(state_size, action_size).to(self.device)
    self.target_qnetwork = Network(state_size, action_size).to(self.device)

    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate) # Intiallizing the optimizer
    self.memory = ReplayMemory(capacity = replay_buffer_size) # Intiallize the memory
    self.t_step = 0 # Time step

  # Store experiences and decide when to learn from them
  def step(self, state, action, reward, next_state, done):
    self.memory.push((state, action, reward, next_state, done)) # store the event components (in tuple) into the agent memory using the push method we created
    self.t_step = (self.t_step + 1) % 4 # increment the time step by 1, then reset it it when it is equal to 4, so we learn every 4 steps
    if self.t_step == 0: # we reached the 4 steps
      if len(self.memory.memory) > minibatch_size: #there are at least 100 of observations
        experiences = self.memory.sample(minibatch_size) # take the sample of the observations

        # then learn
        self.learn(experiences, discount_factor)

  #Act method thatt will select an action based on a given state and certain epsilon value for an epsilon greedy action selection policy
  def act(self, state, epsilon = 0.):
      state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # convert state to torch tensor, add an extra dimension to the state correspond to the batch (which batch this state belongs to) at the begainning of the vector
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
      states, next_states, actions, rewards, dones = experiences
      next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) # forward propagate next state from our target q network, this gives the action values of our target q network propagating the next state, detatch the action values in the tensror, since we want to take the maximum q values, we need the maximum value along dimension 1, square bracket zero is because we dont want its indices
      q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))

      q_expected = self.local_qnetwork(states).gather(1, actions)

      # Compute the cost function (loss)
      loss = F.mse_loss(q_expected, q_targets)

      # Intialize the optimizer (reset it)
      self.optimizer.zero_grad()

      # Back Propagate the loss
      loss.backward()

      # single optimization step
      self.optimizer.step()

      # Update the target network parameters with thios of local network parameters
      self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

  # Method that will update the parameters
  def soft_update(self, local_model, target_model, interpolation_parameter):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)


agent = Agent(state_size = state_size, action_size = number_actions)


number_episodes = 2000
max_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)

for episodes in range(1, number_episodes + 1):
  # reset enviroment to intial state
  state, _ = env.reset()

  # intialize the score (cumulative reward)
  score = 0

  for t in range(max_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)

    agent.step(state=state, action=action, reward=reward, next_state=next_state, done=done)
    state = next_state
    score += reward

    if done:
      break

  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

  # Dynamic print
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episodes, np.mean(scores_on_100_episodes)), end = "")
  if episodes % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episodes, np.mean(scores_on_100_episodes)))

  if np.mean(scores_on_100_episodes) >= 200.0:
    print('\nEnviroment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episodes - 100, np.mean(scores_on_100_episodes)))

    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break #exit training

import glob
import io
import base64
import imageio
from IPython.display import HTML, display

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v3')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()