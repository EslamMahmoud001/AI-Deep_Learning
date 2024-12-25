import cv2
import os  # For interacting with the operating system
import random  # For generating random parameters and sampling
import numpy as np  # For numerical operations and array manipulations
import torch  # For PyTorch operations
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms
import torch.nn.functional as F  # For activation functions
import gym
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import glob
import io
import base64
from IPython.display import HTML, display
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class Network(nn.Module):

  def __init__(self, action_size):
    super(Network, self).__init__()

    # Convolutional Layers (eyes)
    self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride =2) #input channels crossponding to 4 gray scale frames
    self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride =2)
    self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride =2)

    # Creating the flattening layer
    self.flatten = torch.nn.Flatten()


    #flattening formula for each convolutional layer

    # For the KungFuMaster game using ALE (Arcade Learning Environment), the default input frame size is typically 84x84

    # Recalculate the output size of each convolutional layer based on the input size (32x32),
    # kernel size, stride, and padding using the formulas:
    #
    # H_out = floor((H_in - kernel_size + 2 * padding) / stride) + 1
    # W_out = floor((W_in - kernel_size + 2 * padding) / stride) + 1

    # result is 512

    # Fully connected layers (brain)
    self.fc1 = torch.nn.Linear(512, 128)

    # Final output layers

    # action-values output
    self.fc2a = torch.nn.Linear(128, action_size)

    # state value ouput (critic)
    self.fc2s = torch.nn.Linear(128, 1)

  def forward(self, state):

    # forward propagating through convoltuion layer
    x = self.conv1(state)
    x = F.relu(x)  #activate signal
    x = self.conv2(x)
    x = F.relu(x)  #activate signal
    x = self.conv3(x)
    x = F.relu(x)  #activate signal

    # forward propagating through the flattening layer
    x = self.flatten(x)

    # forward propagating through the fully connected layer
    x = self.fc1(x)


    # forward propagating through the output layers
    action_values = self.fc2a(x)
    state_value = self.fc2s(x)[0]  # [0] to access the values

    return action_values, state_value

class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self, *args, **kwargs):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset(*args, **kwargs)
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)
    
    
import ale_py
import gymnasium as gym

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()
state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.env.get_action_meanings())

learning_rate = 1e-4    # From expermintation for Training Ai to play pacman
discount_factor = 0.99  # Close to one to make the agent look for accumlated future reward (not being short sighted)
number_enviroments = 10  # Here it is the A3C

# minibatch_size = 64     # Number of observations used in one step of the training to update the weights

# REPLAY MEMORY NOT NEEDED FOR CONVOLUTIONAL DEPP Q - LEARNING
# Soft update not needed for this specific enviroment

class Agent():

  def __init__(self, action_size):
    self.device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.network = Network(action_size).to(self.device) #brain
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)

  # Act method using the softmax policy
  def act(self, state):

    # batching the state
    if state.ndim == 3: # if state not in batch
      state = [state]  # Batch it
    
    # convert state into torch tensor
    state = torch.tensor(state, dtype = torch.float32, device= self.device)

    action_values, _ = self.network.forward(state) 

    #applying the softmax
    policy = F.softmax(action_values, dim = -1)
    return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])

  #step and learn integerated now (we taking batches here)
  def step(self, state, action, reward, next_state, done):
    batch_size = state.shape[0]

    #convert the numpy arrays to tensors
    state = torch.tensor(state, dtype = torch.float32, device= self.device)
    next_state = torch.tensor(next_state, dtype = torch.float32, device= self.device)  
    reward = torch.tensor(reward, dtype = torch.float32, device= self.device)
    done = torch.tensor(done, dtype = torch.bool, device= self.device).to(dtype=torch.float32)

    action_values, state_value = self.network(state)
    _, next_state_value = self.network(next_state)

    # compute the target state value using the bellman eqn
    target_state_value = reward + discount_factor * next_state_value * (1 - done)

    # Implementing advantage from the A3C model
    advantage = target_state_value - state_value

    # Implementing the critic loss part from the A3C model
    probs = F.softmax(action_values, dim = -1) #probabilities
    logprobs = F.log_softmax(action_values, dim = -1) # log probabilities
    entropy = -torch.sum(probs * logprobs, axis = -1)
    
    batch_idx = np.arange(batch_size)
    logprobs_actions = logprobs[batch_idx, action]  # select the action that are taken from the patch
    
    
    actor_loss = -(logprobs_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
    critic_loss = F.mse_loss(target_state_value.detach(), state_value)

    total_loss = actor_loss + critic_loss

    # Intialize the optimizer (reset it)
    self.optimizer.zero_grad()

    # Back Propagate the loss
    total_loss.backward()

    # single optimization step
    self.optimizer.step()

agent = Agent(action_size= number_actions)

def evaluate(agent, env, n_episodes = 1):
  episodes_rewards = []
  for _ in range(0, n_episodes):
    state, _ = env.reset()
    total_reward = 0
    while True:
      action = agent.act(state)
      state, reward, done, info, _ = env.step(action[0])
      total_reward += reward

      if done:
        break
    
    episodes_rewards.append(total_reward)
  
  return episodes_rewards
# Applying Asynchronus from the A3C model

class EnvBatch:

  def __init__(self, n_envs = 10):
    self.envs = [make_env() for _ in range (0, n_envs)]

  def reset(self):
    _state = []
    for env in self.envs:
      _state.append(env.reset()[0])
    
    return np.array(_state)

  def step(self, actions):
    next_states, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for a, env in zip(actions,self.envs)]))

    for i in range(len(self.envs)):
      if dones[i]:
        next_states[i] = self.envs[i].reset()[0]

    return next_states, rewards, dones, infos
import tqdm  # for the progress bar

env_batch = EnvBatch(n_envs = number_enviroments)
batch_states = env_batch.reset()

with tqdm.trange(0, 3001) as progress_bar: # 3000 iterations
  for i in progress_bar:
    batch_actions = agent.act(state = batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(actions = batch_actions)

    # stabilize the training
    batch_rewards *= 0.01  #(10 agents)

    # backpropagate (learn)
    agent.step(state=batch_states, action=batch_actions, reward=batch_rewards, next_state=batch_next_states, done=batch_dones)

    #update the states
    batch_states = batch_next_states

    # print the average reward every 1000 iterations
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent=agent, env=env, n_episodes=30)))


def show_video_of_model(agent, env):
    # Wrapping the environment with the RecordVideo wrapper to record the video
    env = RecordVideo(env, video_folder='KungFuVideo', episode_trigger=lambda episode_id: True)
    
    state, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])
    
    env.close()  # Close the environment when done
    # The video will be saved in the 'video' folder by default

show_video_of_model(agent, env)

def show_video():
    mp4list = glob.glob('video/*.mp4')  # Look for videos in the 'video' directory
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