from random import randint
import numpy as np

import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
#from PIL import Image
import cv2
try:
  from google.colab.patches import cv2_imshow
except:
  pass

class Agent:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.model = None

  def getX(self):
    return self.x
  def getY(self):
    return self.y

  def setX(self, x):
    self.x = x
  def setY(self, y):
    self.y = y

  def setModel(self, model):
    self.model = model


    
class Block:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def getX(self):
    return self.x
  def getY(self):
    return self.y


class Food:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def getX(self):
    return self.x
  def getY(self):
    return self.y

##
# Create a single agent in the top left corner, it has a vision of +-5,
# the size of the grid is 10x10 with food in the bottom right corner and
# there is 20 blocks in the grid
##
class Environment:
  def __init__(self, size_x = 10, size_y = 10):
    
    self.blocks = []
    self.agents = []
    self.food = []

    self.size_x = size_x
    self.size_y = size_y
    
    
    #  # Minimum number of steps in a memory to start training


    self.MODEL_NAME = '2x256'
    self.MIN_REWARD = -200  # For model save
    self.MEMORY_FRACTION = 0.20

    # Environment settings
    self.EPISODES = 100000

    # Exploration settings
    self.epsilon = 1  # not a constant, going to be decayed
    self.EPSILON_DECAY = 0.99975
    self.MIN_EPSILON = 0.001
    
    self.ep_rewards = [-200]
    self.AGGREGATE_STATS_EVERY = 20  # episodes





  def createEnv(self):
    self.agents.append(Agent(0,0))
    self.food.append(Food(9,9))

    for i in range(20):
      self.blocks.append(Block(randint(1,9),randint(1,9)))


  def step(self, agent, action):

    #reward: food +10, move into a block: -1 & episode done
    reward = int(max(0, (agent.getX() + agent.getY())/2 -3 ))
    done = False
    # action 0 - 3 -> move

    if action == 0:

        if agent.getX() == self.food[0].getX() and agent.getY() - 1 == self.food[0].getY():
          reward = 666
          done = True
        for i in self.blocks:
          if agent.getX() == i.getX() and agent.getY() - 1 == i.getY():
            reward = -10
            done = True
        
        if agent.getY() == 0:
          reward = -10
          done = True
        agent.setY( agent.getY() -1 )

    if action == 1:
        if agent.getX() +1 == self.food[0].getX() and agent.getY() == self.food[0].getY():
          reward = 666
          done = True
        for i in self.blocks:
          if agent.getX() +1 == i.getX() and agent.getY() == i.getY():
            reward = -10
            done = True
        
        if agent.getX() +1 == self.size_x:
          reward = -10
          done = True

        agent.setX( agent.getX() + 1 )



    if action == 2:
        if agent.getX() == self.food[0].getX() and agent.getY() + 1 == self.food[0].getY():
          reward = 666
          done = True
        for i in self.blocks:
          if agent.getX() == i.getX() and agent.getY() + 1 == i.getY():
            reward = -10
            done = True
        
        if agent.getY() == self.size_y - 1:
          reward = -10
          done = True

        agent.setY( agent.getY() + 1 )

    if action == 3:
        if agent.getX() -1  == self.food[0].getX() and agent.getY() == self.food[0].getY():
          reward = 666
          done = True
        for i in self.blocks:
          if agent.getX() -1 == i.getX() and agent.getY() == i.getY():
            reward = -10
            done = True
        
        if agent.getX() == 0:
          reward = -10
          done = True
        agent.setX( agent.getX() -1 )


    return reward, done

  def getVision(self, x, y):
    vision = np.zeros((11,11))

    for cx in range(11):
      for cy in range(11):
        if cx-5 + x >= 0 and cx-5 + x <= self.size_x and cy-5 + y >= 0 and cy-5 + y <= self.size_y :
          vision[cx,cy] = 32
    
    for i in self.blocks:
      if i.getX() > x - 5 and i.getX() < x + 5 and i.getY() > y - 5 and i.getY() < y + 5:
        vision[i.getX()-x+5,i.getY()-y+5] = 96

    for i in self.food:
      if i.getX() > x - 5 and i.getX() < x + 5 and i.getY() > y - 5 and i.getY() < y + 5:
        vision[i.getX()-x+5,i.getY()-y+5] = 220


    vision[5,5] = 255

    return vision


  def createDQNs(self):

    for i in self.agents:
      i.setModel(DQNAgent(1))
    

    


  def run(self):
        
  ##############
  # This is where the magic happens
  #
  #
  ##############
    for episode in tqdm(range(1, self.EPISODES + 1), ascii=True, unit='episodes'):

      # Update tensorboard step every episode
        
      for agent in self.agents:
        #agent.model.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        agent.setX(0)
        agent.setY(0)
        step = 0
        # Reset flag and start iterating until episode ends
      done = False

      reward_sum_for_ep = 0
      

      while not done:
            
        for agent in self.agents:


          current_state = self.getVision(agent.getX(),agent.getY())

          #print(current_state)

          if np.random.random() > self.epsilon:
            action = np.argmax(agent.model.get_qs(current_state))
          else:
            action = np.random.randint(0,agent.model.ACTION_SPACE_SIZE)
            
          #print(["up", "right", "down", "left"][action])  

          reward, done = self.step(agent,action)

          reward_sum_for_ep += reward

          next_state = self.getVision(agent.getX(),agent.getY())

          agent.model.update_replay_memory((current_state, action, reward, next_state, done))
          agent.model.train(done, step)

          #current_states[i] = new_state
        step += 1

      if reward_sum_for_ep > max(self.ep_rewards):
        best_try = next_state

      for index, agent in enumerate(self.agents, start=0):
        self.ep_rewards.append(reward_sum_for_ep)
        if episode % self.AGGREGATE_STATS_EVERY == 0:
          average_reward = sum(self.ep_rewards[1:])/(len(self.ep_rewards)-1)
          min_reward = min(self.ep_rewards[1:])
          max_reward = max(self.ep_rewards)
          avg_100 = sum(self.ep_rewards[-100:])/(100)
          print("max_reward", max_reward, "min_reward", min_reward, "avg", average_reward, "len", len(self.ep_rewards), "100 avg", avg_100)

          
          cv2_imshow(cv2.resize(best_try, (250,250), interpolation = cv2.INTER_AREA) )

        #agent.model.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

      if self.epsilon > self.MIN_EPSILON:
        self.epsilon *= self.EPSILON_DECAY
        self.epsilon = max(self.MIN_EPSILON, self.epsilon)
        

      #print("------------------------ FINISHED EPISODE ----------------------------")
        




# Agent class
class DQNAgent:
  def __init__(self, agent_id):

    self.DISCOUNT = 0.99

    self.MIN_REPLAY_MEMORY_SIZE = 200
    self.MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
    self.UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
    self.INPUTSHAPE = (11,11,1) # maybe should update to 4 consecutive frames like the original paper suggests
    self.ACTION_SPACE_SIZE = 4 # move around

    self.REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
    
    
    
    self.global_replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
    
    self.model = self.create_model()
        
    # Target network
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())    
    # An array with last n steps for training
    #self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    

    # Custom tensorboard object
    #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    # Used to count when to update target network with main network's weights
    self.target_update_counter = 0

  def create_model(self):
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=self.INPUTSHAPE))  # (11, 11, 4) a 11x11 x 4 BW images.
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(self.ACTION_SPACE_SIZE, activation='softmax'))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
  
  def update_replay_memory(self, transition):
        self.global_replay_memory.append(transition)

  def train(self, terminal_state, step):
    if len(self.global_replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
      return

    minibatch = random.sample(self.global_replay_memory, self.MINIBATCH_SIZE)
    
    #minibatch.shape = (32,11,11,1)

    current_states = np.array([transition[0] for transition in minibatch])/255
    current_states.shape = (self.MINIBATCH_SIZE,11,11,1)
    current_qs_list = self.model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in minibatch])/255
    new_current_states.shape = (self.MINIBATCH_SIZE,11,11,1)
    future_qs_list = self.target_model.predict(new_current_states)

    X = []
    y = []

    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

      if not done:
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + self.DISCOUNT * max_future_q
      else:
        new_q = reward

      current_qs = current_qs_list[index]
      current_qs[action] = new_q

      X.append(current_state)
      y.append(current_qs)
      
    X = np.array(X)
    X.shape = (self.MINIBATCH_SIZE,11,11,1)

    #self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
    self.model.fit(X/255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[] if terminal_state else None)

    if terminal_state:
      self.target_update_counter += 1

    if self.target_update_counter > self.UPDATE_TARGET_EVERY:
      self.target_model.set_weights(self.model.get_weights())
      self.target_update_counter = 0

  def get_qs(self, state):
    state = np.array(state)
    
    state.shape = (1,11,11,1)
    
    return self.model.predict(state/255)[0]


env = Environment()
env.createEnv()
env.createDQNs()

env.run()

