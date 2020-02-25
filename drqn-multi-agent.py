from random import randint
import numpy as np

#import keras.backend.tensorflow_backend as backend

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#os.environ["GPU_MAX_HEAP_SIZE"] = "99"
#os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
#os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LSTM, TimeDistributed
from keras.optimizers import Adam, RMSprop
#from keras.callbacks import TensorBoard
#import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm

#from PIL import Image
import cv2

jupyter = True

if not jupyter:

  try:
    from google.colab.patches import cv2_imshow
    runincolab = True
  except:
    runincolab = False
    import cv2
else:
  runincolab = False

  from IPython.display import Image
  import IPython.display
  import PIL.Image
  

class Agent:
  def __init__(self, x, y, iad):
    self.x = x
    self.y = y
    self.model = None
    self.agentId = iad
    
    self.attached = False
    self.attached_block = None
    
    self.experience = []
    self.epsum = 0
    
    self.state = None

  def setAttach(self, food):
    if not food.attached:
        self.attached = True
        self.attached_block = food
        food.setAttached(True)
    
  def setDetach(self):
    if self.attached:
      self.attached = False
      self.attached_block.setAttached(False)
      pos = (self.attached_block.getX(), self.attached_block.getY())
      self.attached_block = None
      return pos
    return ()
    
    
  def getX(self):
    return self.x
  def getY(self):
    return self.y

  def setX(self, x):
    prevpos = self.x
    self.x = x
    if self.attached:
      self.attached_block.setX(self.attached_block.getX() + (x - prevpos) )
    
  def setY(self, y):
    prevpos = self.y
    self.y = y
    if self.attached:
      self.attached_block.setY(self.attached_block.getY() + (y - prevpos) )
    

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
    self.attached = False

  def getX(self):
    return self.x
  def getY(self):
    return self.y
  
  def setX(self, x):
    self.x = x
  def setY(self, y):
    self.y = y

  def setAttached(self, status):
    self.attached = status

  def getAttached(self):
    return self.attached

class Dispenser:
  def __init__(self,x ,y):
    self.x = x
    self.y = y
    

##
# Multi agent environment where agents should move food to the center area independently
#
#
##
class Environment:
  def __init__(self, size_x = 10, size_y = 10):
    
    self.blocks = []
    self.agents = []
    self.food = []
    self.dispensers = []
    self.submission_areas = []
    

    self.size_x = size_x
    self.size_y = size_y
    
    self.SUBMISSION_AREA = (3,6) # square submission area

    self.MIN_REWARD = -200

    self.EPISODES = 10000
    self.MAX_STEPS = 32

    self.epsilon = 1
    self.EPSILON_DECAY = 0.9995
    self.MIN_EPSILON = 0.05
    
    self.ep_rewards = [-200]
    self.AGGREGATE_STATS_EVERY = 20  # episodes

    self.FOOD_REWARD = 100
    self.ATTACH_REWARD = 10
    self.BLOCK_REWARD = -1

    self.maxQs = []

  def createFoods(self):
    self.food = []
    positions = []
    
    for i in range(1,9):
      positions.append((0,i))
    for i in range(1,9):
      positions.append((9,i))
    for i in range(1,9):
      positions.append((i,0))
    for i in range(1,9):
      positions.append((i,9))
    
    for i in range(6):
        pos = random.randint(0,len(positions)-1)
        self.food.append(Food(positions[pos][0],positions[pos][1]))
        positions.pop(pos)

    
  def createEnv(self):
    self.agents.append(Agent(0,0,1))
    self.agents.append(Agent(9,0,2))
    self.createFoods()
    self.createBlocks()

    if jupyter:
      IPython.display.display(PIL.Image.fromarray(cv2.resize(self.getMap(), (250,250), interpolation = cv2.INTER_AREA)))

  def createBlocks(self):
    self.blocks = []
    for i in range(6):
      self.blocks.append(Block(randint(1,8),randint(1,2)))
      self.blocks.append(Block(randint(1,8),randint(7,8)))
    for i in range(3):
      self.blocks.append(Block(randint(1,2),randint(3,6)))      
      self.blocks.append(Block(randint(7,8),randint(3,6)))


  def step(self, agent, action, step):

    reward = 0

    if step > self.MAX_STEPS:
      done = True
    else:
      done = False
    
    blocks = False

    if action == 0:
        
        for i in self.blocks:
          if agent.getX() == i.getX() and agent.getY() - 1 == i.getY():
            #reward = -1
            blocks = True
        
        if agent.getY() == 0 or blocks:
          #reward = -1
          pass
        else:
          agent.setY( agent.getY() -1 )

    if action == 1:
        
        for i in self.blocks:
          if agent.getX() +1 == i.getX() and agent.getY() == i.getY():
            #reward = -1
            blocks = True
        
        if agent.getX() +1 == self.size_x or blocks:
          #reward = -1
          pass
        else:
          agent.setX( agent.getX() + 1 )

    if action == 2:
        
        for i in self.blocks:
          if agent.getX() == i.getX() and agent.getY() + 1 == i.getY():
            #reward = -1
            blocks = True
        
        if agent.getY() == self.size_y - 1 or blocks:
          #reward = -1
          pass
        else:
          agent.setY( agent.getY() + 1 )

    if action == 3:
        
        for i in self.blocks:
          if agent.getX() -1 == i.getX() and agent.getY() == i.getY():
            #reward = -1
            blocks = True
        
        if agent.getX() == 0 or blocks:
          #reward = -1
          pass
        else:
          agent.setX( agent.getX() -1 )
          
          
    # attach
    if action == 4:
      acceptable = [(agent.getX()-1, agent.getY()),(agent.getX()+1, agent.getY()),(agent.getX(), agent.getY()-1),(agent.getX(), agent.getY()+1)]
      for i in self.food:
        
        if (i.getX(),i.getY()) in acceptable:
          #and ( (i.getX() < self.SUBMISSION_AREA[0] or i.getX() > self.SUBMISSION_AREA[1]) and (i.getY() < self.SUBMISSION_AREA[0] or i.getY() > self.SUBMISSION_AREA[1] )) :
          #print("attaching")
          agent.setAttach(i)
          if not i.getAttached():
            reward = self.ATTACH_REWARD
            i.setAttached()
          break
    
    # detach
    if action == 5:
      foodtoremove = agent.attached_block
      d_in = agent.setDetach()
      if len(d_in) > 0 and d_in[0] >= self.SUBMISSION_AREA[0] and d_in[0] <= self.SUBMISSION_AREA[1] and d_in[1] >= self.SUBMISSION_AREA[0] and d_in[1] <= self.SUBMISSION_AREA[1]:
        reward = self.FOOD_REWARD
        self.food.remove(foodtoremove)
      
    if blocks:
      reward = self.BLOCK_REWARD 
    #if len(self.food) == 0:
    #  done = True


    return reward, done

  # Get the vision of single agent
  def getVision(self, x, y):
    vision = np.zeros((11,11), dtype="uint8")

    for cx in range(11):
      for cy in range(11):
        if cx-5 + x >= 0 and cx-5 + x < self.size_x and cy-5 + y >= 0 and cy-5 + y < self.size_y :
          vision[cx,cy] = 32
        if cx-5 + x >= self.SUBMISSION_AREA[0] and cx-5 + x <= self.SUBMISSION_AREA[1] and cy-5 + y >= self.SUBMISSION_AREA[0] and cy-5 + y <= self.SUBMISSION_AREA[1] :
          vision[cx,cy] = 24
    
    for i in self.blocks:
      if i.getX() > x - 5 and i.getX() < x + 5 and i.getY() > y - 5 and i.getY() < y + 5:
        vision[i.getX()-x+5,i.getY()-y+5] = 64

    for i in self.food:
      if i.getX() > x - 5 and i.getX() < x + 5 and i.getY() > y - 5 and i.getY() < y + 5:
        vision[i.getX()-x+5,i.getY()-y+5] = 150
        
    for i in self.agents:
      if i.getX() > x - 5 and i.getX() < x + 5 and i.getY() > y - 5 and i.getY() < y + 5:
        vision[i.getX()-x+5,i.getY()-y+5] = 220
    


    vision[5,5] = 255

    return vision

  # Visualization of the full map
  def getMap(self):
    map = np.full((self.size_x, self.size_y), fill_value=32, dtype="uint8")

    map[self.SUBMISSION_AREA[0]:self.SUBMISSION_AREA[1]+1,self.SUBMISSION_AREA[0]:self.SUBMISSION_AREA[1]+1] = np.full( (self.SUBMISSION_AREA[1] - self.SUBMISSION_AREA[0] + 1,self.SUBMISSION_AREA[1] - self.SUBMISSION_AREA[0] +1 ), fill_value=16)

    for i in self.agents:
      map[i.getX(),i.getY()] = 255
    for i in self.blocks:
      map[i.getX(),i.getY()] = 64
    for i in self.food:
      try:
        map[i.getX(),i.getY()] = 150
      except:
        pass
    

    return map

  def createDQNs(self):
    shared_model = DQNAgent(1)
    for i in self.agents:
      i.setModel(shared_model)

  def run(self):
        
  ##############
  # This is where the magic happens
  #
  #
  ##############


    for episode in tqdm(range(1, self.EPISODES + 1), ascii=True, unit='episodes'):

      self.createFoods()
    
      startin_pos = [(0,0),(9,0),(0,9),(9,9),(3,3),(6,6),(3,6),(6,3)]
      for agent in self.agents:

        init_pos = startin_pos[random.randint(0,len(startin_pos)-1 )]
        startin_pos.remove(init_pos)
        agent.setDetach()
        agent.setX(init_pos[0])
        agent.setY(init_pos[1])
        agent.epsum = 0
        
        agent.experience = []
            
        current_vision = self.getVision(agent.getX(),agent.getY())
        agent.state = current_vision #np.stack([current_vision]*10 , axis = 0)
        agent.state.shape = (11,11,1)

      step = 0
  
      done = False

      reward_sum_for_ep = 0


      while not done:
            
        for agent in self.agents:

          if np.random.random() > self.epsilon:
            stack =  agent.model.create_stack()
            stack.shape = (1,agent.model.HISTORYFRAMES,11,11,1)
            actions = agent.model.get_qs(stack)
            action = np.argmax(actions)
            self.maxQs.append(max(actions))
          else:
            action = np.random.randint(0,agent.model.ACTION_SPACE_SIZE)

          reward, done = self.step(agent,action, step)

          agent.epsum += reward
          reward_sum_for_ep += reward  

          next_observation = self.getVision(agent.getX(),agent.getY())

          # take the next observation as the last frame of the states
          #next_state = np.append(agent.state[1:,:, :, : ], next_observation, axis=0 )
          #print("state,", agent.state.shape, "obs", next_observation.shape)
          next_state = next_observation #np.insert(agent.state[1:,:, :], 0, next_observation, axis=0)
          next_state.shape=(11,11,1)  
          #next_state = agent.state
          #for i in range(len(next_state.shape[0])):
          #  next_state[0]
          #next_state = next_observation


          #agent.model.update_replay_memory((agent.state, action, reward, next_state, done))
          
          agent.experience.append((agent.state, action, reward, next_state, done))
          
          #if episode > 320:
          agent.model.train(done, step)

          #current_states[i] = new_state
        
          agent.state = next_state
            
          if not runincolab:
            img = cv2.resize(next_observation, (250,250), interpolation = cv2.INTER_AREA)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.5
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(img,str(reward_sum_for_ep)+" "+["up", "right", "down", "left","attach","detach"][action], 
              (10,15), 
              font, 
              fontScale,
              fontColor,
              lineType)
            
            #cv2.imshow("Agent number %s"%(str(agent.agentId)), img)
            Image(data=img)
            #cv2.waitKey(1)
        step += 1

      if reward_sum_for_ep >= max(self.ep_rewards):
        best_try = self.getMap()

        
      # share experiences after episode
      for i in range(len(self.agents)):
        #for x in range(len(self.agents)):
        #  if x == i:
        #    continue
        for exp in self.agents[i].experience:
          self.agents[i].model.update_replay_memory(exp)

      for index, agent in enumerate(self.agents, start=0):
        self.ep_rewards.append(reward_sum_for_ep)
        if episode % self.AGGREGATE_STATS_EVERY == 0:
          average_reward = sum(self.ep_rewards[1:])/(len(self.ep_rewards)-1)
          min_reward = min(self.ep_rewards[1:])
          max_reward = max(self.ep_rewards)
          avg_100 = sum(self.ep_rewards[-100:])/(100)
          max_100 = max(self.ep_rewards[-100:])
          print("| max_reward", max_reward, "| min_reward", min_reward, "| avg", average_reward, "| avg for last 100 eps", avg_100, "|max for last 100: ", max_100,  "| epsilon", self.epsilon)

          if runincolab:
            cv2_imshow(cv2.resize(best_try, (250,250), interpolation = cv2.INTER_AREA) )
          if jupyter:
            IPython.display.display(PIL.Image.fromarray(cv2.resize(best_try, (250,250), interpolation = cv2.INTER_AREA)))
          

        #agent.model.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

      if self.epsilon > self.MIN_EPSILON:
        self.epsilon *= self.EPSILON_DECAY
        self.epsilon = max(self.MIN_EPSILON, self.epsilon)        




# Model for agent
class DQNAgent:
  def __init__(self, agent_id):

    self.DISCOUNT = 0.99

    self.MIN_REPLAY_MEMORY_SIZE = 320
    self.MINIBATCH_SIZE = 32
    self.UPDATE_TARGET_EVERY = 5
    self.INPUTSHAPE = (11,11, 1) # 
    self.ACTION_SPACE_SIZE = 6
    self.HISTORYFRAMES = 10

    self.REPLAY_MEMORY_SIZE = 50000

    self.global_replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
    

    self.model = self.create_model()        
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())    

    self.target_update_counter = 0

  def create_model(self):
    

    # plaidML doesn't currently support various length inputs for LSTM
    # create many-to-one LSTM network with TimeDistributed wrapper
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu'), input_shape=(self.HISTORYFRAMES,11, 11, 1)) )
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')))

    model.add(TimeDistributed(Flatten()))
    #model.add(Dense(512))
    model.add(LSTM(512, return_sequences=False))
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    model.add(Dense(self.ACTION_SPACE_SIZE))
    model.add(Activation('softmax'))

    #model.compile(loss="mse", optimizer=RMSprop(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    return model
  
  def update_replay_memory(self, transition):
        self.global_replay_memory.append(transition)

  # instead of storing past frames in replay memory, create them on the fly
  def create_stack(self, index=None, future=False):

    if index is None:
      index = len(self.global_replay_memory)-1
    
    counter = 1
    stack = []
    stack.append(self.global_replay_memory[index][0 if not future else 3])
    while True:
      # tmp is a tuple: (current_state, action, reward, new_current_state, done)
      tmp = self.global_replay_memory[index-counter]
      if index-counter < 0 or tmp[4] or len(stack) == self.HISTORYFRAMES:
        break
        
      else:
        
        stack.append(tmp[0 if not future else 3])
        
      counter += 1
    
    
    for i in range(self.HISTORYFRAMES-len(stack)):
      stack.append(stack[-1])
    
    stack.reverse()
    stack = np.array(stack)
    
    stack.shape = (self.HISTORYFRAMES,11,11,1)
    
    return stack
    
    
  def train(self, terminal_state, step):
    if len(self.global_replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
      return
    
    
    #batch_start = random.randint(0,(len(self.global_replay_memory)-32)//32)*32
    batch_start = random.randint(0,len(self.global_replay_memory)-33)
    
    minibatch = [self.global_replay_memory[i] for i in range(batch_start,batch_start+self.MINIBATCH_SIZE)]  #self.global_replay_memory[batch_start:batch_start+32]
    
    current_states = np.array([self.create_stack(i) for i in range(batch_start,batch_start+self.MINIBATCH_SIZE)])/255
    current_states.shape = (self.MINIBATCH_SIZE,self.HISTORYFRAMES,11,11,1)
    current_qs_list = self.model.predict(current_states, batch_size=self.MINIBATCH_SIZE)

    new_current_states = np.array([self.create_stack(i, future=True) for i in range(batch_start,batch_start+self.MINIBATCH_SIZE)])/255
    
    new_current_states.shape = (self.MINIBATCH_SIZE,self.HISTORYFRAMES,11,11,1)
    
    future_qs_list = self.target_model.predict(new_current_states)
    #print("future qs",future_qs_list)

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

      X.append(current_states[index])
      y.append(current_qs)
      
    X = np.array(X)
    X.shape = (self.MINIBATCH_SIZE,self.HISTORYFRAMES,11,11,1)

    #self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
    self.model.fit(X, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False) #, callbacks=[] if terminal_state else None)

    if terminal_state:
      self.target_update_counter += 1

    if self.target_update_counter > self.UPDATE_TARGET_EVERY:
      self.target_model.set_weights(self.model.get_weights())
      self.target_update_counter = 0


  def get_qs(self, stack):
    return self.model.predict(stack/255)[0]


env = Environment()
env.createEnv()
env.createDQNs()

env.run()