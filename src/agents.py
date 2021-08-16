

from env import get_available_actions_mask, NUM_MARKS
from network import Memory, Network
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import one_hot
from tensorflow.keras import backend as K
import tensorflow as tf
import numba 
from numba import jit

def one_hot_encode(states):
    """formats environment states into network suitable for the network via one hot encoding.

    Args:
        states (list): list of (board, player) environment states

    Returns:
        2D numpy array: one hot encoding of the states
    """
 
    #changes the board from mark encoded to agent vs. opponent encoded and flattens to match input shape of network
    board_agent_povs = np.array([board.flatten()*player for board, player in states])

    #encodes the board as a one-hot array
    return to_categorical(board_agent_povs % NUM_MARKS, num_classes=NUM_MARKS)

class Agent:

    def __init__(self):
        self.record = []
        self.action_space=np.arange(9)

    def policy(self, state):
        pass

    def learn(self):
        pass

    def remember(self, episode):
        pass

class HumanAgent(Agent):
    
    def policy(self, state):
        action = input('choose a square (0-8)')
        return int(action)

class RandomAgent(Agent):

    def policy(self, state):
        board, player = state
        available_actions = get_available_actions_mask(board)
        assert available_actions.any(), 'full board passed in, cant make any moves'
        
        return np.random.choice(self.action_space[available_actions])

class DQNAgent(Agent):


    def __init__(self, explore=True,epsilon=1, epsilon_min=.1, epsilon_decay=.0005, tau=1/1024, eta=.001, eta_min=.0001,eta_decay=.0005, capacity=16384, gamma=.95 ,batch_size=2, dropout=False):
            super().__init__()

            self.memory = Memory(capacity=capacity)
            self.network = Network(eta=eta, tau=tau, dropout=dropout)
            self.explore = explore
            self.eta = eta
            self.eta_min = eta_min
            self.eta_decay = eta_decay
            self.tau = tau
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size

            
    
    def policy(self, state):
        board, player = state
        available_actions = get_available_actions_mask(board)
        assert available_actions.any(), 'full board passed in, cant make any moves'

       #epsilon-greedy policy, first decide if should choose random action
        if self.explore and np.random.random() < self.epsilon:
           return np.random.choice(self.action_space[available_actions])
       
        #otherwise get the argmax from the network output

        q_table = self.q([state])[0]

        #filter out unavailable actions
        q_table[~available_actions] = float('-inf')

        #the index of the q_table with the highest value is our chosen action.
        return np.argmax(q_table)

    def learn(self):
        #first ensure that there are sufficient experiences in memory
        if len(self.memory) < 4*self.batch_size:
            return
        
        #get sample experiences from memory
        states, actions, rewards, next_states, dones = self.memory.sample_experiences(self.batch_size)

        target_q_values = self.q_target(states, actions, rewards, next_states, dones)

     
        self.network.model.fit(one_hot_encode(states), target_q_values, epochs=1, verbose=0)

        self.network.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= 1 - self.epsilon_decay

        #do learning_rate decay
        if self.eta > self.eta_min:
            self.eta *= 1 - self.eta_decay
            K.set_value(self.network.model.optimizer.learning_rate, self.eta)
            K.set_value(self.network.target_model.optimizer.learning_rate, self.eta)


    def remember(self, episode):
        state, action, reward, done = episode.pop(0)
        while(len(episode) > 0):
            next_state, next_action, next_reward, next_done = episode.pop(0)
            self.memory.add(state, action, reward, next_state, done)
            state = next_state
            action = next_action
            reward = next_reward
            done = next_done

    def q(self, states, use_target_model=False):
        network_input = one_hot_encode(states)
        return self.predict(network_input, use_target_model)

    def predict(self, network_input, use_target_model=False):
        if use_target_model:
            return self.network.target_model.predict(network_input)
        else: 
            return self.network.model.predict(network_input)

    def q_target(self, states, actions, rewards, next_states, dones):

        target_q_values = self.q(states, use_target_model=False)

        next_q_values = self.q(next_states, use_target_model=True)
        max_next_q_values = np.max(next_q_values, axis=1)
     
        target_q_values_for_action = rewards + (1-dones)*self.gamma*max_next_q_values

        target_q_values[:, actions] = target_q_values_for_action

        return target_q_values

    def get_hyper_params(self):
        return {
            'eta' : self.eta,
            'tau' : self.tau,
            'gamma' : self.gamma,
            'capacity' : self.memory.capacity,
            'batch_size' : self.batch_size
        }




  

