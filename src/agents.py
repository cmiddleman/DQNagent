
from env import get_available_actions_mask, NUM_MARKS
from network import Memory, Network
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import one_hot

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

hyper_param_grid = {
    'etas':[.01, .005, .001, .0005, .0001],
    'taus':[1/8, 1/16, 1/32, 1/64, 1/128],
    'capacities':[2048, 4096, 8192, 16384],
    'gammas' : [.99, .95, .9],
    'batch_sizes' : [32, 64, 128]
}

class Agent:

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

class DQNAgent(Agent):


    def __init__(self, action_space=np.arange(9), explore=True,epsilon=1, epsilon_min=.1, epsilon_decay=.005, tau=1/8, eta=.001, capacity=4096, gamma=.99 ,batch_size=64):

            #TODO add randomized hyperparameter grid for player corpus.
            self.action_space = action_space
            self.memory = Memory(capacity=capacity)
            self.network = Network(eta=eta, tau=tau)
            self.explore = explore
            self.eta = eta
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
        if use_target_model:
            return self.network.target_model.predict(network_input)
        else: 
            return self.network.model.predict(network_input)

    def q_target(self, states, actions, rewards, next_states, dones):

        target_q_values = self.q(states, use_target_model=True)

        next_q_values = self.q(next_states, use_target_model=True)
        max_next_q_values = np.max(next_q_values, axis=1)
     
        target_q_values_for_action = rewards + (1-dones)*self.gamma*max_next_q_values

        target_q_values[:, actions] = target_q_values_for_action

        return target_q_values




    


human = HumanAgent()
human.learn()