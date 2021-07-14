
from env import get_available_actions_mask, NUM_MARKS
from network import Memory, Network
import numpy as np
from tensorflow.keras.utils import to_categorical

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


    def __init__(self, action_space=np.arange(9), explore=True,epsilon=1, epsilonMin=.2,epsilonDecay=.995, tau=1/8, eta=.01, capacity=4096, gamma=.99 ,batchSize=128):

            #TODO add randomized hyperparameter grid for player corpus.
            self.action_space = action_space
            self.memory = Memory(capacity=capacity)
            self.network = Network(eta=eta, tau=tau)
            self.explore = explore
            self.eta = eta
            self.tau = tau
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilonMin = epsilonMin
            self.epsilonDecay = epsilonDecay
            self.batchSize = batchSize
    
    def policy(self, state):
        board, player = state
        available_actions = get_available_actions_mask(board)
        assert available_actions.any(), 'full board passed in, cant make any moves'

       #epsilon-greedy policy, first decide if should choose random action
        if self.explore and np.random.random() < self.epsilon:
           return np.random.choice(self.action_space[available_actions])
       
        #otherwise get the argmax from the network output
        q_table = self.q(state)[0]
        
        #filter out unavailable actions
        q_table[~available_actions] = float('-inf')

        #the index of the q_table with the highest value is our chosen action.
        return np.argmax(q_table)

    def learn(self):
        pass

    def remember(self, episode):
        state, action, reward, done = episode.pop(0)
        while(len(episode) > 0):
            next_state, next_action, next_reward, next_done = episode.pop(0)
            self.memory.add(state, action, reward, next_state, done)
            state = next_state
            action = next_action
            reward = next_reward
            done = next_done

    def q(self, state):
        network_input = one_hot_encode([state])
        return self.network.model.predict(network_input)

    def qtar(self):
        pass

    


human = HumanAgent()
human.learn()