
from env import get_available_actions_mask
from network import Memory, Network

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

    def __init__(self):
        #TODO add randomized hyperparameter grid for player corpus.
        self.network = Network()
        self.memory = Memory()
    
    def policy(self, state):
        action = input('choose a square (0-8)')
        return int(action)

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
        print(self.memory.memory)


human = HumanAgent()
human.learn()