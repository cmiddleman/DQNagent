


from env import TicTacToe, SuperTicTacToe, X, O
from agents import HumanAgent, DQNAgent
import numpy as np


hyper_param_grid = {
    'etas':[.01, .005, .001, .0005, .0001],
    'taus':[1/8, 1/16, 1/32, 1/64, 1/128],
    'capacities':[2048, 4096, 8192, 16384],
    'gammas' : [.99, .95, .9],
    'batch_sizes' : [32, 64, 128, 256]
}

def create_random_agent():
    eta = np.random.choice(hyper_param_grid['etas'])
    tau = np.random.choice(hyper_param_grid['taus'])
    gamma = np.random.choice(hyper_param_grid['gammas'])
    capacity = int(np.random.choice(hyper_param_grid['capacities']))
    batch_size = int(np.random.choice(hyper_param_grid['batch_sizes']))

    return DQNAgent(eta=eta, tau=tau, gamma=gamma, capacity=capacity, batch_size=batch_size)

class Game_Handler:
    
    def __init__(self, num_agents=16):
        self.agent_corpus = []
        for _ in range(num_agents):
            self.agent_corpus.append(create_random_agent())

    def play_game(self, env=TicTacToe(), players = {X:DQNAgent(), O:DQNAgent()}, do_render = False):
        """play through a single loop of the environment attirbute and pass the game's episode to the agents as well as calls a round of learning on each.

        Args:
            do_render (bool, optional): whether or not to render the environment after each step. Defaults to False.

        """
        #reset environment and initilize the two-player episode dict
        env.reset()
        episode = {X: [], O: []}

        #game loop
        while(True):
            board, player = env.get_obs()

            #get the action from the current player's policy
            action = players[player].policy((board, player))
            new_obs, reward, done, info = env.step(action)
            #TODO make sure things work out okay with the player changing when the agent saves the board states as network input.

            episode[player].append(((board, player), action, player*reward, done))

            if(do_render):
                env.render()

            if done:
                #If the current player's move resulted in a terminal state, add that terminal state to the episode.
                #The 'reward' and 'done' from entering the terminal state will have been stored already in the previous (s,a,r,d) of the player's episode!
                #Also need to override the fact that the state has the other player as its turn!
                terminal_board, _ = new_obs
                episode[player].append(((terminal_board, player), None, None, None))

                #append the terminal state to the opponent player's episode. Since it has been discovered that the opponent's previous move led to a terminal state,
                #we must add in that reward to the previous state and say that it is terminal
                last_state, last_action, _, _ = episode[-player].pop()
                #TODO figure out reward business
                episode[-player].append((last_state, last_action, -player*reward, done))
                
                #append the terminal state for the opponent.
                episode[-player].append((new_obs, None, None, None))

                #have the agents write episode to their memory queue
                for agent in players.keys():
                    players[agent].record.append(agent*reward)
                    players[agent].remember(episode[agent])
                   
                

                break

    def play_round_robin(self):
        """Each agent in the player corpus takes a turn playing every other agent (including itself) from both X and O positions.
        """
        for agentX in self.agent_corpus:
            for agentO in self.agent_corpus:
                self.play_game(players={X: agentX, O: agentO})

        for agent in self.agent_corpus:
            agent.learn()
        


            



