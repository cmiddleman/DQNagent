


from env import TicTacToe, SuperTicTacToe, X, O
from agents import HumanAgent, DQNAgent





class Game_Handler:
    
    def __init__(self):
        pass
        
    def play_game(self, env=TicTacToe(), players = {X:DQNAgent(), O:DQNAgent()}, do_render = False):
        """play through a single loop of the environment attirbute and pass the game's episode to the agents.

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
                    players[agent].remember(episode[agent])

                break
        


            



#Game_Handler().play_game(do_render=True, players={X: DQNAgent(explore=False), O: DQNAgent(explore=False)})

