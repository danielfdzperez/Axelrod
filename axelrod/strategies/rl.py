from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from axelrod.random_ import random_choice
from axelrod.strategy_transformers import FinalTransformer, TrackHistoryTransformer
import random
import numpy as np
from collections import deque

C, D = Action.C, Action.D

class Brain():

    """
    A player starts by cooperating and then mimics the previous action of the
    opponent.

    This strategy was referred to as the *'simplest'* strategy submitted to
    Axelrod's first tournament. It came first.

    Note that the code for this strategy is written in a fairly verbose
    way. This is done so that it can serve as an example strategy for
    those who might be new to Python.

    Names:

    - Rapoport's strategy: [Axelrod1980]_
    - TitForTat: [Axelrod1980]_
    """

    actions = {C:0, D:1}
    reverse_actions = {0:C, 1:D}

    memory_size = 5
    '''
        0 -> Both cooperates 
        1 -> The oppnent defect
        2 -> The player defect
        3 -> Both defects
    '''
    rewards = {0:1, 1:-3, 2:3, 3:-1}

    def __init__(self, training = False):
        #super().__init__()

        self.training = training

        #self.memory = np.zeros(5)
        self.memory = deque([0]*Brain.memory_size,maxlen=Brain.memory_size)
        self.qtable = np.zeros((1365, 2))
        self.learning_rate = 0.8
        self.gamma = 0.95 
        self.train_step = 0


        # Exploration parameters
        self.epsilon = 1.0                 # Exploration rate
        self.max_epsilon = 1.0             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability
        self.decay_rate = 0.005            # Exponential decay rate for exploration prob

    def reset(self):
        self.memory = deque([0]*Brain.memory_size,maxlen=Brain.memory_size)

    def generate_memory_state(self):
        return self.memory[0] + self.memory[1]*4 + self.memory[2]*16 + self.memory[3]*64 + self.memory[4] * 256

    def generate_new_state(self, action_player, action_coplayer):
        #Convierte a binario la jugada de este turno
        sub_state = action_player*2 + action_coplayer
        self.memory.appendleft(sub_state)
        return

    def play(self,player,opponent,noise=0):
        return self.simultaneous_play(player,opponent, noise)

    def simultaneous_play(self, player, coplayer, noise=0):
        #super().simultaneous_play(player,coplayer,noise)
        """This pits two players against each other."""
        s1, s2 = player.strategy(coplayer), coplayer.strategy(player)

        if self.training:
            self.train_step += 1
            state = self.generate_memory_state()
            my_action = self.actions[s1]
            opponent_action = self.actions[s2]
            self.generate_new_state(my_action, opponent_action)
            new_state = self.generate_memory_state()
            reward = self.rewards[self.memory[0]]

            self.qtable[state, my_action] = self.qtable[state, my_action] + self.learning_rate * (reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, my_action])

            # Reduce exploration
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*self.train_step) 

        if noise:
            s1 = random_flip(s1, noise)
            s2 = random_flip(s2, noise)
        player.update_history(s1, s2)
        coplayer.update_history(s2, s1)

        return s1, s2


    def strategy(self, opponent: Player) -> Action:
        """This is the actual strategy"""

        state = self.generate_memory_state()
        #Realizar la accion
        if not self.training or random.uniform(0, 1) > self.epsilon:
            action = self.reverse_actions[np.argmax(self.qtable[state,:])]
        else:
            action = self.reverse_actions[random.randint(0, 1)]


        return action


class RL(Player):
    brain = Brain(True)
    # These are various properties for the strategy
    name = "RL"
    classifier = {
        "memory_depth": 5,  # Four-Vector = (1.,0.,1.,0.)
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }
    def __init__(self, training = False):
        super().__init__()
        self.brain.reset()

    def play(self,opponent,noise=0):
        return self.brain.play(self,opponent, noise)

    def strategy(self, opponent: Player) -> Action:
        return self.brain.strategy(opponent)

    def reset(self):
        self.brain.reset()


