"""Haakon8855"""

from monte_carlo_ts import MonteCarloTreeSearch
from game_nim import GameNim
from actor_network import ActorNetwork


class ReinforcementLearner():
    """
    A reinforcement learning algorithm implementing Monte Carlo Tree Search
    (MCTS) to train the default policy (in this case an ANN).
    """

    def __init__(self, save_interval: int = 5):
        self.save_interval = save_interval
        self.sim_world = GameNim(num_pieces=10, max_take=3)
        self.actor_network = None
        self.mcts_p1 = None
        self.mcts_p2 = None
        self.initialize_actor_network()
        self.initialize_mcts()

    def initialize_actor_network(self):
        """
        Initializes the actor network with the correct input
        and output parameters.
        """
        input_size = self.sim_world.get_state_size()
        output_size = self.sim_world.get_move_size()
        self.actor_network = ActorNetwork(input_size, output_size)

    def initialize_mcts(self):
        """
        Initializes monte carlo tree search with the correct parameters.
        """
        self.mcts_p1 = MonteCarloTreeSearch(board=self.sim_world,
                                            default_policy=self.actor_network,
                                            pid=0)
        self.mcts_p2 = MonteCarloTreeSearch(board=self.sim_world,
                                            default_policy=self.actor_network,
                                            pid=1)

    def run(self):
        """
        Runs the traning algorithm to train the default policy neural network
        mapping states to actions.
        """


def main():
    """
    Main function for running this python script.
    """
    rl = ReinforcementLearner()
    rl.run()


if __name__ == "__main__":
    main()
