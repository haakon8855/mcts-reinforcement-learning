"""Haakon8855"""

from actor_network import ActorNetwork
from game_hex import GameHex
from reinforcement_learning import ReinforcementLearner


class Tournament():
    """
    Performs a tournament with the saved policies.
    """

    def __init__(self, sim_world):
        self.policies = []
        self.sim_world = sim_world
        self.init_policies()

    def init_policies(self):
        """
        Loads all saved weights and saves the instances of ActorNetwork (policies).
        """
        for i in range(ReinforcementLearner.num_policies):
            input_size = self.sim_world.get_state_size()
            output_size = self.sim_world.get_move_size()
            save_path = ReinforcementLearner.weights_path + self.sim_world.identifier + str(
                i)
            network = ActorNetwork(input_size, output_size, self.sim_world,
                                   save_path)
            network.load_weights()
            self.policies.append(network)

    def run(self):
        """
        Runs a tournament.
        """


def main():
    """
    Main function for running this python script.
    """
    sim_world = GameHex(4)
    topp = Tournament(sim_world)
    topp.run()


if __name__ == "__main__":
    main()
