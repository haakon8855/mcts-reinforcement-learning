"""Haakon8855"""

from configuration import Config
from game_hex import GameHex
from reinforcement_learning import ReinforcementLearner
from actor_network import ActorNetwork
from monte_carlo_ts import MonteCarloTreeSearch
from topp import Tournament
# import cProfile
# import pstats


class TestTopp:
    """
    Main class for training the reinforcement learner and running the TOPP.
    """

    def __init__(self, config_file):
        self.config = Config.get_config(config_file)

        # Fetch config for rl system
        rl_conf = self.config['RL']
        self.weights_path = rl_conf['weights_path']
        self.weights_index = int(rl_conf['weights_index'])
        self.num_policies = int(rl_conf['num_policies'])
        self.num_games = int(rl_conf['num_games'])
        self.epochs_per_episode = int(rl_conf['epochs_per_episode'])
        self.batch_size = int(rl_conf['batch_size'])
        self.rl_epsilon = float(rl_conf['epsilon'])
        # Fetch config for actor
        actor_conf = self.config['ACTOR']
        self.lrate = float(actor_conf['lrate'])
        self.optimizer = actor_conf['optimizer']
        # Fetch config for mcts
        mcts_conf = self.config['MCTS']
        self.sim_games = int(mcts_conf['sim_games'])
        self.mcts_epsilon = float(mcts_conf['epsilon'])
        # Fetch config for the simworld
        simworld_conf = self.config['SIMWORLD']
        self.board_size = int(simworld_conf['board_size'])
        # Fetch config for the network structure
        section = 'ACTOR_NETWORK'
        network_conf = self.config[section]
        self.layer_sizes = []
        self.layer_acts = []
        i = 0
        while self.config.has_option(
                section, "layer_size" + str(i)) and self.config.has_option(
                    section, "layer_act" + str(i)):
            self.layer_sizes.append(int(network_conf['layer_size' + str(i)]))
            self.layer_acts.append(network_conf['layer_act' + str(i)])
            i += 1

        # Initialize classes
        self.sim_world = GameHex(self.board_size)
        input_size = self.sim_world.get_state_size()
        output_size = self.sim_world.get_move_size()
        self.actor_network = ActorNetwork(input_size, output_size,
                                          self.sim_world, self.weights_path,
                                          self.layer_sizes, self.layer_acts,
                                          self.lrate)
        self.mcts = MonteCarloTreeSearch(self.sim_world, self.actor_network)
        self.reinforcement_learner = ReinforcementLearner(
            self.sim_world, self.actor_network, self.mcts, self.num_policies,
            self.weights_path, self.weights_index, self.num_games,
            self.rl_epsilon)

    def train(self):
        """
        Trains the agent.
        """
        self.reinforcement_learner.train()

    def run(self, play=False):
        """
        Runs the program according to the configuration.
        """
        if play:
            self.reinforcement_learner.play_hex()
        else:
            topp = Tournament(self.sim_world, self.num_policies,
                              self.weights_path)
            topp.run()


def main():
    """
    Main function for running this python script.
    """
    test_topp = TestTopp("config/config2.ini")
    test_topp.train()
    test_topp.run(play=False)


if __name__ == "__main__":
    main()
    # prof = cProfile.Profile()
    # prof.run('main()')
    # prof.dump_stats('output.prof')

    # stream = open('output.txt', 'w')
    # stats = pstats.Stats('output.prof', stream=stream)
    # stats.sort_stats('cumtime')
    # stats.print_stats()
