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
        if self.weights_index < 0:
            self.weights_index = self.num_policies - 1
        self.num_games = int(rl_conf['num_games'])
        if self.num_policies > self.num_games + 1:
            self.num_policies = self.num_games + 1
        elif self.num_policies < 2:
            self.num_policies = 2
        self.epochs_per_episode = int(rl_conf['epochs_per_episode'])
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
        self.layer_sizes = []
        self.layer_acts = []
        layers = [x for x in self.config.sections() if 'LAYER' in x]
        for layer in layers:
            layer_conf = self.config[layer]
            self.layer_sizes.append(int(layer_conf['layer_size']))
            self.layer_acts.append(layer_conf['layer_act'])

        # Initialize classes
        self.sim_world = GameHex(self.board_size)
        self.weights_path = self.weights_path + self.sim_world.identifier
        input_size = self.sim_world.get_state_size()
        output_size = self.sim_world.get_move_size()
        self.actor_network = ActorNetwork(input_size, output_size,
                                          self.sim_world, self.weights_path,
                                          self.layer_sizes, self.layer_acts,
                                          self.optimizer, self.lrate)
        self.mcts = MonteCarloTreeSearch(self.sim_world, self.actor_network,
                                         self.sim_games)
        self.reinforcement_learner = ReinforcementLearner(
            self.sim_world, self.actor_network, self.mcts, self.num_policies,
            self.weights_index, self.num_games, self.rl_epsilon)

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
                              self.weights_path, self.layer_sizes,
                              self.layer_acts, self.optimizer)
            topp.run()


def main():
    """
    Main function for running this python script.
    """
    # test_topp = TestTopp("config/config1.ini")  # 4x4 200ep 500sim
    # test_topp = TestTopp("config/config2.ini")  # 7x7 272ep 500sim
    # test_topp = TestTopp("config/config3.ini")  # 4x4 20ep 500sim
    test_topp = TestTopp("config/config4.ini")  # demo config, free to edit
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
