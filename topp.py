"""Haakon8855"""

import numpy as np
from matplotlib import pyplot as plt

from actor_network import ActorNetwork


class Tournament():
    """
    Performs a tournament with the saved policies.
    """

    def __init__(self, sim_world, num_policies: int, weights_path: str,
                 network_layer_sizes: list, network_layer_acts: list):
        self.sim_world = sim_world
        self.num_policies = num_policies
        self.weights_path = weights_path
        self.network_layer_sizes = network_layer_sizes
        self.network_layer_acts = network_layer_acts
        self.policies = []
        self.policies_win_count = [0] * num_policies
        self.init_policies()

    def init_policies(self):
        """
        Loads all saved weights and saves the instances of ActorNetwork (policies).
        """
        for i in range(self.num_policies):
            input_size = self.sim_world.get_state_size()
            output_size = self.sim_world.get_move_size()
            save_path = self.weights_path + str(i)
            network = ActorNetwork(input_size, output_size, self.sim_world,
                                   save_path, self.network_layer_sizes,
                                   self.network_layer_acts)
            network.load_weights()
            self.policies.append(network)

    def run(self):
        """
        Runs a tournament.
        """
        for i in range(len(self.policies) - 1):
            for j in range(i + 1, len(self.policies)):
                self.play_one_match(i, j)
        print(f"Wins for each agent was: {self.policies_win_count}")
        self.plot_policies_win_count()

    def plot_policies_win_count(self):
        """
        Plots the win counts for each policy/agent.
        """
        plt.bar(np.arange(0, len(self.policies_win_count)),
                self.policies_win_count)
        plt.show()

    def play_one_match(self, index_a, index_b):
        """
        Plays one match between player at index_a and player at index_b,
        two games where each player gets to start once.
        """
        self.play_one_game(index_a, index_b)
        self.play_one_game(index_a, index_b)
        self.play_one_game(index_b, index_a)
        self.play_one_game(index_b, index_a)

    def play_one_game(self, index_0, index_1):
        """
        Play one hex game between two policies.
        """
        player_0 = self.policies[index_0]
        player_1 = self.policies[index_1]

        state = self.sim_world.get_initial_state()
        while True:
            action = player_0.propose_action(state)
            state = self.sim_world.get_child_state(state, action)
            final = self.sim_world.state_is_final(state)
            if final:
                self.policies_win_count[index_0] += 1
                return
            action = player_1.propose_action(state)
            state = self.sim_world.get_child_state(state, action)
            final = self.sim_world.state_is_final(state)
            if final:
                self.policies_win_count[index_1] += 1
                return
