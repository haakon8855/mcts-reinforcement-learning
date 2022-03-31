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
        self.policies_win_count = [0] * ReinforcementLearner.num_policies
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
        for i in range(len(self.policies) - 1):
            for j in range(i + 1, len(self.policies)):
                self.play_one_match(i, j)
        print(f"Wins for each agent was: {self.policies_win_count}")

    def play_one_match(self, index_a, index_b):
        """
        Plays one match between player at index_a and player at index_b,
        two games where each player gets to start once.
        """
        player_a = self.policies[index_a]
        player_b = self.policies[index_b]
        if self.play_one_game(player_a, player_b) == 0:
            self.policies_win_count[index_a] += 1
        else:
            self.policies_win_count[index_b] += 1
        if self.play_one_game(player_b, player_a) == 0:
            self.policies_win_count[index_b] += 1
        else:
            self.policies_win_count[index_a] += 1

    def play_one_game(self, player0, player1):
        """
        Play one hex game between two policies.
        """
        state = self.sim_world.get_initial_state()
        while True:
            action = player0.propose_action(state)
            state = self.sim_world.get_child_state(state, action)
            final = self.sim_world.state_is_final(state)
            if final:
                return 0
            action = player1.propose_action(state)
            state = self.sim_world.get_child_state(state, action)
            final = self.sim_world.state_is_final(state)
            if final:
                return 1


def main():
    """
    Main function for running this python script.
    """
    board_size = 4
    sim_world = GameHex(board_size)
    topp = Tournament(sim_world)
    topp.run()


if __name__ == "__main__":
    main()
