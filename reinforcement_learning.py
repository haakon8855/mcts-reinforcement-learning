"""Haakon8855"""

import numpy as np

from monte_carlo_ts import MonteCarloTreeSearch
from game_nim import GameNim
from actor_network import ActorNetwork


class ReinforcementLearner():
    """
    A reinforcement learning algorithm implementing Monte Carlo Tree Search
    (MCTS) to train the default policy (in this case an ANN).
    """

    def __init__(self, num_games: int = 40, save_interval: int = 5):
        self.num_games = num_games
        self.save_interval = save_interval
        self.rbuf_distributions = []
        self.rbuf_states = []
        self.epsilon = 0.1

        # simworld
        self.num_pieces = 14
        self.max_take = 2
        self.sim_world = GameNim(num_pieces=self.num_pieces,
                                 max_take=self.max_take)

        self.actor_network = None
        self.mcts = None
        self.initialize_actor_network()
        self.initialize_mcts()

    def initialize_actor_network(self):
        """
        Initializes the actor network with the correct input
        and output parameters.
        """
        input_size = self.sim_world.get_state_size()
        output_size = self.sim_world.get_move_size()
        self.actor_network = ActorNetwork(input_size, output_size,
                                          self.sim_world)

    def initialize_mcts(self):
        """
        Initializes monte carlo tree search with the correct parameters.
        """
        self.mcts = MonteCarloTreeSearch(board=self.sim_world,
                                         default_policy=self.actor_network)

    def run(self):
        """
        Runs the traning algorithm to train the default policy neural network
        mapping states to actions.
        """
        # Clear replay buffer RBUF
        self.rbuf_distributions = []
        self.rbuf_states = []
        # Randomly init ANET - Done
        # for game in num_games
        for i in range(self.num_games):
            # s_init <- starting_board_state
            state = self.sim_world.get_initial_state()
            self.mcts.initialize_variables()
            while not self.sim_world.state_is_final(state):
                # Initialize mcts to a single root which represents s_init
                # and run a simulated game from the root state.
                action, distribution = self.mcts.mc_tree_search(state)
                # Append distribution to RBUF
                self.rbuf_distributions.append(distribution)
                self.rbuf_states.append(state)
                # Choose actual move from D
                chosen_action_index = np.argmax(distribution) + 1
                if np.random.random() < self.epsilon:
                    # Choose one random legal action
                    # Copy distribution
                    random_distribution = distribution.copy()
                    # Set all probabilities larger than 0 to 1
                    random_distribution[random_distribution > 0] = 1
                    # Normalize vector to choose uniformly between legal actions
                    random_distribution = random_distribution / random_distribution.sum(
                    )
                    chosen_action_index = np.random.choice(
                        len(random_distribution), 1,
                        p=random_distribution)[0] + 1
                action = self.sim_world.get_one_hot_action(chosen_action_index)
                # Perform chosen action
                state = self.sim_world.get_child_state(state, action)
            # Train ANET on random minibatch of cases from RBUF
            self.train_actor_network()
            if i % (self.num_games // self.save_interval) == 0:
                # TODO: If i % (self.num_games//self.save_interval) == 0: save weights
                pass

    def train_actor_network(self):
        """
        Trains the actor network on a minibatch of cases from RBUF.
        """
        self.actor_network.fit(train_x=np.array(self.rbuf_states),
                               train_y=np.array(self.rbuf_distributions),
                               epochs=10)

    def test_nim(self):
        """
        Tests the trained nim policy.
        """
        for i in range(1, self.num_pieces + 1):
            state = self.sim_world.get_one_hot_state((i, 0))
            # Print state
            num_state = self.sim_world.get_num_discs_from_one_hot(state)
            print(f"State: {num_state}")
            # Find move
            action, distr = self.actor_network.propose_action(
                state, get_distribution=True)
            # Print move
            print(f"Proposed action: {action}")
            print(f"Proposed action distribution: {distr}")


def main():
    """
    Main function for running this python script.
    """
    reinforcement_learner = ReinforcementLearner()
    reinforcement_learner.run()
    reinforcement_learner.test_nim()


if __name__ == "__main__":
    main()
