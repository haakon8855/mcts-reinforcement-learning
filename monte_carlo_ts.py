"""Haakon8855"""

import numpy as np

from game_nim import GameNim
from actor_network import ActorNetwork


class MonteCarloTreeSearch:
    """
    Class for running Monte Carlo Tree search on a simworld.
    """

    def __init__(self,
                 board,
                 default_policy,
                 pid: int,
                 simulations: int = 500,
                 default_exp_const: int = 1):
        self.board = board  # Of type simworld
        self.board = GameNim(num_pieces=10, max_take=4)
        self.default_policy = default_policy
        self.default_exp_const = default_exp_const
        self.simulations = simulations  # M-value for number of simulations
        self.state = None
        self.pid = pid
        self.tree = []
        self.heuristic = {}
        self.visit_counts_s = {}
        self.visit_counts_sa = {}

    def mc_tree_search(self, root_state):
        """
        Performs MCTS for self.simulations number of times. Each resulting in
        a new node in the tree being created.
        """
        for _ in range(self.simulations):
            self.simulate(root_state)
        self.state = root_state
        return self.select_action(root_state, 0)

    def simulate(self, root_state):
        """
        Simulates one run of the game from the root state. Walks down the tree
        and performs rollout if a leaf node is reached before the simulated
        game is over.
        """
        self.state = root_state
        visited_states, performed_actions = self.simulate_tree()
        outcome, first_action = self.simulate_default()
        performed_actions.append(first_action)
        self.backup(visited_states, performed_actions, outcome)

    def simulate_tree(self):
        """
        Simulates the walk of moves down the tree itself.
        """
        exploration = self.default_exp_const
        visited_states = []
        performed_actions = []
        while not self.board.state_is_final(self.state):
            state_t = self.state
            visited_states.append(state_t)
            if state_t not in self.tree:
                self.new_node(state_t)
                return visited_states, performed_actions
            action = self.select_action(state_t, exploration)
            performed_actions.append(action)
            self.state = self.board.get_child_state(self.state, action)
        return visited_states, performed_actions

    def simulate_default(self):
        """
        Simulates the rollout of default moves after a leaf node in the tree
        is reached. The default policy is used to determine the moves.
        """
        if not self.board.state_is_final(self.state):
            first_action = self.default_policy.propose_action(self.state)
            self.state = self.board.get_child_state(self.state, first_action)
        while not self.board.state_is_final(self.state):
            action = self.default_policy.propose_action(self.state)
            self.state = self.board.get_child_state(self.state, action)
        return self.board.winner_is_pid(self.state, self.pid), first_action

    def select_action(self, state, exploration):
        """
        Selects an action to perform when walking down the tree. Actions never
        performed before, or actions which have been performed relatively few
        number of times are favoured.
        """
        legal_actions = self.board.get_legal_actions(state)
        action_values = []
        if self.board.pid_to_play(state, self.pid):  # If our turn
            for action in legal_actions:
                action_value = self.heuristic[
                    (state, action)] + exploration * np.sqrt(
                        np.log(self.visit_counts_s[state]) /
                        self.visit_counts_sa[(state, action)])
                action_values.append(action_value)
            chosen_action_index = np.argmax(np.array(action_values))
            chosen_action = legal_actions[chosen_action_index]
        else:
            for action in legal_actions:
                action_value = self.heuristic[
                    (state, action)] - exploration * np.sqrt(
                        np.log(self.visit_counts_s[state]) /
                        self.visit_counts_sa[(state, action)])
                action_values.append(action_value)
            chosen_action_index = np.argmin(np.array(action_values))
            chosen_action = legal_actions[chosen_action_index]
        return chosen_action

    def backup(self, visited_states, performed_actions, outcome):
        """
        Runs the backup algorithm on the network to update the values along
        the nodes and paths based on the game's outcome.
        """
        for stat_act in zip(visited_states, performed_actions):
            self.visit_counts_s[
                stat_act[0]] = self.visit_counts_s[stat_act[0]] + 1
            self.visit_counts_sa[stat_act] = self.visit_counts_sa[stat_act] + 1
            self.heuristic[stat_act] = self.heuristic[stat_act] + (
                outcome -
                self.heuristic[stat_act]) / self.visit_counts_sa[stat_act]

    def new_node(self, state_t):
        """
        Creates a new node in the tree and sets all its values to 0.
        """
        self.tree.append(state_t)
        self.visit_counts_s[state_t] = 0
        legal_actions = self.board.get_legal_actions(state_t)
        for action in legal_actions:
            self.visit_counts_sa[(state_t, action)] = 0
            self.heuristic[(state_t, action)] = 0
