"""Haakon8855"""

import numpy as np
from tensorflow import keras as ks


class ActorNetwork:
    """
    Actor network for providing the default policy of the agent.
    """

    def __init__(self, input_size: int, output_size: int, board):
        self.board = board
        self.learning_rate = 0.003
        self.model = ks.models.Sequential([
            ks.layers.Input(shape=input_size),
            ks.layers.Flatten(),
            ks.layers.Dense(300, activation='relu'),
            ks.layers.Dense(300, activation='relu'),
            ks.layers.Dense(output_size, activation='softmax'),
        ])
        self.compile_network()

    def compile_network(self):
        """
        Compiles the network and adds an optimizer and a learning rate.
        """
        self.model.compile(
            optimizer=ks.optimizers.Adam(self.learning_rate),
            loss=ks.losses.CategoricalCrossentropy(),
        )

    def propose_action(self, state):
        """
        Returns a proposed action given a state.
        """
        legal_actions = self.board.get_legal_actions(state)
        legal_actions_filter = np.array(legal_actions).sum(axis=0)
        state_as_np = np.array(state).reshape(1, -1)
        proposed_action_distribution = self.model(
            state_as_np).numpy() * legal_actions_filter
        proposed_action_num = self.board.get_action_num_from_one_hot(
            proposed_action_distribution)
        proposed_action = self.board.get_one_hot_action(proposed_action_num)
        return proposed_action

    def fit(self, train_x, train_y, epochs):
        """
        Trains the network on a minibatch of cases.
        """
        self.model.fit(train_x, train_y, epochs)
