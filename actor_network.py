"""Haakon8855"""

import numpy as np
from tensorflow import keras as ks
from lite_model import LiteModel


class ActorNetwork:
    """
    Actor network for providing the default policy of the agent.
    """
    legal_activation_funcs = ['relu', 'sigmoid', 'tanh', 'linear']

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 board,
                 save_path: str,
                 layer_sizes: list,
                 layer_acts: list,
                 learning_rate: float = 0.003):
        self.board = board
        self.save_path = save_path
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.learning_rate = learning_rate
        self.save_count = 0

        self.model = ks.models.Sequential()
        self.model.add(ks.layers.Input(shape=input_size))
        self.model.add(ks.layers.Flatten())
        for size, act in zip(layer_sizes, layer_acts):
            if act not in ActorNetwork.legal_activation_funcs:
                act = ActorNetwork.legal_activation_funcs[0]
            self.model.add(ks.layers.Dense(size, activation=act))
        self.model.add(ks.layers.Dense(output_size, activation='softmax'))

        self.lite_model = None
        self.compile_network()

    def compile_network(self):
        """
        Compiles the network and adds an optimizer and a learning rate.
        """
        self.model.compile(
            optimizer=ks.optimizers.Adagrad(self.learning_rate),
            loss=ks.losses.CategoricalCrossentropy(),
        )
        self.lite_model = LiteModel.from_keras_model(self.model)

    def propose_action(self, state, get_distribution=False, epsilon=0):
        """
        Returns a proposed action given a state.
        """
        legal_actions = self.board.get_legal_actions(state)
        legal_actions_filter = np.array(legal_actions).sum(axis=0)
        state_as_np = np.array(state).reshape(1, -1)
        proposed_action_distribution = (self.lite_model.predict(state_as_np) +
                                        0.00001) * legal_actions_filter
        if np.random.random() < epsilon:
            uniform_distribution = proposed_action_distribution.copy()[0]
            uniform_distribution[uniform_distribution > 0] = 1
            uniform_distribution = uniform_distribution / uniform_distribution.sum(
            )
            proposed_action_num = np.random.choice(len(uniform_distribution),
                                                   1,
                                                   p=uniform_distribution)[0]
        else:
            proposed_action_num = self.board.get_action_num_from_one_hot(
                proposed_action_distribution)
        proposed_action = self.board.get_one_hot_action(proposed_action_num)
        if get_distribution:
            return proposed_action, proposed_action_distribution
        return proposed_action

    def fit(self, train_x, train_y, epochs):
        """
        Trains the network on a minibatch of cases.
        """
        self.model.fit(train_x, train_y, epochs=epochs)
        self.lite_model = LiteModel.from_keras_model(self.model)

    def save_weights(self, save_count):
        """
        Saves the weights of the network to a file for loading at a later time.
        """
        self.model.save_weights(filepath=self.save_path + str(save_count))
        print("Saved weights to file")

    def load_weights(self, save_count=None):
        """
        Attempts to load weights from file. Returns True if successful
        """
        try:
            if save_count is None:
                self.model.load_weights(filepath=self.save_path)
            else:
                self.model.load_weights(filepath=self.save_path +
                                        str(save_count))
            print("Read weights successfully from file")
            self.lite_model = LiteModel.from_keras_model(self.model)
            return True
        except:  # pylint: disable=bare-except
            print("Could not read weights from file")
            return False
