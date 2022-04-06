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
                 optimizer_str: str,
                 learning_rate: float = 0.003):
        self.board = board  # Sim world object
        self.save_path = save_path  # Path to save weights to
        self.layer_sizes = layer_sizes  # Size of each layer
        self.layer_acts = layer_acts  # Activation function of each layer
        self.optimizer_str = optimizer_str  # Optimizer to use during training
        self.learning_rate = learning_rate  # Learning rate during training
        self.save_count = 0

        # Initialize the network according to the config
        self.model = ks.models.Sequential()
        self.model.add(ks.layers.Input(shape=input_size))
        self.model.add(ks.layers.Flatten())
        for size, act in zip(layer_sizes, layer_acts):
            if act not in ActorNetwork.legal_activation_funcs:
                act = ActorNetwork.legal_activation_funcs[0]
            self.model.add(ks.layers.Dense(size, activation=act))
        self.model.add(ks.layers.Dense(output_size, activation='softmax'))

        # Create a lite model instance later
        self.lite_model = None
        self.compile_network()

    def compile_network(self):
        """
        Compiles the network and adds an optimizer and a learning rate.
        """
        # Get correct optimizer according to config
        if self.optimizer_str == "sgd":
            optimizer = ks.optimizers.SGD(self.learning_rate)
        elif self.optimizer_str == "rmsprop":
            optimizer = ks.optimizers.RMSprop(self.learning_rate)
        elif self.optimizer_str == "adagrad":
            optimizer = ks.optimizers.Adagrad(self.learning_rate)
        else:
            optimizer = ks.optimizers.Adam(self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=ks.losses.CategoricalCrossentropy(),
        )
        # Create the lite model from the network, used during prediction
        self.lite_model = LiteModel.from_keras_model(self.model)

    def propose_action(self, state, get_distribution=False, epsilon=0):
        """
        Returns a proposed action given a state. Can also return the
        distribution output from the network corresponding to the confidence
        the network has in each action.
        Epsilon (exploration) parameter defines the exploration level and
        chooses an action at random from the legal actions if a random number
        is below epsilon. Epsilon = 0 means no exploration and is thus a
        strictly greedy strategy.
        """
        # Get the current legal actions
        legal_actions = self.board.get_legal_actions(state)
        # Combine each one-hot encoded acition into one vector
        legal_actions_filter = np.array(legal_actions).sum(axis=0)
        # Run the state through the network to get a distribution of which
        # action to choose.
        # To facilitate exploration, a very small number is added to the
        # distribution in order to be able to choose an action at random
        # between all the legal moves available.
        state_as_np = np.array(state).reshape(1, -1)
        proposed_action_distribution = (self.lite_model.predict(state_as_np) +
                                        0.00001) * legal_actions_filter
        if np.random.random() < epsilon:
            # Choose the action randomly with uniform probability.
            uniform_distribution = proposed_action_distribution.copy()[0]
            uniform_distribution[uniform_distribution > 0] = 1
            uniform_distribution = uniform_distribution / uniform_distribution.sum(
            )
            proposed_action_num = np.random.choice(len(uniform_distribution),
                                                   1,
                                                   p=uniform_distribution)[0]
        else:
            # Choose the action with the highest number in the output from the
            # network.
            proposed_action_num = self.board.get_action_num_from_one_hot(
                proposed_action_distribution)
        # Return the action as a one-hot encoded action
        proposed_action = self.board.get_one_hot_action(proposed_action_num)
        if get_distribution:
            return proposed_action, proposed_action_distribution
        return proposed_action

    def fit(self, train_x, train_y, epochs):
        """
        Trains the network on the provided cases and creates a new lite model
        after training is finished.
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
        Attempts to load weights from file. Returns True if weights
        were loaded successfully. Creates a new lite model after weights have
        been loaded.
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
