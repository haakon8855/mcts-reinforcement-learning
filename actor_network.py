"""Haakon8855"""

from tensorflow import keras as ks


class ActorNetwork:
    """
    Actor network for providing the default policy of the agent.
    """

    def __init__(self, input_size: int, output_size: int):
        self.model = ks.models.Sequential([
            ks.layers.Input(shape=input_size),
            ks.layers.Flatten(),
            ks.layers.Dense(300, activation='relu'),
            ks.layers.Dense(300, activation='relu'),
            ks.layers.Dense(output_size, activation='softmax'),
        ])

    def propose_action(self, state):
        """
        Returns a proposed action given a state.
        """
        return self.model(state)

    def fit(self, train_x, train_y, epochs, batch_size):
        """
        Trains the network on a minibatch of cases.
        """
        self.model.fit(train_x, train_y, epochs, batch_size)
