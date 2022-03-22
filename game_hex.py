"""Haakon8855"""

import numpy as np


class GameHex:
    """
    Implementation of the HEX game as a simworld.
    """

    def __init__(self, board_size):
        self.board_size = board_size

    def get_initial_state(self):
        """
        Returns the initial state of the game.
        State is represented as TODO
        """
        board = np.zeros((self.board_size, self.board_size, 2))
        board = np.arange(0, 32).reshape((self.board_size, self.board_size, 2))
        pid = [1.0, 0.0]
        print(board)
        return tuple(board.flatten()) + tuple(pid)

    def get_state_size(self):
        """
        Returns the length of the state vector.
        """
        # TODO: Implement

    def get_move_size(self):
        """
        Returns the length of the move vector.
        """
        # TODO: Implement

    def get_legal_actions(self, state):
        """
        Returns all allowed actions from current state.
        """
        # TODO: Implement

    def get_child_state(self, state, action):
        """
        Makes a move by TODO
        Exceptions:
            Throws ValueError if action is not within legal parameters
        """
        # TODO: Implement

    def get_all_child_states(self, state):
        """
        Returns all child states and the action to reach it from the given state.
        """
        # TODO: Implement

    def state_is_final(self, state):
        """
        Returns a boolean for whether the given state is a goal state or not.
        """
        # TODO: Implement

    def p0_to_play(self, state):
        """
        Returns True if the next player is pid, False otherwise.
        """
        # TODO: Implement

    def winner_is_p0(self, state):
        """
        Return 1 if the winner of this game is player 1, -1 otherwise.
        """
        return [-1, 1][self.state_is_final(state)
                       and not self.p0_to_play(state)]

    def get_one_hot_state(self, board_and_pid):
        """
        Returns a one-hot encoded vector of the state given the state.
        """
        # TODO: Implement

    def get_one_hot_action(self, action_num):
        """
        Returns the given action as a one-hot encoded vector.
        """
        # TODO: Implement

    def get_board_and_pid_from_state(self, state):
        """
        Returns the board and pid as numpy arrays given a gamestate.
        """
        board = np.array(state[:-2]).reshape(
            (self.board_size, self.board_size, 2))
        pid = np.array(state[-2:])

        return board, pid

    def __str__(self):
        return f"Board size is {self.board_size}x{self.board_size}."


def main():
    """
    Main function for running this python script.
    """
    simworld = GameHex(4)
    print(simworld)
    state = simworld.get_initial_state()
    print(f"State: {state}")
    board, pid = simworld.get_board_and_pid_from_state(state)
    print(f"Board: {board}")
    print(f"PID:   {pid}")
    # print(simworld.get_legal_actions(state))


if __name__ == "__main__":
    main()
