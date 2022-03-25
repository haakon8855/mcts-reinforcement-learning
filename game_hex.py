"""Haakon8855"""

import numpy as np


class GameHex:
    """
    Implementation of the HEX game as a simworld.
    """

    def __init__(self, board_size):
        self.board_size = board_size
        self.neighbor_offsets = np.array([
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
        ])
        self.identifier = "hex"

    def get_initial_state(self):
        """
        Returns the initial state of the game.
        State is represented as a flattened array of shape (k, k, 2) with the
        pid represented as [x, x] appended to it.
        """
        board = np.zeros((self.board_size, self.board_size, 2))
        pid = [1.0, 0.0]
        return tuple(board.flatten()) + tuple(pid)

    def get_state_size(self):
        """
        Returns the length of the state vector, result should be (k*k*2 + 2),
        where k is equal to self.board_size.
        """
        return (self.board_size**2) * 2 + 2

    def get_move_size(self):
        """
        Returns the length of the move vector.
        """
        return self.board_size**2

    def get_legal_actions(self, state):
        """
        Returns all allowed actions from current state.
        """
        legal_actions = []
        board, _ = self.get_board_and_pid_from_state(state)
        board_semiflat = board.copy().reshape((self.board_size**2, 2))
        legal_indexes = board_semiflat.sum(axis=1)
        for i, cell in enumerate(legal_indexes):
            if cell == 0.0:
                action = np.zeros(self.board_size**2)
                action[i] = 1.0
                legal_actions.append(tuple(action))
        return legal_actions

    def get_child_state(self, state, action):
        """
        Makes a move by adding a piece to the position denoted by the action.
        Exceptions:
            Throws ValueError if action is not within legal parameters
        """
        board, pid = self.get_board_and_pid_from_state(state)
        action_index = np.argmax(action)
        board_semiflat = board.copy().reshape((self.board_size**2, 2))
        if board_semiflat[action_index].sum() > 0:
            raise ValueError("""Given action not within legal parameters.
                 Selected square already occupied.""")
        board_semiflat[action_index] = pid
        child_state_board = board_semiflat.reshape(
            (self.board_size, self.board_size, 2))
        chils_state_pid = np.flip(pid)
        return self.get_one_hot_state(child_state_board, chils_state_pid)

    def state_is_final(self, state, get_winner=False):
        """
        Returns a boolean for whether the given state is a goal state or not.
        """
        board = self.get_board_readable(state)
        p0_has_path = self.player_has_path(board, 1)
        board = board.T
        p1_has_path = self.player_has_path(board, 2)
        is_final = p0_has_path or p1_has_path
        if get_winner:
            winner_pid = None
            if is_final:
                winner_pid = [(1.0, 0.0), (0.0, 1.0)][p1_has_path]
            return is_final, winner_pid
        return is_final

    def player_has_path(self, board, player):
        """
        Helper method for self.state_is_final. Returns wheter the given player
        has a path from left to right.
        """
        visited = np.zeros((board.shape))
        stack = []
        start = board[:, 0]
        for i, cell in enumerate(start):
            if cell == player:
                # Visit
                visited[i, 0] = 1
                stack.append((i, 0))
                while len(stack) > 0:
                    current = stack[-1]
                    visited[current] = 1
                    if current[1] == self.board_size - 1:
                        return True
                    neighbor = self.get_unvisited_neighbor_of_player(
                        board, visited, current[0], current[1])
                    if neighbor is None:
                        stack.pop()
                        continue
                    stack.append(neighbor)
        return False

    def get_unvisited_neighbor_of_player(self, board, visited, xpos, ypos):
        """
        Returns the coordinates of any unvisited neighbor of
        the current square.
        """
        player = board[xpos, ypos]
        position = np.array([xpos, ypos])
        for offset in self.neighbor_offsets:
            pos_and_offset = position + offset
            if np.any(pos_and_offset < 0) or np.any(
                    pos_and_offset >= self.board_size):
                continue
            pos_and_offset = tuple(pos_and_offset)
            if board[pos_and_offset] == player and visited[pos_and_offset] == 0:
                return pos_and_offset
        return None

    def p0_to_play(self, state):
        """
        Returns True if the next player is p0, False otherwise.
        """
        _, pid = self.get_board_and_pid_from_state(state)
        return list(pid) == [1, 0]

    def winner_is_p0(self, state):
        """
        Return 1 if the winner of this game is player 1, -1 otherwise.
        """
        return [-1, 1][self.state_is_final(state)
                       and not self.p0_to_play(state)]

    def get_one_hot_state(self, board, pid):
        """
        Returns a one-hot encoded vector of the state given the state.
        """
        return tuple(board.flatten()) + tuple(pid)

    def get_one_hot_action(self, action_num):
        """
        Returns the given action as a one-hot encoded vector.
        """
        action_oh = np.zeros(self.board_size**2)
        action_oh[action_num] = 1
        return tuple(action_oh)

    def get_action_num_from_one_hot(self, action_oh):
        """
        Returns the action as a number given a one-hot encoded action.
        """
        return np.argmax(action_oh)

    def get_board_and_pid_from_state(self, state):
        """
        Returns the board and pid as numpy arrays given a gamestate.
        """
        board = np.array(state[:-2]).reshape(
            (self.board_size, self.board_size, 2))
        pid = np.array(state[-2:])
        return board, pid

    def get_board_readable(self, state):
        """
        Returns a string of the board in readable format.
        """
        board, _ = self.get_board_and_pid_from_state(state)
        board = board.reshape((self.board_size**2, 2))
        p0_pieces = ((board == (1.0, 0.0)).all(axis=1))
        p1_pieces = ((board == (0.0, 1.0)).all(axis=1))
        board = np.zeros(self.board_size**2)
        board[p0_pieces] = 1
        board[p1_pieces] = 2
        return board.reshape((self.board_size, self.board_size))

    def __str__(self):
        return f"Board size is {self.board_size}x{self.board_size}."


def main():
    """
    Main function for running this python script.
    """
    simworld = GameHex(4)
    state = simworld.get_initial_state()

    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[4])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[0])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[0])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[12])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[3])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[7])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[1])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[-1])
    legal_actions = simworld.get_legal_actions(state)
    state = simworld.get_child_state(state, legal_actions[1])

    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[4])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[1])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[5])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[3])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[6])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[5])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[0])
    # legal_actions = simworld.get_legal_actions(state)
    # state = simworld.get_child_state(state, legal_actions[6])

    board_str = simworld.get_board_readable(state)
    print(f"Board: \n{board_str}\n")
    print(f"Won? \n{simworld.state_is_final(state, get_winner=True)}\n")


if __name__ == "__main__":
    main()
