"""Haakon8855"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
        self.player_to_begin = 0
        self.identifier = "hex"

    def get_initial_state(self, randomize_start=False):
        """
        Returns the initial state of the game.
        State is represented as a flattened array of shape (k, k, 2) with the
        pid represented as [x, x] appended to it.
        """
        board = np.zeros((self.board_size, self.board_size, 2))
        if randomize_start:
            # Randomize the starting player
            pid = [[1.0, 0.0], [0.0, 1.0]][self.player_to_begin]
        else:
            # Black starts
            pid = [1.0, 0.0]
        self.player_to_begin = 1 - self.player_to_begin
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
        Returns all allowed actions from current state. State parameter
        is one-hot encoded.
        """
        legal_actions = []
        board, _ = self.get_board_and_pid_from_state(state)
        # Board is one-hot encoded and each cell consists of two values
        board_semiflat = board.copy().reshape((self.board_size**2, 2))
        # Sum these two values for each cell. All cells summing to 0 are emtpy
        # and that action is thus allowed.
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
        # Get matrix-like board representation and pid from OH-encoded state.
        board, pid = self.get_board_and_pid_from_state(state)
        # Get action number from one-hot encoded action.
        action_index = np.argmax(action)
        board_semiflat = board.copy().reshape((self.board_size**2, 2))
        # Check if chosen action places a piece in an empty cell.
        if board_semiflat[action_index].sum() > 0:
            raise ValueError("""Given action not within legal parameters.
                 Selected square already occupied.""")
        # Place a piece for that player in the appropriate cell.
        board_semiflat[action_index] = pid
        # Reshape the board to matrix-like shape
        child_state_board = board_semiflat.reshape(
            (self.board_size, self.board_size, 2))
        # Change the current player to other pid
        chils_state_pid = np.flip(pid)
        # Return one-hot encoded state
        return self.get_one_hot_state(child_state_board, chils_state_pid)

    def state_is_final(self, state, get_winner=False):
        """
        Returns a boolean for whether the given state is a goal state or not.
        """
        # Get matrix-like readable state representation
        board = self.get_board_readable(state)
        # Check if player 0 has a path from NW to SE
        p0_has_path = self.player_has_path(board, 1)
        # Transpose the board and check if player 1 has a path from NW to SE
        board = board.T
        if p0_has_path:
            # If p0 has a path, p1 cannot have a path
            p1_has_path = False
        else:
            p1_has_path = self.player_has_path(board, 2)
        # If one of the players has a path, the game is over
        is_final = p0_has_path or p1_has_path
        # Returns the result, along with the winner pid if that is asked for
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
        # Run depth-first search on the board starting with each cell in
        # the first column. If a cell in the rightmost column is visited,
        # the given player has a path across the board.
        visited = np.zeros((board.shape))
        stack = []
        start = board[:, 0]
        # For each cell in the first column
        for i, cell in enumerate(start):
            # If the current cell contains a piece with same color as player
            if cell == player:
                # Set cell as visited and add it to the stack
                visited[i, 0] = 1
                stack.append((i, 0))
                while len(stack) > 0:
                    current = stack.pop()
                    visited[current] = 1
                    # If current cell is in the rightmost column, player has won
                    if current[1] == self.board_size - 1:
                        return True
                    # If not, add all unvisited neighbors of current cell to
                    # the stack.
                    neighbors = self.get_unvisited_neighbors(
                        board, visited, current[0], current[1])
                    stack = stack + neighbors
        return False

    def get_unvisited_neighbors(self, board, visited, xpos, ypos):
        """
        Returns the coordinates of any unvisited neighbor of
        the current square.
        """
        # The number of the player to check
        player = board[xpos, ypos]
        # The cell who's neighbors are wanted
        position = np.array([xpos, ypos])
        # Start with all possible neighbors
        neighbors = position + self.neighbor_offsets
        # Remove all neighbors with negative coordinates
        filter1 = (neighbors >= 0).all(axis=1)
        # Remove all neighbors with coordinates larger than game board
        filter2 = (neighbors < self.board_size).all(axis=1)
        neighbors = neighbors[np.logical_and(filter1, filter2)]
        board_vals = board[[neighbors[:, 0]]][np.arange(0, len(neighbors)),
                                              neighbors[:, 1]]
        visited_vals = visited[[neighbors[:, 0]]][np.arange(0, len(neighbors)),
                                                  neighbors[:, 1]]
        # Remove all neighbors with empty cells or cells with pieces from opponent
        filter3 = board_vals == player
        # Remove all neighbors we have already visited
        filter4 = visited_vals == 0
        neighbors = neighbors[np.logical_and(filter3, filter4)]
        return list(map(tuple, neighbors))

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
        Returns the board in readable format. A matrix where empty cells
        are denoted as 0s, black pieces are 1s and red pieces are 2s.
        """
        board, _ = self.get_board_and_pid_from_state(state)
        board = board.reshape((self.board_size**2, 2))
        p0_pieces = ((board == (1.0, 0.0)).all(axis=1))
        p1_pieces = ((board == (0.0, 1.0)).all(axis=1))
        board = np.zeros(self.board_size**2)
        board[p0_pieces] = 1
        board[p1_pieces] = 2
        return board.reshape((self.board_size, self.board_size))

    def show_visible_board(self, state, title: str = ""):
        """
        Shows a visual representation of the board state using
        networkx and matplotlib.
        """
        plt.clf()
        plt.title(title)
        board = self.get_board_readable(state)
        board = board.flatten()
        coords = {}  # Coordinates of each cell
        graph = nx.Graph()  # Init network graph
        neighbor_offsets = [-1, -self.board_size, -self.board_size + 1]
        # Pivot point to rotate all points around
        pivot = ((self.board_size - 1) / 2, (self.board_size - 1) / 2)
        # Angle to rotate each point around pivot
        angle = 5 * np.pi / 4
        possible_colors = ["lightgray", "black", "red"]
        colors = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                # Get index in flattened board representation
                key = i * self.board_size + j
                point = (i, j)
                rotated_point = GameHex.rotate_point(point, pivot, angle)
                # For all cells except the first:
                if key != 0:
                    # Add the path from current cell to neighbors prior to
                    # current in the flattened board representation.
                    for k, offset in enumerate(neighbor_offsets):
                        if key + offset >= 0 and not (
                                k == 0 and key % self.board_size == 0
                        ) and not (k == 2 and
                                   (key + 1) % self.board_size == 0):
                            graph.add_edge(key + offset, key)
                # Add the rotated point to the dict of coordinates
                coords[key] = rotated_point
                color = possible_colors[int(board[key])]
                colors.append(color)

        # Draw the network on the plot and save the plot to file.
        nx.draw(graph, pos=coords, node_color=colors, with_labels=True)
        plt.savefig("figs/graph.png")
        # Clear the plot in order to not interfere with other plots.
        plt.clf()

    @staticmethod
    def rotate_point(point, pivot, angle):
        """
        Rotates a point around another point (rot_point) a given angle.
        """
        sin = np.sin(angle)
        cos = np.cos(angle)
        point = (point[0] - pivot[0], point[1] - pivot[1])
        new_point = (point[0] * cos - point[1] * sin,
                     point[0] * sin + point[1] * cos)
        new_point = (new_point[0] + pivot[0], new_point[1] + pivot[1])
        return new_point

    @staticmethod
    def get_correct_state_from_oht_state(oht_state):
        """
        Returns a state representation in the same representation as used by
        this game manager. (i.e. converts from OHT's state representation
        to this class's state representation)
        """
        player = oht_state[0]
        board = np.array(oht_state[1:])

        oh_board = []
        for cell in board:
            if cell == 0:
                oh_board += [0.0, 0.0]
            elif cell == 1:
                oh_board += [0.0, 1.0]
            else:
                oh_board += [1.0, 0.0]
        if player == 1:
            oh_board += [0.0, 1.0]
        else:
            oh_board += [1.0, 0.0]
        return tuple(oh_board)

    @staticmethod
    def get_row_and_col_from_oh_action(oh_action, board_size):
        """
        Returns the row and column corresponding to the location where the
        piece should be placed according to the given one-hot-encoded action.
        """
        action = np.array(oh_action)
        num = np.where(action == 1.0)[0][0]
        row = num // board_size
        col = num % board_size
        return row, col

    def __str__(self):
        return f"Board size is {self.board_size}x{self.board_size}."
