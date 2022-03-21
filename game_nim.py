"""Haakon8855"""


class GameNim:
    """
    Implementation of the NIM game as a simworld.
    """

    def __init__(self, num_pieces: int = 10, max_take: int = 2):
        self.num_pieces = num_pieces  # N, number of pieces on the board
        self.max_take = max_take  # K, maximium amount of pieces allowed to take

    def get_initial_state(self):
        """
        Returns the initial state of the game.
        State is represented as (number of remaining pieces, pid of next player to move (0 or 1))
        """
        return (self.num_pieces, 0)

    def get_state_size(self):
        """
        Returns the length of the state vector.
        """
        return len(self.get_initial_state())

    def get_move_size(self):
        """
        Returns the length of the move vector.
        """
        return 1

    def get_legal_actions(self, state):
        """
        Returns all allowed actions from current state.
        """
        return list(range(1, min(state[0], self.max_take) + 1))

    def get_child_state(self, state, action: int):
        """
        Makes a move by removing the specified amount of pieces from the board.
        Parameters:
            int action: Number of pieces to remove from the board
        Exceptions:
            Throws ValueError if action is not within legal parameters
        """
        if action < 1 or action > self.max_take:
            raise ValueError(f"""Given action not within legal parameters.
                 Must be greater than 1 and less than the maximium allowed
                 pieces to take ({self.max_take})""")
        if action > state[0]:
            raise ValueError(f"""Given action not within legal parameters.
                 Must be greater than 1 and less than the current number 
                 of pieces ({self.max_take})""")
        child_state_pieces = state[0] - action
        child_state_pid = 1 - state[1]
        return (child_state_pieces, child_state_pid)

    def get_all_child_states(self, state):
        """
        Returns all child states and the action to reach it from the given state.
        """
        actions = self.get_legal_actions(state)
        action_state_pairs = []
        for action in actions:
            action_state_pairs.append(
                (action, self.get_child_state(state, action)))

    def state_is_final(self, state):
        """
        Returns a boolean for whether the given state is a goal state or not.
        """
        return state[0] == 0

    def pid_to_play(self, state, pid: int):
        """
        Returns True if the next player is pid, False otherwise.
        """
        return state[1] == pid

    def winner_is_opponent(self, state, pid: int):
        """
        Return True if the winner of this game is player 2, False otherwise.
        """
        return self.state_is_final(state) and not self.pid_to_play(state, pid)

    def __str__(self):
        return f"N = {self.num_pieces}, K = {self.max_take}"


def main():
    """
    Main function for running this python script.
    """
    simworld = GameNim(num_pieces=10, max_take=4)
    print(simworld)
    state = simworld.get_initial_state()
    print(simworld.get_legal_actions(state))
    print(simworld.get_legal_actions(10))


if __name__ == "__main__":
    main()
