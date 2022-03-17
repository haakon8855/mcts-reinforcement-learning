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
        """
        return self.num_pieces

    def get_legal_actions(self, state: int):
        """
        Returns all allowed actions from current state.
        """
        return list(range(1, min(state, self.max_take) + 1))

    def get_child_state(self, state: int, action: int):
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
        if action > state:
            raise ValueError(f"""Given action not within legal parameters.
                 Must be greater than 1 and less than the current number 
                 of pieces ({self.max_take})""")
        return state - action

    def get_all_child_states(self, state: int):
        """
        Returns all child states of the given state.
        TODO Might return action to reach each state later
        """
        actions = self.get_legal_actions(state)
        action_state_pairs = []
        for action in actions:
            action_state_pairs.append(
                (action, self.get_child_state(state, action)))

    def state_is_goal_state(self, state: int):
        """
        Returns a boolean for whether the given state is a goal state or not.
        """
        return state == 0

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
