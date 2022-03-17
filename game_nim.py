"""Haakon8855"""


class GameNim:
    """
    Implementation of the NIM game as a simworld.
    """

    def __init__(self, num_pieces: int = 10, max_take: int = 2):
        self.num_pieces = num_pieces  # N, number of pieces on the board
        self.max_take = max_take  # K, maximium amount of pieces allowed to take

    def make_move(self, state: int, action: int):
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

    def __str__(self):
        return f"N = {self.num_pieces}, K = {self.max_take}"


def main():
    """
    Main function for running this python script.
    """
    simworld = GameNim(num_pieces=10, max_take=2)
    print(simworld)
    print(simworld.make_move(10, 2))
    print(simworld.make_move(10, 2))


if __name__ == "__main__":
    main()
