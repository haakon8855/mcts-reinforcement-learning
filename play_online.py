"""Haakon8855"""

from actor_network import ActorNetwork
from game_hex import GameHex
from client.ActorClient import ActorClient

board_size = 7
sim_world = GameHex(board_size)
save_path = "model/actor_7x7_1_100/" + sim_world.identifier

input_size = sim_world.get_state_size()
output_size = sim_world.get_move_size()
actor = ActorNetwork(input_size, output_size, sim_world, save_path)
weights_loaded = actor.load_weights(save_count=15)
if not weights_loaded:
    raise FileNotFoundError("Error: Weights not loaded!!")


class MyClient(ActorClient):
    """
    test
    """

    def handle_get_action(self, state):
        """
        sdfjlsdk
        """
        formatted_state = GameHex.get_correct_state_from_oht_state(state)
        oh_action = actor.propose_action(formatted_state)
        row, col = GameHex.get_row_and_col_from_oh_action(
            oh_action, board_size)
        return int(row), int(col)


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth="7b50427476a6496f9b35ef03b1f0af1d", qualify=False)
    client.run()
