"""Haakon8855"""

from actor_network import ActorNetwork
from game_hex import GameHex
from client.ActorClient import ActorClient
from time import time

board_size = 7
sim_world = GameHex(board_size)
save_path = "model/actor_7x7_200_500/" + sim_world.identifier

input_size = sim_world.get_state_size()
output_size = sim_world.get_move_size()
layer_sizes = [200, 200, 100]
layer_acts = ['relu', 'relu', 'relu']
actor = ActorNetwork(input_size, output_size, sim_world, save_path,
                     layer_sizes, layer_acts)
weights_loaded = actor.load_weights(save_count=15)
if not weights_loaded:
    raise FileNotFoundError("Error: Weights not loaded!!")


class MyClient(ActorClient):
    """
    ActorClient with custom action handler.
    """

    def handle_get_action(self, state):
        """
        Handles returning an action given the game state.
        """
        starttime = time()
        formatted_state = GameHex.get_correct_state_from_oht_state(state)
        oh_action = actor.propose_action(formatted_state)
        row, col = GameHex.get_row_and_col_from_oh_action(
            oh_action, board_size)
        print(f"Time taken to move: {time() - starttime}")
        return int(row), int(col)


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth="7b50427476a6496f9b35ef03b1f0af1d", qualify=False)
    client.run()
