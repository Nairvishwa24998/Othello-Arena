import numpy as np

from constant_strings import ALPHA_BETA_PRUNING
from utility_methods import setup_tictactoe_instance_for_simulations

# self-play bot would be in charge of invoking simulations
# for MCTS, alpha-beta, MCTS+NN
class SelfPlayBot:

    def __init__(self):
        # trying to keep records of game runs for each board size
        self.game_info = {
            2 : 0,
            3 : 0,
            4 : 0,
            5 : 0,
            6 : 0,
            7 : 0
        }
        self.training_data = []

    def save_training_data(self, filename):
        states = []
        policies = []
        values = []
        for entry in self.training_data:
            # mapping to ASCII values since it is memory efficient
            states.append([c for c in entry["state"]])  # Convert string to int list
            policies.append(entry["policy"])
            values.append(entry["value"])

        np.savez_compressed(filename + ".npz",
                            states=np.array(states),
                            policies=np.array(policies, dtype=np.float32),
                            values=np.array(values, dtype=np.int8))

    # just in case we wish to clear the data
    def reset_training_data(self):
        self.training_data = []

    def generate_file_names(self, size):
        base_filename =  "game_data_board_size"
        return base_filename + str(size)

    def run_simulations(self, size, ai_type):
        count = 0
        win_X = 0
        win_O = 0
        draws = 0
        while count < 10000:
            tictactoe = setup_tictactoe_instance_for_simulations(size = size,ai_type = ai_type)
            result = tictactoe.run_game()  # This sets tictactoe.match_result
            if result == 1:
                win_X += 1
            if result == 0:
                draws += 1
            if result == -1:
                win_O += 1
            for index, (state, policy) in enumerate(tictactoe.move_list):
                # value is from AI's perspective: match_result is 1 (X wins), 0 (draw), -1 (O wins)
                effective_result = result if index % 2 == 0 else -result
                data = {
                    "game_id": count,
                    "move_id": index,
                    "state": state,
                    "policy": policy.tolist(),
                    "value": effective_result
                }
                self.training_data.append(data)
            count += 1
        generated_file_name = self.generate_file_names(size)
        self.save_training_data(generated_file_name)
        print(f" Simulation Complete for board size {size} × {size}")
        print(f" Wins by X: {win_X}")
        print(f" Wins by O: {win_O}")
        print(f" Draws: {draws}")

        generated_file_name = self.generate_file_names(size)
        self.save_training_data(generated_file_name)
        print(f" Training data stored in file: {generated_file_name}")



if __name__ == "__main__":
    bot = SelfPlayBot()
    bot.run_simulations(4, ALPHA_BETA_PRUNING)




