import numpy as np

from common_utils import validate_bot_play_inp_config
from constant_strings import ALPHA_BETA_PRUNING, MCTS, MCTS_NN
from utility_methods import setup_tictactoe_instance_for_training_simulations, setup_tictactoe_instance_for_bot_matches


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
        self.bot_contest_data = []

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

    # just in case we wish to clear the data
    def reset_bot_contest_data(self):
        self.bot_contest_data = []

    def generate_file_names(self, size):
        base_filename =  "game_data_board_size"
        return base_filename + str(size)

    def run_simulations(self, size, ai_type):
        count = 0
        win_X = 0
        win_O = 0
        draws = 0
        while count < 10000:
            tictactoe = setup_tictactoe_instance_for_training_simulations(size = size, ai_type = ai_type)
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


    def run_bot_v_bot_matches(
            self,
            ai_player_1: str,
            ai_player_2: str,
            rounds: int = 100,
            board_size: int = 3,
    ):
        validate_bot_play_inp_config(ai_player_1, ai_player_2, rounds, board_size)
        # clear any previous contest records
        self.reset_bot_contest_data()
        wins_p1 = wins_p2 = draws = 0
        for game_number in range(1, rounds + 1):
            game = setup_tictactoe_instance_for_bot_matches(board_size, ai_player_1)
            while True:
                turn = game.determine_player_turn()
                # player 1 plays
                if turn == 0:
                    game.ai_player_code = 0
                    game.set_AI_type(ai_player_1)
                    game.make_ai_move(ai_player_1)
                # player_2 plays
                else:
                    game.ai_player_code = 1
                    game.set_AI_type(ai_player_2)
                    game.make_ai_move(ai_player_2)

                outcome = game.detect_win_loss()
                if outcome is not None:
                    break
            # printing outcomes from player 1's perspective
            if outcome == 1:
                wins_p1 += 1
                result_str = f"{ai_player_1} wins"
            elif outcome == -1:
                wins_p2 += 1
                result_str = f"{ai_player_2} wins"
            else:
                draws += 1
                result_str = "Draw"

            # storing into a potential log file
            self.bot_contest_data.append(f"Round Number - {game_number}: {result_str}")

        print(f"total_rounds = {rounds}")
        print(f"{ai_player_1}-wins = {wins_p1}")
        print(f"{ai_player_2}-wins = {wins_p2}")
        print(f"draws = {draws}")



if __name__ == "__main__":
    bot = SelfPlayBot()
    # commented out for testing purposes
    # bot.run_simulations(4, ALPHA_BETA_PRUNING)
    bot.run_bot_v_bot_matches(ai_player_1=MCTS_NN, ai_player_2=ALPHA_BETA_PRUNING, rounds=5, board_size=4)




