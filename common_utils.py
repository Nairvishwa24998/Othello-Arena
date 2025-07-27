# to keep values in range [-1,1]
from constant_strings import ALPHA_BETA_PRUNING, MCTS, MCTS_NN


def clamp(value):
    if value == 0.0:
        return value
    elif value > 0.0:
        return min(1.0, value)
    else:
        return max(-1.0, value)

def validate_bot_play_inp_config(ai_player_1, ai_player_2, rounds, board_size):
    valid_ai = {ALPHA_BETA_PRUNING, MCTS, MCTS_NN}
    if ai_player_1 not in valid_ai or ai_player_2 not in valid_ai:
        raise ValueError(f"ai_player_* must be one of {valid_ai}")
    if not (2 <= board_size <= 7):
        raise ValueError("board_size must be between 2 to 7, both inclusive")
    if rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    # only happens if none of the validations are broken
    return True