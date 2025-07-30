# to keep values in range [-1,1]
from constant_strings import ALPHA_BETA_PRUNING, MCTS, MCTS_NN, OTHELLO_BOARD_SIZE, MOVE_B, MOVE_W


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
        raise ValueError(f"ai_players must be one of {valid_ai}")
    if not (2 <= board_size <= 7):
        raise ValueError("board_size must be between 2 to 7, both inclusive")
    if rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    # only happens if none of the validations are broken
    return True





def set_starting_othello_board():
    result =  [['.']*OTHELLO_BOARD_SIZE for row in range(OTHELLO_BOARD_SIZE)]
    # placing 4 pieces in the conventional Othello starting position
    result[3][3] = MOVE_W
    result[4][4] = MOVE_W
    result[3][4] = MOVE_B
    result[4][3] = MOVE_B
    return result


