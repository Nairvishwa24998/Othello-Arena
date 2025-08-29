import math
from constant_strings import MOVE_B, MOVE_W, OTHELLO_BOARD_SIZE

from constant_strings import TEMPERATURE_CONTROL_FOR_MAX_RANDOMNESS, ALPHA_BETA_PRUNING, MCTS, MCTS_NN, \
    TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS, GAME_TICTACTOE, GAME_OTHELLO
from othello import Othello
from tictactoe_variant import Tictactoe


def prompt_user_fresh_game_or_custom_position(game_name):
    invalid_response = True
    response = None
    while invalid_response:
        try:
            response = int(input("Would you like to start the game afresh or from a custom position?- Press 0 for the former and 1 for the latter"))
            if response not in [0,1]:
                print("Please enter an integer value of 0 or 1")
            else:
                invalid_response = False
        except ValueError:
            print("Please provide an integer response")
    if invalid_response is False:
        if response == 0:
            if game_name == GAME_TICTACTOE:
                launch_fresh_tictactoe_game_with_user_config()
            # method call prior to this ensures only two game_name options can be passed
            else:
                launch_fresh_othello_game_with_user_config()
        else:
            if game_name == GAME_OTHELLO:
                launch_othello_game_from_pre_defined_position()
            # method call prior to this ensures only two game_name options can be passed
            else:
                launch_tictactoe_game_from_pre_defined_position()

def obtain_desired_board_size():
    board_size = 0
    invalid_board_size = True
    while invalid_board_size:
        try:
            board_size = int(input("Please input a natural number as the input in the range 2 to 7!"))
            if board_size < 2 or board_size > 7:
                print("Please ensure the provided value is a natural number in the range 2 to 7!")
            else:
                invalid_board_size = False
        except ValueError:
            print("Please ensure the provided value is a natural number in the range 2 to 7!")
    return board_size

# whether human opponent or AI
def choose_opponent_type():
    invalid_opponent_type = True
    response = ""
    while invalid_opponent_type:
        try:
            response = int(input("Please choose 0 for human opponent and 1 for AI opponent"))
            if response in [0,1]:
                invalid_opponent_type = False
        except ValueError:
            print("Please enter either 0 or 1!")
    return response

def choose_play_order():
    invalid_play_order = True
    response = ""
    while invalid_play_order:
        try:
            response = int(input("Please choose 0 to make the AI go first and 1 to make the AI go second"))
            if response in [0,1]:
                invalid_play_order = False
        except ValueError:
            print("Please enter either 0 or 1!")
    return response

def choose_and_map_ai_type():
    invalid_ai_type = True
    response = ""
    ai_type_map = {
        0 : ALPHA_BETA_PRUNING,
        1: MCTS,
        2 : MCTS_NN
    }
    while invalid_ai_type:
        try:
            print("0 - Alpha-Beta Pruning AI")
            print("1 - Pure MCTS AI")
            print("2 - MCTS + Neural Network (if available)")
            response = int(input("Please choose a number from 0,1 and 2"))
            if response in [0, 1,2]:
                invalid_ai_type = False
        except ValueError:
            print("Please enter either 0,1,2!")
    return ai_type_map[response]



# method to get user requirements
# we can use it for both start new game and set custom position
# Also add flag to choose level of AI agent - Alpha Beta, MCTS, MCTS with Neural Net
def get_user_requirements():
    user_requirements = {
        "vs_human" : True,
        "ai_player_code": -1,
        "opponent": choose_opponent_type(),
        "board_size": -1,
        # mcts/mcts+nn/alpha-bet
        "ai_type": -1
    }
    # AI opponent case
    if user_requirements["opponent"] == 1:
        user_requirements["vs_human"] = False
        user_requirements["ai_player_code"] = choose_play_order()
        # we only add this when the user is intending to play with AI
        user_requirements["ai_type"] = choose_and_map_ai_type()
    else:
        user_requirements["vs_human"] = True
        # we only add this when the user is intending to play with AI
        user_requirements["ai_type"] = None
        user_requirements["ai_player_code"] = None
    return user_requirements



def launch_tictactoe_game_from_pre_defined_position():
    user_req = get_user_requirements()
    ai_player_code = user_req["ai_player_code"]
    vs_human = user_req["vs_human"]
    ai_type = user_req["ai_type"]
    input_grid = None
    nx = None
    no = None
    while True:
        raw = input("Enter board as comma-sep X/O (· for empty):\n")
        pieces = [s.strip().upper() for s in raw.split(",")]
        n = len(pieces)
        root = int(math.sqrt(n))
        if root * root != n:
            print("Board must be a square!")
            continue
        if not 4 <= n <= 49:
            print("Supported sizes: 2 × 2 to 7 × 7")
            continue
        if any(p not in ("X", "O", ".") for p in pieces):
            print("Only X, O or . are allowed")
            continue

        input_grid = [pieces[i:i + root] for i in range(0, n, root)]
        nx = sum(p == "X" for p in pieces)
        no = sum(p == "O" for p in pieces)
        if abs(nx - no) > 1:
            print("Impossible move count (X/O difference > 1)")
            continue
        if nx > no:
            if ai_player_code == 0:
                print("Usual convention in TicTacToe indicates X should go first.Hence, given the current position AI cannot be allowed to play the second move. Please enter a different one")
                continue
        if nx < no:
            print("Usual convention in TicTacToe indicates X should go first. Hence, the given position is not a valid one since number of Os cannot be greater than the number of 1s. Please enter a different one")
            continue

        if nx == no:
            if ai_player_code == 1:
                print(
                    "Usual convention in TicTacToe indicates O should go second. Hence, given the current position AI cannot be allowed to play the second move. Please enter a different one")
                continue
        user_req["board_size"] = root
        break  # input accepted

    tictactoe = Tictactoe(size=user_req["board_size"], win_length=user_req["board_size"], vs_human=vs_human, ai_player_code=ai_player_code,ai_type = ai_type)
    tictactoe.board = input_grid
    tictactoe.total_moves = tictactoe.get_move_count_from_position(input_grid)
    if tictactoe.detect_win_loss() is not None:
        print("That position is already a finished game – please enter another.")
        return
    tictactoe.run_game()

def launch_othello_game_from_pre_defined_position():
    game = None
    user_req        = get_user_requirements()
    ai_player_code  = user_req["ai_player_code"]
    vs_human        = user_req["vs_human"]
    ai_type         = user_req["ai_type"]

    while True:
        raw = input("Enter 64 pieces as comma-sep B/W (· for empty):\n")
        pieces = [p.strip().upper() for p in raw.split(",")]
        if len(pieces) != 64:
            print("Exactly 64 entries required for an 8×8 board.")
            continue
        if any(p not in (MOVE_B, MOVE_W, ".") for p in pieces):
            print("Only B, W or · are allowed.")
            continue

        grid     = [pieces[i:i + 8] for i in range(0, 64, 8)]
        n_black  = pieces.count(MOVE_B)
        n_white  = pieces.count(MOVE_W)
        if min(n_black, n_white) == 0:
            print("Both colours must be present.")
            continue

        nxt = input("Who plays next? (B/W): ").strip().upper()
        if nxt not in (MOVE_B, MOVE_W):
            print("Please enter B or W.")
            continue
        next_player = 0 if nxt == MOVE_B else 1   # 0 = Black, 1 = White

        # Create a provisional game to validate legality
        game = Othello(
            size=OTHELLO_BOARD_SIZE,                 # <-- fixed
            vs_human=vs_human,
            ai_player_code=ai_player_code,
            ai_type=ai_type,
        )
        game.board       = grid
        game.total_moves = n_black + n_white
        game.last_moved  = 1 - next_player       # so current_player() returns next_player

        if game.detect_win_loss() is not None:
            print("That position is already a finished game – please enter another.")
            continue
        if not game.get_possible_moves(next_player):
            print(f"{nxt} has no legal moves in that position – please re-enter.")
            continue

        break   # position accepted
    game.run_game()



# Launch game with user set configuration
def launch_fresh_tictactoe_game_with_user_config():
    board_size = obtain_desired_board_size()
    user_requirements = get_user_requirements()
    user_requirements["board_size"] = board_size
    vs_human = user_requirements["vs_human"]
    ai_player_code = user_requirements["ai_player_code"]
    ai_type = user_requirements["ai_type"]
    # In human case, play order doesn't really matter since current implementation makes sure the signs alternate
    tictactoe = Tictactoe(size=board_size, win_length=board_size, vs_human=vs_human, ai_player_code=ai_player_code,ai_type=ai_type)
    tictactoe.run_game()

# Launch game with user set configuration
def launch_fresh_othello_game_with_user_config():
    user_requirements = get_user_requirements()
    user_requirements["board_size"] = OTHELLO_BOARD_SIZE
    board_size = OTHELLO_BOARD_SIZE
    vs_human = user_requirements["vs_human"]
    ai_player_code = user_requirements["ai_player_code"]
    ai_type = user_requirements["ai_type"]
    # In human case, play order doesn't really matter since current implementation makes sure the signs alternate
    othello = Othello(size=board_size, vs_human=vs_human, ai_player_code=ai_player_code,ai_type=ai_type)
    othello.run_game()


def setup_tictactoe_instance_for_training_simulations(size, ai_type):
    tictactoe = Tictactoe(size=size, vs_human=False, temperature_control=TEMPERATURE_CONTROL_FOR_MAX_RANDOMNESS)
    # setting the preferred AI type. Will dictate the type of AI used in simulations
    tictactoe.set_AI_type(ai_type)
    # we only need it for alpha beta pruning
    if ai_type == ALPHA_BETA_PRUNING:
        tictactoe.set_temperature_control(TEMPERATURE_CONTROL_FOR_MAX_RANDOMNESS)
    # simulation mode so AI starts with the first move
    tictactoe.ai_player_code = 0
    tictactoe.set_to_simulation_mode()
    return tictactoe


def setup_tictactoe_instance_for_bot_matches(size, first_player_ai_type):
    tictactoe = Tictactoe(size=size, vs_human=False)
    # setting the preferred AI type. Will dictate the type of AI used in simulations
    tictactoe.set_AI_type(first_player_ai_type)
    # this is to demo how well it performs so make it min_randomness
    if first_player_ai_type == ALPHA_BETA_PRUNING:
        tictactoe.set_temperature_control(TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS)
    # simulation mode so the AI starts with the first move
    tictactoe.ai_player_code = 0
    tictactoe.set_to_simulation_mode()
    # suppress board prints for speed/clarity
    tictactoe.logging_mode = False
    return tictactoe


def setup_othello_instance_for_bot_matches(first_player_ai_type):
    othello = Othello(vs_human=False)
    # setting the preferred AI type. Will dictate the type of AI used in simulations
    othello.set_AI_type(first_player_ai_type)
    # this is to demo how well it performs so make it min_randomness
    if first_player_ai_type == ALPHA_BETA_PRUNING:
        othello.set_temperature_control(TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS)
    # simulation mode so the AI starts with the first move
    othello.ai_player_code = 0
    othello.set_to_simulation_mode()
    # suppress board prints for speed/clarity
    othello.logging_mode = False
    return othello

# method to choose a game
def choose_game():
    invalid_game_type = True
    response = ""
    while invalid_game_type:
        try:
            response = int(input("Please choose 0 for Tictactoe and 1 for Othello"))
            if response in [0,1]:
                invalid_game_type = False
        except ValueError:
            print("Please enter either 0 or 1!")
    # we have weeded out all other cases above so only these two can happen
    game_name = GAME_TICTACTOE if response == 0 else GAME_OTHELLO
    return game_name

# method to divert the user into the branch they like
def commence_game_play():
    game_name = choose_game()
    prompt_user_fresh_game_or_custom_position(game_name=game_name)


# Note this is needed otherwise self-play bot won't run
if __name__ == "__main__":
    commence_game_play()


