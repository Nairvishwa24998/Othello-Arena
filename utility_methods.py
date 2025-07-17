import math

from tictactoe_variant import Tictactoe


def prompt_user_fresh_game_or_custom_position():
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
            launch_fresh_game_with_user_config()
        else:
            launch_game_from_pre_defined_position()



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

# method to get user requirements
# we can use it for both start new game and set custom position
# Also add flag to choose level of AI agent- Alpha Beta, MCTS, MCTS with Neural Net
def get_user_requirements():
    user_requirements = {
        "vs_human" : True,
        "ai_player_code": -1,
        "opponent": choose_opponent_type(),
        "board_size": -1,
    }
    # AI opponent case
    if user_requirements["opponent"] == 1:
        user_requirements["vs_human"] = False
        user_requirements["ai_player_code"] = choose_play_order()
    return user_requirements

# to get total moves count by board state
# needed since our game order logic is reliant on total moves played
def get_move_count_from_position(board_positions):
    move_count_by_piece = {
        "X": 0,
        "O": 0
    }
    for row in range(len(board_positions)):
        for column in range(len(board_positions)):
            current_move = board_positions[row][column]
            if current_move in ["X", "O"]:
                move_count_by_piece[current_move] += 1
    return sum(move_count_by_piece.values())



def launch_game_from_pre_defined_position():
    user_req = get_user_requirements()
    ai_player_code = user_req["ai_player_code"]
    vs_human = user_req["vs_human"]
    grid = None
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

        grid = [pieces[i:i + root] for i in range(0, n, root)]
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

    tictactoe = Tictactoe(size=user_req["board_size"], win_length=user_req["board_size"], vs_human=vs_human, ai_player_code=ai_player_code)
    tictactoe.board = grid
    tictactoe.total_moves = get_move_count_from_position(grid)
    if tictactoe.detect_win_loss() is not None:
        print("That position is already a finished game – please enter another.")
        return
    tictactoe.run_game()


# Launch game with user set configuration
def launch_fresh_game_with_user_config():
    board_size = obtain_desired_board_size()
    user_requirements = get_user_requirements()
    user_requirements["board_size"] = board_size
    vs_human = user_requirements["vs_human"]
    ai_player_code = user_requirements["ai_player_code"]
    # In human case, play order doesn't really matter since current implementation makes sure the signs alternate
    tictactoe = Tictactoe(size=board_size, win_length=board_size, vs_human=vs_human, ai_player_code=ai_player_code)
    tictactoe.run_game()


prompt_user_fresh_game_or_custom_position()