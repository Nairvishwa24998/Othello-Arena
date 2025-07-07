# get preferred board size
# If not in valid range then keep asking until is
import math


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

# Launch game with user set configuration
def launch_game_with_user_config():
    vs_human = True
    ai_player_code = None
    board_size = obtain_desired_board_size()
    opponent = choose_opponent_type()
    # AI opponent case
    if opponent == 1:
        vs_human = False
        ai_player_code = choose_play_order()
    # In human case, play order doesn't really matter since current implementation makes sure the signs alternate
    tictactoe = Tictactoe(size=board_size, win_length=board_size, vs_human=vs_human, ai_player_code=ai_player_code)
    tictactoe.run_game()


# Let us represent players with player 1 by 0 and player 2 by 1
# default setting to be against human player, and size and win_length to conventional 3*3
class Tictactoe():
    def __init__(self, size=3, win_length=3, vs_human = True, ai_player_code = None):
        self.size = size
        self.win_length = win_length
        self.board = [['']*size for _ in range(size)]
        self.total_moves = 0
        # to indicate which player moves X and which moves 0
        # currently the player who goes first gets X and the one who goes second gets 0
        self.assigned_move = {
            0 : "X",
            1 : "O"
        }
        self.result_map = {
            0 : "Draw",
            1 : "Win for Player 1",
            -1 : "Win for Player 2"
        }
        self.vs_human = vs_human
        self.ai_player_code = ai_player_code
        # Ran into error first
        self.central_heuristic_evaluation_map = self.set_central_control_heuristic_map()

    # To get board_dimesions/size
    def get_board_size(self):
        return self.size

    # To get the current board_state
    def get_current_board_state(self):
        return self.board

    # to get the provided player's symbol
    def get_player_symbol(self, player_code):
        return self.assigned_move[player_code]

    # to get AI player code
    def get_AI_player_code(self):
        return self.ai_player_code

    # get remaining moves/possible moves based on empty slots
    def get_possible_moves(self):
        current_board_state = self.get_current_board_state()
        board_size = self.get_board_size()
        result = [(x,y) for x in range(board_size) for y in range(board_size) if current_board_state[x][y] == ""]
        return result

    # to get total moves so far
    def get_total_move_count(self):
        return self.total_moves

    # to increment move counter by 1
    def increment_total_move_count(self):
        self.total_moves += 1
        return self.total_moves

    # to decrease move counter by 1
    def decrement_total_move_count(self):
        self.total_moves -= 1
        return self.total_moves

    # to undo last move to try out different branches
    def undo_last_move(self, last_move):
        self.decrement_total_move_count()
        self.board[last_move[0]][last_move[1]] = ""


    # To display the board to the user
    def display_board(self):
        current_board_state = self.get_current_board_state()
        print("\nCurrent board:")
        for row in current_board_state:
            print(" | ".join(cell if cell != '' else ' ' for cell in row))


    # check if attempted position is occupied or not
    def is_occupied(self,move_coordinates):
        current_board_state = self.get_current_board_state()
        # attempted position is not empty, so we shouldn't allow the move
        if current_board_state[move_coordinates[0]][move_coordinates[1]] != "":
            print("The position is occupied try a different position!")
            return True
        return False

    # determine which player goes next based on move order
    def determine_player_turn(self):
        current_move_count = self.get_total_move_count()
        # basically returns 0 or 1, with former indicating first player's turn
        # latter indicates second player's turn
        result = current_move_count % 2
        print(f"Player {result + 1}'s turn to make a move")
        return result % 2

    # just cleaned up user input for co-ordinates
    def get_input_position(self):
        invalid_inputs_provided = True
        result = None
        while invalid_inputs_provided:
            try:
                result = list(map(int, input().split()))
                if len(result) != 2:
                    raise ValueError
                invalid_inputs_provided = False
            except ValueError:
                print("Please provide only and exactly 2 co-ordinates, indexed from 0 indicating where u want ur move to be placed")
        return result

    # to use input position from above and update board
    # if the given input position is not filled
    # Note to be used to make move by humans
    def make_move(self):
        invalid_move_check = True
        current_board_state = self.get_current_board_state()
        player_to_move = self.determine_player_turn()
        move_coordinates = ""
        while invalid_move_check:
            move_coordinates = self.get_input_position()
            if not self.is_occupied(move_coordinates):
                invalid_move_check = False
            else:
                print("Please make a move into a currently unoccupied position")
        current_board_state[move_coordinates[0]][move_coordinates[1]] = self.assigned_move[player_to_move]
        self.display_board()
        # increment total move counter by 1
        self.increment_total_move_count()


    def make_ai_move(self):
        ai_player_code = self.get_AI_player_code()
        best_move = self.select_optimal_ai_move()
        print(best_move)
        self.board[best_move[0]][best_move[1]] = self.get_player_symbol(ai_player_code)
        self.increment_total_move_count()
        self.display_board()

    # diagonal and opp-diagonal check
    def detect_matching_diagonals(self, player_code):
        diagonal_match_detected = True
        opp_diagonal_match_detected = True
        current_board_state = self.get_current_board_state()
        board_size = self.get_board_size()
        player_symbol = self.get_player_symbol(player_code)
        # checking the diagonal
        for number in range(board_size):
            if current_board_state[number][number] != player_symbol:
                diagonal_match_detected = False
        # checking the opp diagonal
        for number in range(board_size):
            if current_board_state[number][(board_size-1) - number] != player_symbol:
                opp_diagonal_match_detected = False
        # if either are true then we can return true
        return diagonal_match_detected or opp_diagonal_match_detected

    # row check
    def detect_matching_rows(self, player_code):
        current_board_state = self.get_current_board_state()
        player_symbol = self.get_player_symbol(player_code)
        # checking the diagonal
        for row in current_board_state:
            if all([symbol == player_symbol for symbol in row]):
                return True
        return False

    # column check
    def detect_matching_columns(self, player_code):
        column_match_detected = False
        current_board_state = self.get_current_board_state()
        board_size = self.get_board_size()
        player_symbol = self.get_player_symbol(player_code)
        # checking for each column
        for col in range(board_size):
            if all(current_board_state[row][col] == player_symbol for row in range(board_size)):
                return True
        return False


    # detect if a win, loss or draw has been made
    # 1 for win for player 1, -1 for loss for player 1/win for player 2 and 0 for draw
    # we should try to run it after each make_move call
    def detect_win_loss(self):
        winner_found = False
        # check for diagonal
        diagonal_player_0 = self.detect_matching_diagonals(0)
        row_player_0 = self.detect_matching_rows(0)
        column_player_0 = self.detect_matching_columns(0)

        diagonal_player_1 = self.detect_matching_diagonals(1)
        row_player_1 = self.detect_matching_rows(1)
        column_player_1 = self.detect_matching_columns(1)

        # if either lines are filled with one character, the particular player wins
        if diagonal_player_0 or row_player_0 or column_player_0:
            return 1
        if diagonal_player_1 or row_player_1 or column_player_1:
            return -1
        # draw condition. Basically all spaces are filled and no winner yet, so it has to be a draw
        if self.total_moves == self.size * self.size:
            return 0
        # Basically a move which doesn't lead to a win, draw or a loss
        return None

    # Our detect win loss returns 1, -1 and 0 when player 1 wins,player 2 wins and draw respectively
    # using it directly in minimax can cause issues since when AI is player 2 it would want to maximize its score
    # but doing it based on the detect_win_loss function would give wrong results
    def generate_win_loss_metrics_wrt_AI(self, outcome):
        ai_player_code = self.get_AI_player_code()
        if ai_player_code == 1:
            if outcome in [-1,1,0]:
                return -1 * outcome
            else:
                return outcome
        else:
            return outcome


    # Show game result
    def fetch_result_map(self):
        return self.result_map

    # #  to run the game and link above methods together
    # def run_game(self):
    #     print("Game has now begun")
    #     self.display_board()
    #     game_ongoing = True
    #     while game_ongoing:
    #         self.make_move()
    #         result = self.detect_win_loss()
    #         result_map = self.fetch_result_map()
    #         if result is not None:
    #             game_ongoing = False
    #             print(result_map[result])

    #  to run the game and link above methods together
    def run_game(self):
        print("Game has now begun")
        self.display_board()
        game_ongoing = True
        if self.ai_player_code is None:
            while game_ongoing:
                self.make_move()
                result = self.detect_win_loss()
                result_map = self.fetch_result_map()
                if result is not None:
                    game_ongoing = False
                    print(result_map[result])
        else:
            # would indicate the order
            ai_player_order = self.get_AI_player_code()
            while game_ongoing:
                if self.get_total_move_count() % 2 == ai_player_order:
                    self.make_ai_move()
                else:
                    self.make_move()
                result = self.detect_win_loss()
                result_map = self.fetch_result_map()
                if result is not None:
                    game_ongoing = False
                    print(result_map[result])

    # method to get a numerical metric for the next possible move
    def minimax(self, isMax):
        # doesn't return anything if game is going on, so only return
        # if it actually has an outcome
        # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
        # the challenge is only when we have to build up recursively when there is no immediate final result
        outcome = self.detect_win_loss()
        # this second layer explanation can be understood from documentation of the method
        ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
        if ai_adjusted_outcome is not None:
            return ai_adjusted_outcome
        # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
        best_score =-math.inf if isMax else math.inf
        # get the current turn player's symbol
        current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
        current_symbol = self.get_player_symbol(current_player)
        possible_moves = self.get_possible_moves()

        for move in possible_moves:
            self.board[move[0]][move[1]] = current_symbol
            self.increment_total_move_count()

            # Recursively call minimax for the next player's turn
            score = self.minimax(not isMax)

            # Undo the move
            self.undo_last_move(move)

            # Step 6: Update best score
            if isMax:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

        return best_score

    # method to get a numerical metric for the next possible move with alpha beta pruning
    # aim to prune branches where alpha >= beta to diminish search space
    def minimax_with_alpha_beta_pruning(self, isMax, alpha = -math.inf, beta = math.inf):
        # doesn't return anything if game is going on, so only return
        # if it actually has an outcome
        # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
        # the challenge is only when we have to build up recursively when there is no immediate final result
        outcome = self.detect_win_loss()
        # this second layer explanation can be understood from documentation of the method
        ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
        if ai_adjusted_outcome is not None:
            return ai_adjusted_outcome
        # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
        best_score =-math.inf if isMax else math.inf
        # get the current turn player's symbol
        current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
        current_symbol = self.get_player_symbol(current_player)
        possible_moves = self.get_possible_moves()

        for move in possible_moves:
            self.board[move[0]][move[1]] = current_symbol
            self.increment_total_move_count()

            # Recursively call minimax for the next player's turn
            score = self.minimax_with_alpha_beta_pruning(not isMax, alpha, beta)

            # Undo the move
            self.undo_last_move(move)

            # Step 6: Update best score
            if isMax:
                best_score = max(best_score, score)
                alpha = max(best_score, alpha)
            else:
                best_score = min(best_score, score)
                beta = min(best_score, beta)

            if alpha >= beta:
                break

        return best_score

    # heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
    # method to get a numerical metric for the next possible move with alpha beta pruning
    # aim to prune branches where alpha >= beta to diminish search space
    def heuristic_minimax_with_alpha_beta_pruning(self, isMax, depth, alpha=-math.inf, beta=math.inf,):
            # doesn't return anything if game is going on, so only return
            # if it actually has an outcome
            # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
            # the challenge is only when we have to build up recursively when there is no immediate final result
            outcome = self.detect_win_loss()
            # this second layer explanation can be understood from documentation of the method
            ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
            if ai_adjusted_outcome is not None:
                return ai_adjusted_outcome

            # reached the max depth we set so we can just return the heuristic evaluation of the board
            if depth == 0:
                return self.heuristically_evaluate_board()

            # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
            best_score = -math.inf if isMax else math.inf
            # get the current turn player's symbol
            current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
            current_symbol = self.get_player_symbol(current_player)
            possible_moves = self.get_possible_moves()

            for move in possible_moves:
                self.board[move[0]][move[1]] = current_symbol
                self.increment_total_move_count()

                # Recursively call minimax for the next player's turn
                score = self.heuristic_minimax_with_alpha_beta_pruning(not isMax, depth -1,  alpha, beta)

                # Undo the move
                self.undo_last_move(move)

                # Step 6: Update best score
                if isMax:
                    best_score = max(best_score, score)
                    alpha = max(best_score, alpha)
                else:
                    best_score = min(best_score, score)
                    beta = min(best_score, beta)

                if alpha >= beta:
                    break

            return best_score


    # basically using the move evaluation found in the previous step to choose an optimal move by evaluating
    # for each move possible given current empty spaces
    def select_optimal_ai_move(self):
        current_board_size = self.get_board_size()
        current_player = self.ai_player_code
        current_symbol = self.get_player_symbol(current_player)
        possible_moves = self.get_possible_moves()
        # choosing the lowest abs value possible initially
        best_score = -math.inf
        best_follow_up_move = None
        for next_move in possible_moves:
            # trying each of the list of possible empty space given the current board state
            self.board[next_move[0]][next_move[1]] = current_symbol
            # increase the move_count by one
            self.increment_total_move_count()
            # trying to evaluate min player's
            # score to be used for minimax without alpha beta pruning
            # score = self.minimax(False)
            # score to be used for minimax with alpha beta pruning
            # score = self.minimax_with_alpha_beta_pruning(False, -math.inf, math.inf)
            score = 0
            if current_board_size <= 3:
                score = self.minimax_with_alpha_beta_pruning(False, -math.inf, math.inf)
            else:
                score = self.heuristic_minimax_with_alpha_beta_pruning(False,9-current_board_size, -math.inf, math.inf)
            # it was only for trial so need to go back to previous state after trying
            self.undo_last_move(next_move)
            if score > best_score:
                best_score = score
                best_follow_up_move = next_move
        return best_follow_up_move


    # To be used to prevent our alpha beta minimax from going till the end
    # and get slowed down instead we can try and use a heuristic function to look for what seems like a better positions
    # only needed when game size is greater than 3 otherwise exhaustive search does the trick
    def heuristically_evaluate_board(self):
        multiplier = 0.18
        fork_multiplier = 0.28
        heuristic_value_diagonals = self.clamp(self.calculate_heuristic_value_diagonals())
        heuristic_value_rows = self.clamp(self.calculate_heuristic_value_rows())
        heuristic_value_columns = self.clamp(self.calculate_heuristic_value_columns())
        heuristic_value_central_dominance = self.clamp(self.calculate_heuristic_value_central_dominance())
        heuristic_value_fork = self.clamp(self.calculate_heuristic_value_fork())
        return multiplier*(heuristic_value_central_dominance + heuristic_value_columns + heuristic_value_rows + heuristic_value_diagonals) + (fork_multiplier* heuristic_value_fork)


    # to avoid repetition
    # can be used for rows, columns and diagonals
    def streak_heuristic_helper(self, streak):
        current_board_size = self.get_board_size()
        ai_player_code = self.ai_player_code
        ai_symbol = self.get_player_symbol(ai_player_code)
        opponent_symbol = self.get_player_symbol((ai_player_code + 1) % 2)
        ai_symbol_count = streak.count(ai_symbol)
        opponent_symbol_count = streak.count(opponent_symbol)
        empty_count = streak.count("")
        # basically if your opponent has even a single piece along the diagonal, not worth exploring much
        if opponent_symbol_count != 0:
            if opponent_symbol_count < current_board_size//2:
                return 0
            else:
                # defensive, penalizes increasing opponent presence if u haven't established your presence already
                if ai_symbol_count == 0:
                    return  - (2 ** (opponent_symbol_count / current_board_size))
                else:
                    return - (opponent_symbol_count / current_board_size) * 0.5  # lighter penalty
        # part after + is a bias term to incentivize exploration of completely empty rows, columns and diagonals
        # offensive move
        return 2 ** (ai_symbol_count/current_board_size) + (empty_count / current_board_size) * 0.1


    # method to obtain heuristic value for both diagonals
    def calculate_heuristic_value_diagonals(self):
        current_board_state = self.get_current_board_state()
        current_board_size = self.get_board_size()
        # diagonal 1
        diagonal_1 = [current_board_state[x][x] for x in range(current_board_size)]
        diagonal_1_heuristic_score = self.streak_heuristic_helper(diagonal_1)
        # diagonal 2
        diagonal_2 = [current_board_state[x][(current_board_size-1) - x] for x in range(current_board_size)]
        diagonal_2_heuristic_score = self.streak_heuristic_helper(diagonal_2)
        return (diagonal_1_heuristic_score + diagonal_2_heuristic_score)/2

    def calculate_heuristic_value_rows(self):
        current_board_state = self.get_current_board_state()
        current_board_size = self.get_board_size()
        result = 0
        for row in current_board_state:
            result += self.streak_heuristic_helper(row)
        # otherwise result goes too much for bigger boards, averaging keeps it somewhat normalized
        result = result/current_board_size
        return result

    # to keep values in range [-1,1]
    def clamp(self, value):
        if value == 0.0:
            return value
        elif value > 0.0:
            return min(1.0, value)
        else:
            return max(-1.0, value)

    def calculate_heuristic_value_columns(self):
        current_board_state = self.get_current_board_state()
        current_board_size = self.get_board_size()
        result = 0
        for number in range(current_board_size):
            column = [current_board_state[value][number] for value in range(current_board_size)]
            result += self.streak_heuristic_helper(column)
        # otherwise result goes too much for bigger boards, averaging keeps it somewhat normalized
        result = result / current_board_size
        return result


    def set_central_control_heuristic_map(self):
        current_board_size = self.get_board_size()
        current_board_state = self.get_current_board_state()
        score_template_map = [[0.0 for x in range(current_board_size)] for y in range(current_board_size)]
        mid = (current_board_size - 1) / 2 if current_board_size % 2 == 0 else current_board_size // 2
        ai_player_code = self.get_AI_player_code()
        ai_symbol = self.get_player_symbol(ai_player_code)
        for i in range(current_board_size):
            for j in range(current_board_size):
                if current_board_state[i][j] == ai_symbol:
                    # using manhattan distance from center for convenience
                    row_dist = abs(i - mid)
                    col_dist = abs(j - mid)
                    total_dist = row_dist + col_dist
                    # this is mostly to prevent values from really explode for bigger tictactoe boards
                    # also prevents these metrics from dominating over other metrics
                    normalized_total_distance = total_dist/current_board_size
                    # Center is most valuable, then edges, then corners
                    # squaring to penalize distant ones even more
                    score_template_map[i][j] = (1 - normalized_total_distance) ** 2
        return score_template_map

    # heavily incentivizes central control
    def calculate_heuristic_value_central_dominance(self):
        result = 0
        central_heuristic_evaluation_map = self.set_central_control_heuristic_map()
        for row in central_heuristic_evaluation_map:
            result += sum(row)
        return result

    # basically if there is at least 1 winnable line through a square
    def is_unchallenged(self, line, player_symbol, opponent_symbol):
        return line.count(opponent_symbol) == 0 and line.count(player_symbol) >= 1

    # given a co-ordinate get its intersecting lines
    def get_intersecting_lines(self, move):
        current_board_size = self.get_board_size()
        current_board_state = self.get_current_board_state()
        x, y = move
        row = [current_board_state[x][n] for n in range(current_board_size)]
        column = [current_board_state[n][y] for n in range(current_board_size)]
        result = [row, column]
        # point lies on main diagonal
        if x == y:
            result.append([current_board_state[a][a] for a in range(current_board_size)])
        # point lies on opposite diagonal
        if y == (current_board_size -1) - x:
            result.append([current_board_state[a][(current_board_size -1) - a] for a in range(current_board_size)])
        return result


    #for heuristic offence and defence
    def calculate_heuristic_value_fork(self):
        current_board_size = self.get_board_size()
        current_board_state = self.get_current_board_state()
        ai_player_code = self.ai_player_code
        ai_symbol = self.get_player_symbol(ai_player_code)
        opponent_symbol = self.get_player_symbol((ai_player_code + 1) % 2)
        player_fork_count = 0
        opponent_fork_count = 0
        # to prevent duplicate position counting
        ai_fork_positions = set()
        # to prevent duplicate position counting
        opponent_fork_positions = set()
        for i in range(current_board_size):
            for j in range(current_board_size):
                if current_board_state[i][j] == ai_symbol:
                    intersecting_lines = self.get_intersecting_lines([i,j])
                    ai_winning_lines = 0
                    for line in intersecting_lines:
                        if self.is_unchallenged(line,ai_symbol,opponent_symbol):
                            ai_winning_lines += 1
                        if ai_winning_lines >= 2:
                            ai_fork_positions.add((i,j))
        for i in range(current_board_size):
            for j in range(current_board_size):
                if current_board_state[i][j] == opponent_symbol:
                    intersecting_lines = self.get_intersecting_lines([i,j])
                    opponent_winning_lines = 0
                    for line in intersecting_lines:
                        if self.is_unchallenged(line,opponent_symbol,ai_symbol):
                            opponent_winning_lines += 1
                        if opponent_winning_lines >= 2:
                            opponent_fork_positions.add((i,j))
        ai_fork_score = len(ai_fork_positions)
        opponent_fork_score = len(opponent_fork_positions)
        # this is so I can prevent division by 0 errors
        return (ai_fork_score - opponent_fork_score)/max(1,ai_fork_score + opponent_fork_score)



launch_game_with_user_config()