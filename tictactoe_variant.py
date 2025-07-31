import math
from copy import deepcopy
import random
import time
import numpy as np
from scipy.special import softmax

from Mcts import Mcts
from boardgame import BoardGame
from constant_strings import CONCLUSIVE_RESULT_MULTIPLIER, TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS, \
    MAX_MOVE_COUNT_WITH_INITIAL_TEMPERATURE_CONTROL, MOVE_X, MOVE_O, MCTS, ALPHA_BETA_PRUNING, \
    MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS, MIN_GAME_SIM_BENCHMARK_MCTS, MCTS_NN
from common_utils import clamp, board_hash


# Let us represent players with player 1 by 0 and player 2 by 1
# default setting to be against human player, and size and win_length to conventional 3*3
class Tictactoe(BoardGame) :
    def __init__(self, size=3, win_length=3, temperature_control = TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS,vs_human = True, ai_player_code = None, simulation_mode = False, ai_type = None):
        super().__init__(size=size, vs_human=vs_human, ai_player_code=ai_player_code, ai_type=ai_type,
                         simulation_mode=simulation_mode)
        self.win_length = win_length
        self.board = [['.']*size for _ in range(size)]
        # Note this has been set for simulation purposes
        self.search_depth = 10
        self.temperature_control = temperature_control
        self.logging_mode = True
        # to indicate which player moves X and which moves 0
        # The player who goes first gets X and the one who goes second gets 0
        self.assigned_move = {
            0 : MOVE_X,
            1 : MOVE_O
        }
        self.result_map = {
            0 : "Draw",
            1 : "Win for Player 1",
            -1 : "Win for Player 2"
        }
        # Ran into error first
        # Accidentally called the score_template_map setter method here first
        # lead to issues. Much safer to set it later on
        self.central_heuristic_evaluation_map = None
        self.match_result = None

    # to aid use in MCTS algorithm
    # trying with a slightly light-weight clone in comparison to previous approach
    # previous approach deep-copied the whole instance which created/could create a memory overload
    def clone_instance(self):
        cloned = Tictactoe(
            size=self.size,
            win_length=self.win_length,
            temperature_control=self.temperature_control,
            vs_human=self.vs_human,
            ai_player_code=self.ai_player_code,
            simulation_mode=self.simulation_mode,
            ai_type=self.ai_type
        )

        # Deepcopy only essential mutable attributes
        cloned.board = deepcopy(self.board)
        cloned.total_moves = self.total_moves
        cloned.search_depth = self.search_depth
        cloned.central_heuristic_evaluation_map = deepcopy(self.central_heuristic_evaluation_map)

        # Disable logging for simulation
        cloned.logging_mode = False
        return cloned

    # player whose turn it is to make a move
    # 0 means first player, 1 means second player
    def current_player(self):
        return self.total_moves % 2

    def get_temperature_control(self):
        return self.temperature_control

    def set_temperature_control(self, custom_temperature):
        self.temperature_control = custom_temperature

    # to decide whether game is AI vs AI or not
    def get_game_mode(self):
        return self.simulation_mode
    # to set the type of AI to be used
    def get_AI_type(self):
        return self.ai_type

    def set_AI_type(self, ai_type):
        self.ai_type = ai_type
        return ai_type

    def set_to_simulation_mode(self):
        self.simulation_mode = True
        return self.simulation_mode

    # to get the provided player's symbol
    def get_player_symbol(self, player_code):
        return self.assigned_move[player_code]

    # to get AI player code
    def get_AI_player_code(self):
        return self.ai_player_code

    # to be used to make AI play both sides by
    def alternate_AI_player_code(self):
        if self.ai_player_code == 0:
            self.ai_player_code = 1
        else:
            self.ai_player_code = 0

    # get remaining moves/possible moves based on empty slots
    def get_possible_moves(self):
        current_board_state = self.get_current_board_state()
        board_size = self.get_board_size()
        result = [(x,y) for x in range(board_size) for y in range(board_size) if current_board_state[x][y] == "."]
        return result

    # to get total moves so far
    def get_total_move_count(self):
        return self.total_moves


    # to get total moves count by board state
    # needed since our game order logic is reliant on total moves played
    # this method is to take a custom position and get the move count
    # based on that
    def get_move_count_from_position(self,board_positions):
        move_count_by_piece = {
            "X": 0,
            "O": 0
        }
        for row in range(len(board_positions)):
            for column in range(len(board_positions)):
                current_move = board_positions[row][column]
                if current_move in [MOVE_X, MOVE_O]:
                    move_count_by_piece[current_move] += 1
        return sum(move_count_by_piece.values())

    # to undo last move to try out different branches
    def undo_last_move(self, last_move):
        self.decrement_total_move_count()
        self.board[last_move[0]][last_move[1]] = "."


    # determine which player goes next based on move order
    def determine_player_turn(self):
        current_move_count = self.get_total_move_count()
        # basically returns 0 or 1, with former indicating first player's turn
        # latter indicates second player's turn
        result = current_move_count % 2
        if not self.simulation_mode:
            self.selective_print(f"Player {result + 1}'s turn to make a move")
        return result % 2

    # to use input position from above and update board
    # if the given input position is not filled
    # Note to be used to make move by humans
    def human_make_move(self):
        invalid_move_check = True
        board_size = self.get_board_size()
        current_board_state = self.get_current_board_state()
        player_to_move = self.determine_player_turn()
        move_coordinates = ""
        while invalid_move_check:
            move_coordinates = self.get_input_position()
            if min(move_coordinates[0], move_coordinates[1]) < 0 or max(move_coordinates[0], move_coordinates[1]) > board_size-1:
                print("The provided move co-ordinates are out of the range of values for this board size!. Please enter something within that range! ")
                continue
            if not self.is_occupied(move_coordinates):
                invalid_move_check = False
            else:
                self.selective_print("Please make a move into a currently unoccupied position")
        current_board_state[move_coordinates[0]][move_coordinates[1]] = self.assigned_move[player_to_move]
        self.display_board()
        # increment total move counter by 1
        self.increment_total_move_count()

    # modifications to be done
    def make_best_nn_based_move(self):
        pass



    # checking for moves which can lead to immediate wins or lead to immediate losse
    # so they can be prioritized
    def check_immediate_result(self, possible_moves):
        current_board_state = self.get_current_board_state()
        player_to_move = self.current_player()
        player_symbol = self.get_player_symbol(player_to_move)
        opponent_symbol = self.get_player_symbol(abs(player_to_move-1))
        # first we check if we can win
        # if we can, we play that
        for move in possible_moves:
            current_board_state[move[0]][move[1]] = player_symbol
            self.increment_total_move_count()
            # basically win conditions for the current player
            if (self.detect_win_loss() == 1 and player_symbol == MOVE_X) or (self.detect_win_loss() == -1 and player_symbol == MOVE_O):
                self.undo_last_move(move)
                return move
            self.undo_last_move(move)
        # code reaches here means, we can't win so we check if opponent has a win
        # that we need to block
        for move in possible_moves:
            current_board_state[move[0]][move[1]] = opponent_symbol
            self.increment_total_move_count()
            # basically opponent win condition
            if (self.detect_win_loss() == -1 and player_symbol == MOVE_X) or (self.detect_win_loss() == 1 and player_symbol == MOVE_O):
                self.undo_last_move(move)
                return move
            self.undo_last_move(move)
        # if it reached till here means no move of immediate significance so we can return None
        return None

    def make_ai_move(self,ai_type):
        # Neural Network needs the board state prior to the move, in a flattened form
        # so it can be used as a vector/list/tensor along with the policy map for that corresponding state
        pre_move_flattened_state_2d = "".join(str(cell) for row in self.board for cell in row)
        ai_player_code = self.get_AI_player_code()
        best_move = None
        policy_map = None
        if ai_type == ALPHA_BETA_PRUNING:
            best_move, policy_map = self.select_optimal_ai_move_with_temperature_control()
        if ai_type in [MCTS, MCTS_NN]:
            # this check is very very very important and has been added to prevent
            # missing immediate wins and losses
            conclusive_result = self.check_immediate_result(self.get_possible_moves())
            if conclusive_result is not None:
                best_move = conclusive_result
                # possibly useful for neural net training
                policy_map = np.zeros(self.size * self.size, dtype=np.float32)
                r, c = best_move
                # flattened board indexing
                policy_map[r * self.size + c] = 1.0
            else:
                best_move, policy_map = self.select_optimal_ai_move_mcts()
        self.move_list.append((pre_move_flattened_state_2d, policy_map))
        self.selective_print(best_move)
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

    # Show game result
    def fetch_result_map(self):
        return self.result_map

    # methods to reduce the temperature control after a few moves. Helps to reduce the randomness after the intial moves
    def tweak_temp_control_based_on_move_count(self):
         move_count = self.get_total_move_count()
         size = self.get_board_size()
         # We flip the temperature control after a certain number of games for optimized move selection
         if (size * 2)- move_count < MAX_MOVE_COUNT_WITH_INITIAL_TEMPERATURE_CONTROL:
             self.set_temperature_control(TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS)


    # just for MCTS purposes. Light-weight with an added minimal check to avoid
    # immediate losses
    def make_pseudo_random_move(self):
        possible_moves = self.get_possible_moves()
        # a bit unclear one exact role. More like an additional safety check
        if not possible_moves:
            return None
        player_to_move = self.current_player()
        immediate_result_move = self.check_immediate_result(possible_moves)
        # no move which leads to an immediate result so choose a random one
        move = immediate_result_move if immediate_result_move is not None else random.choice(possible_moves)
        self.get_current_board_state()[move[0]][move[1]] = self.get_player_symbol(player_to_move)
        # increment total move counter by 1
        self.increment_total_move_count()
        return move


    #  to run the game and link above methods together
    def run_game(self):
        simulation_mode = self.get_game_mode()
        ai_type = self.get_AI_type()
        # end_result of the game - 1, 0 and -1 indicating
        # victory for x, draw and defeat for x respectively
        result = None
        # AI vs AI
        if simulation_mode:
            # AI vs AI - no nn- MCTS
            if ai_type == MCTS:
                # AI plays both moves here so we can simulate that by just alternating the
                # symbol usage
                game_ongoing = True
                while game_ongoing:
                    result = self.detect_win_loss()
                    # this check we have added since simulation loop can be entered
                    if result is not None:
                        game_ongoing = False
                        break
                    self.make_pseudo_random_move()
                    result = self.detect_win_loss()
                    if result is not None:
                        # if we don't break here it goes on to alternate the code even if the game has ended
                        break
                    self.alternate_AI_player_code()
            # AI vs AI - Alpha Beta Pruning
            elif ai_type == ALPHA_BETA_PRUNING:
                # AI plays both moves here so we can simulate that by just alternating the
                # symbol usage
                game_ongoing = True
                while game_ongoing:
                    # commented for testing purposes. Don't forget to uncomment before using
                    # self.tweak_temp_control_based_on_move_count()
                    self.make_ai_move(ai_type)
                    result = self.detect_win_loss()
                    result_map = self.fetch_result_map()
                    self.alternate_AI_player_code()
                    if result is not None:
                        game_ongoing = False
                        self.selective_print(result_map[result])
            # AI vs AI - MCTS + NN
            # we will put a placeholder for now
            else:
                pass
        # human playing
        else:
            self.selective_print("Game has now begun")
            self.display_board()
            game_ongoing = True
            # Human vs Human
            if self.ai_player_code is None:
                self.selective_print("Human vs Human mode")
                while game_ongoing:
                    self.human_make_move()
                    result = self.detect_win_loss()
                    result_map = self.fetch_result_map()
                    if result is not None:
                        game_ongoing = False
                        self.selective_print(result_map[result])
            # Human vs AI
            else:
                # would indicate the order
                ai_player_order = self.get_AI_player_code()
                self.selective_print(f"Human vs AI mode - {ai_type}")
                while game_ongoing:
                    if self.get_total_move_count() % 2 == ai_player_order:
                        self.make_ai_move(ai_type)
                    else:
                        self.human_make_move()
                    result = self.detect_win_loss()
                    result_map = self.fetch_result_map()
                    if result is not None:
                        game_ongoing = False
                        self.selective_print(result_map[result])
        self.match_result = result
        return result

    # lighter function to be used for simulations
    def rollout_pseudo_random(self) -> int:
        while True:
            result = self.detect_win_loss()
            if result is not None:
                return result  # terminal

            # ε-greedy: win-in-1, block-in-1, else random
            self.make_pseudo_random_move()

            result = self.detect_win_loss()
            if result is not None:
                return result

            self.alternate_AI_player_code()



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

    # # method to get a numerical metric for the next possible move with alpha beta pruning
    # # aim to prune branches where alpha >= beta to diminish search space
    # # we add a depth_to_result parameter to allow for setting of search depths
    # def minimax_with_alpha_beta_pruning(self, isMax, depth_to_result, alpha = -math.inf, beta = math.inf):
    #     # doesn't return anything if game is going on, so only return
    #     # if it actually has an outcome
    #     # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
    #     # the challenge is only when we have to build up recursively when there is no immediate final result
    #     outcome = self.detect_win_loss()
    #     # this second layer explanation can be understood from documentation of the method
    #     ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
    #     if ai_adjusted_outcome is not None:
    #         if ai_adjusted_outcome > 0:
    #             return ai_adjusted_outcome - depth_to_result
    #         if ai_adjusted_outcome < 0:
    #             # basically, if we are losing, we prolong the result
    #             return ai_adjusted_outcome + depth_to_result
    #         return ai_adjusted_outcome
    #     # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
    #     best_score =-math.inf if isMax else math.inf
    #     # get the current turn player's symbol
    #     current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
    #     current_symbol = self.get_player_symbol(current_player)
    #     possible_moves = self.get_possible_moves()
    #
    #     for move in possible_moves:
    #         self.board[move[0]][move[1]] = current_symbol
    #         self.increment_total_move_count()
    #
    #         # Recursively call minimax for the next player's turn
    #         score = self.minimax_with_alpha_beta_pruning(not isMax, depth_to_result + 1, alpha, beta)
    #
    #         # Undo the move
    #         self.undo_last_move(move)
    #
    #         # Step 6: Update best score
    #         if isMax:
    #             best_score = max(best_score, score)
    #             alpha = max(best_score, alpha)
    #         else:
    #             best_score = min(best_score, score)
    #             beta = min(best_score, beta)
    #
    #         if alpha >= beta:
    #             break
    #
    #     return best_score


    # WITH TT table

    # method to get a numerical metric for the next possible move with alpha beta pruning
    # aim to prune branches where alpha >= beta to diminish search space
    # we add a depth_to_result parameter to allow for setting of search depths
    def minimax_with_alpha_beta_pruning(self, isMax, depth_to_result, alpha = -math.inf, beta = math.inf):
        # first check cache for matches
        cached_score = self.fetch_existing_hash(depth_to_result)
        if cached_score is not None:
            return cached_score
        outcome = self.detect_win_loss()
        terminal_score = self.fit_to_ai_metrics(outcome, depth_to_result)
        if terminal_score is not None:
            # store terminal evaluation in TT table
            self.store_in_transposition_table(terminal_score, depth_to_result)
            return terminal_score

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
            score = self.minimax_with_alpha_beta_pruning(not isMax, depth_to_result + 1, alpha, beta)

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

        self.store_in_transposition_table(best_score, depth_to_result)
        return best_score


    # KEPT FOR TESTING AND DEMO ONLY

    # heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
    # method to get a numerical metric for the next possible move with alpha beta pruning
    # aim to prune branches where alpha >= beta to diminish search space
    # we add a depth_to_result parameter to allow for setting of search depths
    def heuristic_minimax_with_alpha_beta_pruning(self, isMax, max_ai_search_depth, depth_to_result, alpha=-math.inf, beta=math.inf):
            # doesn't return anything if game is going on, so only return
            # if it actually has an outcome
            # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
            # the challenge is only when we have to build up recursively when there is no immediate final result
            outcome = self.detect_win_loss()
            # this second layer explanation can be understood from documentation of the method
            ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
            if ai_adjusted_outcome is not None:
                # alpha beta pruning/minimax doesn't differentiate between quicker and longer wins,
                # if we can reduce the heuristic value according to how many moves before win
                if ai_adjusted_outcome > 0:
                    return ai_adjusted_outcome - depth_to_result
                # basically, if we are losing, we prolong the result
                if ai_adjusted_outcome < 0:
                    return ai_adjusted_outcome + depth_to_result
                return ai_adjusted_outcome

            # reached the max depth we set so we can just return the heuristic evaluation of the board
            if max_ai_search_depth == 0:
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
                score = self.heuristic_minimax_with_alpha_beta_pruning(not isMax, max_ai_search_depth - 1, depth_to_result + 1, alpha, beta)

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



    # TT table added

    # heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
    # method to get a numerical metric for the next possible move with alpha beta pruning
    # aim to prune branches where alpha >= beta to diminish search space
    # we add a depth_to_result parameter to allow for setting of search depths
    def heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(self, isMax, max_ply, depth_to_result, alpha=-math.inf, beta=math.inf):
            # first check cache for matches
            cached_score = self.fetch_existing_hash(depth_to_result)
            if cached_score is not None:
                return cached_score
            outcome = self.detect_win_loss()
            terminal_score = self.fit_to_ai_metrics(outcome, depth_to_result)
            if terminal_score is not None:
                # store terminal evaluation in TT
                self.store_in_transposition_table(terminal_score, depth_to_result)
                return terminal_score

            if max_ply == 0:
                # horizon reached – evaluate statically
                static_score = self.heuristically_evaluate_board()
                # store horizon evaluation in TT
                self.store_in_transposition_table(static_score, depth_to_result)
                return static_score

            # we reach here means no immediate win
            best_score = -math.inf if isMax else math.inf
            ordered_moves = []
            for depth in range(1, max_ply + 1):
                # removing this line greatly increases speed but reduces search width
                # alpha, beta = -math.inf, math.inf
                current_move_scores = []
                curr_best_score = -math.inf if isMax else math.inf
                current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
                current_symbol = self.get_player_symbol(current_player)
                possible_moves = self.get_possible_moves() if len(ordered_moves) == 0 else [move for move,score in ordered_moves]
                for move in possible_moves:
                    self.board[move[0]][move[1]] = current_symbol
                    self.increment_total_move_count()
                    # Recursively call minimax for the next player's turn
                    score = self.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(not isMax, depth - 1, depth_to_result + 1, alpha, beta)
                    # Undo the move
                    self.undo_last_move(move)
                    current_move_scores.append((move, score))
                    # Step 6: Update best score
                    if isMax:
                        curr_best_score = max(curr_best_score, score)
                        alpha = max(curr_best_score, alpha)
                    else:
                        curr_best_score = min(curr_best_score, score)
                        beta = min(curr_best_score, beta)

                    if alpha >= beta:
                        break
                ordered_moves = sorted(current_move_scores, key=lambda x: x[1], reverse=isMax)
                best_score = curr_best_score

            # --- TT: store internal node evaluation after all depths ---
            self.store_in_transposition_table(best_score, depth_to_result)
            return best_score

    # # heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
    # # method to get a numerical metric for the next possible move with alpha beta pruning
    # # aim to prune branches where alpha >= beta to diminish search space
    # # we add a depth_to_result parameter to allow for setting of search depths
    # def heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(self, isMax, max_ply, depth_to_result, alpha=-math.inf, beta=math.inf):
    #         # no need for anything at all if there is an immediate win
    #         outcome = self.detect_win_loss()
    #         ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
    #         if ai_adjusted_outcome is not None:
    #             if ai_adjusted_outcome > 0:
    #                 return ai_adjusted_outcome - depth_to_result
    #             if ai_adjusted_outcome < 0:
    #                 return ai_adjusted_outcome + depth_to_result
    #             return ai_adjusted_outcome
    #
    #         if max_ply == 0:
    #             # horizon reached – evaluate statically
    #             return self.heuristically_evaluate_board()
    #
    #         # we reach here means no immediate win
    #         best_score = -math.inf if isMax else math.inf
    #         ordered_moves = []
    #         for depth in range(1, max_ply + 1):
    #             # removing this line greatly increases speed but reduces search width
    #             # alpha, beta = -math.inf, math.inf
    #             current_move_scores = []
    #             curr_best_move = None
    #             curr_best_score = -math.inf if isMax else math.inf
    #             current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
    #             current_symbol = self.get_player_symbol(current_player)
    #             possible_moves = self.get_possible_moves() if len(ordered_moves) == 0 else [move for move,score in ordered_moves]
    #             for move in possible_moves:
    #                 self.board[move[0]][move[1]] = current_symbol
    #                 self.increment_total_move_count()
    #                 # Recursively call minimax for the next player's turn
    #                 score = self.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(not isMax, depth - 1, depth_to_result + 1, alpha, beta)
    #                 # Undo the move
    #                 self.undo_last_move(move)
    #                 current_move_scores.append((move, score))
    #                 # Step 6: Update best score
    #                 if isMax:
    #                     curr_best_score = max(curr_best_score, score)
    #                     alpha = max(curr_best_score, alpha)
    #                 else:
    #                     curr_best_score = min(curr_best_score, score)
    #                     beta = min(curr_best_score, beta)
    #
    #                 if alpha >= beta:
    #                     break
    #             ordered_moves = sorted(current_move_scores, key=lambda x: x[1], reverse=isMax)
    #             best_score = curr_best_score
    #         return best_score



    # # basically using the move evaluation found in the previous step to choose an optimal move by evaluating
    # # for each move possible given current empty spaces
    # def select_optimal_ai_move(self):
    #     current_board_size = self.get_board_size()
    #     current_player = self.ai_player_code
    #     current_symbol = self.get_player_symbol(current_player)
    #     possible_moves = self.get_possible_moves()
    #     # choosing the lowest abs value possible initially
    #     best_score = -math.inf
    #     best_follow_up_move = None
    #     for next_move in possible_moves:
    #         # trying each of the list of possible empty space given the current board state
    #         self.board[next_move[0]][next_move[1]] = current_symbol
    #         # increase the move_count by one
    #         self.increment_total_move_count()
    #         # trying to evaluate min player's
    #         # score to be used for minimax without alpha beta pruning
    #         # score = self.minimax(False)
    #         # score to be used for minimax with alpha beta pruning
    #         # score = self.minimax_with_alpha_beta_pruning(False, -math.inf, math.inf)
    #         score = 0
    #         # Basically, if the board is smaller, you don't need to rely on heuristics, if not you do
    #         if current_board_size <= 3:
    #             score = self.minimax_with_alpha_beta_pruning(False,1, -math.inf, math.inf)
    #         else:
    #             # If the number of moves is less than this number, we can do a search till the end with alpha beta pruning, instead of having to rely on heuristics
    #             if len(possible_moves) <= 10:
    #                 score = self.minimax_with_alpha_beta_pruning(False, 1,  -math.inf, math.inf)
    #             else:
    #                 # the depth_to_result being set to 1 here is because when u first call this method
    #                 # it is checking the board states exactly 1 move away from now
    #                 # which in turn would update the values and do it 1 move from them and so on
    #                 score = self.heuristic_minimax_with_alpha_beta_pruning(False,9-current_board_size, 1, -math.inf, math.inf)
    #         # it was only for trial so need to go back to previous state after trying
    #         self.undo_last_move(next_move)
    #         if score > best_score:
    #             best_score = score
    #             best_follow_up_move = next_move
    #     self.selective_print(f"Evaluation Score of Position is {best_score}")
    #     return best_follow_up_move

    # basically using the softmax function to create a bunch of probabilities for the given move_score list
    def generate_probability_distribution_with_temperature(self,move_score_list, temperature_control ):
        score_list = np.array([score for move,score in move_score_list], dtype=np.float64)
        # probability_distribution = np.exp(score_list/temperature_control)
        # probability_summation = sum(probability_distribution)
        # probability_distribution = probability_distribution/probability_summation
        probability_distribution = softmax(score_list/temperature_control)
        return probability_distribution


    # this method allows us to create a policy board map which is gen
    # in a flattened format for each move,which is needed for the Neural net
    def generate_flattened_policy_board_map_for_neural_net(self, move_score_list, probability_distribution):
        current_board_size = self.get_board_size()
        policy_full = np.zeros(current_board_size * current_board_size, dtype=np.float32)
        # Fill in the probability for each legal move
        # basically we have move_score_list with available moves in the (x,y), score format
        # then we have probabilities which is just a list of probabilities for
        # each of the available moves
        for (relevant_move, relevant_score), p in zip(move_score_list, probability_distribution):
            # logic to get index of a 2d board flattened in 1d
            flat_idx = relevant_move[0] * current_board_size + relevant_move[1]  # Convert 2D move to 1D index
            policy_full[flat_idx] = p
        return policy_full

    # basically using the move evaluation found in the previous step to choose an optimal move by evaluating
    # for each move possible given current empty spaces
    def select_optimal_ai_move_with_temperature_control(self):
        start_time = time.time()
        temperature_control = self.get_temperature_control()
        current_board_size = self.get_board_size()
        current_player = self.ai_player_code
        current_symbol = self.get_player_symbol(current_player)
        possible_moves = self.get_possible_moves()
        # choosing the lowest abs value possible initially
        best_score = -math.inf
        best_follow_up_move = None
        search_depth = self.search_depth
        move_score_list = []
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
            # Basically, if the board is smaller, you don't need to rely on heuristics, if not you do
            if current_board_size <= 3:
                score = self.minimax_with_alpha_beta_pruning(False,1, -math.inf, math.inf)
            else:
                # If the number of moves is less than this number, we can do a search till the end with alpha beta pruning, instead of having to rely on heuristics
                if len(possible_moves) <= 10:
                    score = self.minimax_with_alpha_beta_pruning(False, 1,  -math.inf, math.inf)
                else:
                    # the depth_to_result being set to 1 here is because when u first call this method
                    # it is checking the board states exactly 1 move away from now
                    # which in turn would update the values and do it 1 move from them and so on

                    # without iterative deepening
                    # score = self.heuristic_minimax_with_alpha_beta_pruning(False,search_depth, 1, -math.inf, math.inf)

                    # with iterative deepening
                    score = self.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
                        isMax=False,  # if you're simulating the opponent's move
                        max_ply=10,  # depth limit — can tweak based on board size/time
                        depth_to_result=1  # always starts from 1
                    )
            # it was only for trial so need to go back to previous state after trying
            self.undo_last_move(next_move)
            # if score > best_score:
            #     best_score = score
            #     best_follow_up_move = next_move
            move_score_list.append((next_move, score))
        probability_distribution = self.generate_probability_distribution_with_temperature(move_score_list,
                                                                                           temperature_control)
        self.selective_print(f"Probability distribution is - {probability_distribution}")
        probability_based_idx = np.random.choice(len(move_score_list), p=probability_distribution)
        best_follow_up_move = move_score_list[probability_based_idx][0]
        best_score = move_score_list[probability_based_idx][1]
        self.selective_print(f"Evaluation Score of Position is {best_score}")
        policy_map = self.generate_flattened_policy_board_map_for_neural_net(move_score_list=move_score_list, probability_distribution=probability_distribution)
        end_time = time.time()
        print(f"AI move computation took {end_time - start_time:.3f} seconds.")
        return best_follow_up_move, policy_map

    # Formula used in the Alpha Go paper prioritizing visit count as the most viable metric πₐ ∝ N(s, a)¹/τ
    # Long story short, ove a decent number of simulations visit count is the best metric to look for
    def select_optimal_ai_move_mcts(self):
        # this will help us not to use too much load in doing and undoing on the
        # original instance and just perform everything on a clone
        simulated_mode = self.get_game_mode()
        cloned_instance = self.clone_instance()
        current_player = cloned_instance.ai_player_code
        # choosing the lowest abs value possible initially
        best_score = -math.inf
        best_follow_up_move = None
        mcts = Mcts(root=None, tictactoe_instance=cloned_instance)
        # variable which assigns different number of max runs based on self or vs human play
        max_runs = MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS if simulated_mode == False else MIN_GAME_SIM_BENCHMARK_MCTS
        mcts.commence_mcts_for_selfplay(max_runs=max_runs)
        parent_node = mcts.get_root()
        children = parent_node.get_children()
        # ───────── DEBUG PRINT: show root statistics once per AI move ─────────
        print("\n=== ROOT AFTER SEARCH FINISHED ===")
        for mv, node in children.items():
            print("[root] move", mv,
                  "visits =", node.get_visits(),
                  "wins   =", node.get_wins())
        print("==================================\n")
        # ───────────────────────────────────────────────────────────────────────

        move_score_list= [(move,children[move].get_visits()) for move in children]
        # basically getting max based on move visit counts
        best_move = max(move_score_list, key=lambda x: x[1])[0]
        # # ── choose best_move ────────────────────────────────────────────────
        # for mv, node in children.items():  # proven-win guard
        #     if node.get_visits() > 0 and node.get_wins() == node.get_visits():
        #         best_move = mv
        #         break
        # else:  # fall back to visits
        #     move_score_list = [(m, c.get_visits()) for m, c in children.items()]
        #     best_move = max(move_score_list, key=lambda x: x[1])[0]
        # # ────────────────────────────────────────────────────────────────────
        total_visits = sum(score for move, score in move_score_list)
        prob_distribution = [score / total_visits for move, score in move_score_list]
        policy_map = self.generate_flattened_policy_board_map_for_neural_net(move_score_list, prob_distribution)
        print("id(self) =", id(self))
        print("id(root.state) =", id(mcts.root.state))
        return best_move, policy_map


    # To be used to prevent our alpha beta minimax from going till the end
    # and get slowed down instead we can try and use a heuristic function to look for what seems like a better position
    # only needed when game size is greater than 3 otherwise exhaustive search does the trick
    def heuristically_evaluate_board(self):
        multiplier = 0.18
        fork_multiplier = 0.28
        heuristic_value_diagonals = clamp(self.calculate_heuristic_value_diagonals())
        heuristic_value_rows = clamp(self.calculate_heuristic_value_rows())
        heuristic_value_columns = clamp(self.calculate_heuristic_value_columns())
        heuristic_value_central_dominance = clamp(self.calculate_heuristic_value_central_dominance())
        heuristic_value_fork = clamp(self.calculate_heuristic_value_fork())
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
        empty_count = streak.count(".")
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
        self.central_heuristic_evaluation_map = score_template_map
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

