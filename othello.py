# Let us represent players with player 1 by 0 and player 2 by 1
# default setting to be against human player, and size and win_length to conventional 3*3
import math
import random
import time
from copy import deepcopy

import numpy as np

from Mcts import Mcts
from MctsOthello import MctsOthello
from boardgame import BoardGame
from common_utils import set_starting_othello_board, othello_camp
from constant_strings import TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS, MOVE_B, MOVE_W, OTHELLO_BOARD_SIZE, DIRECTIONS, \
    COIN_PARITY_HEURISTIC_MULTIPLIER, MOBILITY_HEURISTIC_MULTIPLIER, STABILITY_HEURISTIC_MULTIPLIER, \
    CORNER_CAPTURE_HEURISTIC_MULTIPLIER, CAPTURED_CORNER_WEIGHT, POTENTIAL_CORNER_WEIGHT, UNLIKELY_CORNER_WEIGHT, \
    ALPHA_BETA_PRUNING, MCTS, MCTS_NN, MAX_PLY_DEPTH_TTT, MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS_TTT, \
    MIN_GAME_SIM_BENCHMARK_MCTS, \
    GAME_OTHELLO, MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS_OTHELLO, ASPIRATION_WINDOW_MULTIPLIER, \
    ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT, MAX_PLY_DEPTH_OTHELLO, INF


class Othello(BoardGame) :
    def __init__(self, size=OTHELLO_BOARD_SIZE, temperature_control = TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS,vs_human = True, ai_player_code = None, simulation_mode = False, ai_type = None):
        super().__init__(size=size, vs_human=vs_human, ai_player_code=ai_player_code, ai_type=ai_type,
                         simulation_mode=simulation_mode)
        self.game_name = GAME_OTHELLO
        self.board = set_starting_othello_board()
        # Note this has been set for simulation purposes
        self.max_iterative_depth = MAX_PLY_DEPTH_OTHELLO
        self.temperature_control = temperature_control
        self.logging_mode = True
        # attribute to control who made the last move
        # clever hack to make sure black moves first by starting the game by assuming white has already
        # moved even if they haven't
        self.last_moved = 1
        # to indicate which player moves Black and which moves White
        # currently the player who goes first gets B and the one who goes second gets W
        self.assigned_move = {
            0 : MOVE_B,
            1 : MOVE_W
        }
        self.result_map = {
            0 : "Draw",
            1 : "Win for Player 1",
            -1 : "Win for Player 2"
        }
        self.central_heuristic_evaluation_map = None
        self.match_result = None

    # to aid use in MCTS algorithm
    # trying with a slightly light-weight clone in comparison to previous approach
    # previous approach deep-copied the whole instance which created/could create a memory overload
    def clone_instance(self):
        cloned = Othello(
            size=self.size,
            temperature_control=self.temperature_control,
            vs_human=self.vs_human,
            ai_player_code=self.ai_player_code,
            simulation_mode=self.simulation_mode,
            ai_type=self.ai_type
        )
        # Deepcopy only essential mutable attributes
        cloned.board = deepcopy(self.board)
        cloned.total_moves = self.total_moves
        cloned.max_iterative_depth = self.max_iterative_depth
        cloned.central_heuristic_evaluation_map = deepcopy(self.central_heuristic_evaluation_map)
        # Note this is necessary since we only use cloning in othello
        # otherwise
        cloned.transposition_table = self.transposition_table
        # Disable logging for simulation
        cloned.logging_mode = False
        cloned.last_moved = self.last_moved
        return cloned

    # method added coz undo is hard in Othello. Can be used in Alpha Beta Search
    def clone_board(self):
        return deepcopy(self.board)

    # get the amount of temperature control needed. Comes into picture during alpha beta pruning based engine
    def get_temperature_control(self):
        return self.temperature_control

    def set_temperature_control(self, custom_temperature):
        self.temperature_control = custom_temperature

    def get_last_player(self):
        return self.last_moved

    def set_last_player(self, player_code):
        self.last_moved = player_code
        return self.last_moved

    # to get the provided player's symbol
    def get_player_symbol(self, player_code):
        return self.assigned_move[player_code]

    # basically helps to get count of number of pieces of a given color on the board
    def get_player_piece_count(self, player_piece):
        current_board_state = self.get_current_board_state()
        current_board_size = self.get_board_size()
        piece_count = 0
        for row_index in range(current_board_size):
            for column_index in range(current_board_size):
                if current_board_state[row_index][column_index] == player_piece:
                    piece_count += 1
        return piece_count

    # basically helps to get count of number of pieces of a given color on the board
    def get_player_piece_locations(self, player_piece):
        current_board_state = self.get_current_board_state()
        current_board_size = self.get_board_size()
        result = []
        for row_index in range(current_board_size):
            for column_index in range(current_board_size):
                if current_board_state[row_index][column_index] == player_piece:
                    result.append((row_index, column_index))
        return result

    # can't get possible moves based on simple empty spaces like tictactoe
    def get_possible_moves(self, player_code):
        current_board_state = self.get_current_board_state()
        current_board_size = self.get_board_size()
        # we need to modify this for later
        # for now we can return empty list
        player_symbol = self.assigned_move[player_code]
        opponent_symbol = self.assigned_move[1-player_code]
        result = []
        for row_index in range(current_board_size):
            for column_index in range(current_board_size):
                if self.is_valid_move(current_board_state, row_index, column_index, player_symbol, opponent_symbol):
                    # don't accidentally use square braces since we used tuple
                    # format for tictactoe as well
                    result.append((row_index, column_index))
        return result

    # player whose turn it is to make a move
    # 0 means first player, 1 means second player
    # can't use the approach in tictactoe to solely rely on move count
    def current_player(self):
        last_player = self.get_last_player()
        next_player = 1 - last_player
        # the next player has no moves
        if self.get_possible_moves(next_player):
            return next_player
        # checking if the player who just played has any legal moves
        elif self.get_possible_moves(last_player):
            return last_player
        # if Neither of them have a move we can return -1
        #  this means game is over
        else:
            return -1

    # just to check if a given co-ordinate is a potential candidate for a move
    def is_valid_move(self, board, x, y, player_symbol, opponent_symbol):
        size = len(board)
        # means that position is occupied so no point checking further
        if self.is_occupied([x,y]):
            return False
        for dx, dy in DIRECTIONS:
            updated_x, updated_y = x + dx, y + dy
            found_opponent = False
            while 0 <= updated_x < size and 0 <= updated_y < size:
                cell = board[updated_x][updated_y]
                # this means we should try to proceed ahead and have scope
                # to flank/sandwich more than 1 opponent material
                if cell == opponent_symbol:
                    found_opponent = True
                    # basically we proceed in that direction
                    updated_x += dx
                    updated_y += dy
                elif cell == player_symbol:
                    if found_opponent:
                        return True
                    break
                # this case shouldn't really happen since we covered the
                # empty positions are already checked at the beginning
                else:  # Empty or invalid
                    break
        return False

    # to use input position from above and update board
    # if the given input position is not filled
    # Note to be used to make move by humans
    # returning the code of the player last making the move for convenience
    def human_make_move(self):
        invalid_move_check = True
        board_size = self.get_board_size()
        current_board_state = self.get_current_board_state()
        # shows whether the player is to repeat a move or should the turn go to the other player
        player_to_move = self.current_player()
        player_symbol = self.assigned_move[player_to_move]
        opponent_symbol = self.assigned_move[1-player_to_move]
        move_coordinates = ""
        possible_moves = self.get_possible_moves(player_to_move)
        while invalid_move_check:
            move_coordinates = self.get_input_position()
            if min(move_coordinates[0], move_coordinates[1]) < 0 or max(move_coordinates[0], move_coordinates[1]) > board_size-1:
                print("The provided move co-ordinates are out of the range of values for this board size!. Please enter something within that range! ")
                continue
            if tuple(move_coordinates) in possible_moves:
                invalid_move_check = False
            else:
                self.selective_print("Please make a valid move into a currently unoccupied position")
        current_board_state[move_coordinates[0]][move_coordinates[1]] = self.assigned_move[player_to_move]
        # increment total move counter by 1
        self.increment_total_move_count()
        # just to show before and after
        self.display_board()
        time.sleep(0.5)
        self.implement_flips(move_coordinates[0], move_coordinates[1], player_symbol, opponent_symbol)
        self.display_board()
        self.set_last_player(player_to_move)
        print(f"Human played {move_coordinates[0], move_coordinates[1]}")



    # list of tuples of potential flip candidates given the latest move x,y by the player
    # need to be run after every move
    def get_flip_candidates(self,board, x, y, player_symbol, opponent_symbol):
        size = self.get_board_size()
        flips = []
        # basically we move in 8 directions from any point and check
        for dx, dy in DIRECTIONS:
            path = []
            updated_x, updated_y = x + dx, y + dy
            # to address for the edge case and we don't go out of bounds
            while 0 <= updated_x < size and 0 <= updated_y < size:
                cell = board[updated_x][updated_y]
                # this could be a potential flip candidate
                if cell == opponent_symbol:
                    path.append((updated_x, updated_y))
                    updated_x += dx
                    updated_y += dy
                # if we reached our piece, everything of opposite color upto now can be flipped
                # can see from ordering that loop breaks when empty spot encountered
                elif cell == player_symbol:
                    if path:
                        flips.extend(path)
                    break
                # Hit empty spot, so no point checking in that direction
                else:
                    break
        return flips


    def implement_flips(self, x, y, player_symbol, opponent_symbol):
        # now that move is made we need to check for flips and modify board
        board = self.get_current_board_state()
        flip_candidates = self.get_flip_candidates(board, x, y, player_symbol, opponent_symbol)
        if len(flip_candidates) == 0:
            self.selective_print("No flips needed!")
        for flip_candidate in flip_candidates:
            x,y = flip_candidate
            # overturning the opponent pieces into ours
            self.board[x][y] = player_symbol


    def detect_win_loss(self):
        # basically the case where neither player has any moves left so game has ended
        if self.current_player() == -1:
            player_1_piece_count = self.get_player_piece_count(MOVE_B)
            player_2_piece_count = self.get_player_piece_count(MOVE_W)
            if player_1_piece_count > player_2_piece_count:
                return 1
            elif player_1_piece_count < player_2_piece_count:
                return -1
            # draw case
            else:
                return 0
        # game is still going on
        else:
            return None



    # # just for MCTS purposes. Light-weight with an added minimal check to avoid
    # # immediate losses
    # def make_pseudo_random_move(self):
    #     current_player = self.current_player
    #     possible_moves = self.get_possible_moves(current_player)
    #     # a bit unclear one exact role. More like an additional safety check
    #     if not possible_moves:
    #         return None
    #     player_to_move = self.current_player()
    #     immediate_result_move = self.check_immediate_result(possible_moves)
    #     # no move which leads to an immediate result so choose a random one
    #     move = immediate_result_move if immediate_result_move is not None else random.choice(possible_moves)
    #     self.get_current_board_state()[move[0]][move[1]] = self.get_player_symbol(player_to_move)
    #     # increment total move counter by 1
    #     self.increment_total_move_count()
    #     return move


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
        best_score =-INF if isMax else INF
        # get the current turn player's symbol
        current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
        opponent_player = 1 - current_player
        current_symbol = self.get_player_symbol(current_player)
        opponent_symbol = self.get_player_symbol(opponent_player)
        possible_moves = self.get_possible_moves(current_player)

        for move in possible_moves:
            clone_game_instance = self.clone_instance()
            clone_game_instance.board[move[0]][move[1]] = current_symbol
            clone_game_instance.increment_total_move_count()
            # we need the flip in Othello
            clone_game_instance.implement_flips(move[0], move[1], current_symbol, opponent_symbol)

            # Recursively call minimax for the next player's turn
            score = clone_game_instance.minimax(not isMax)
            # Update best score
            if isMax:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

        return best_score


    def minimax_with_alpha_beta_pruning(
            self, isMax: bool, depth_to_result: int,
            alpha: float = -INF, beta: float = INF):
        # Transposition-table probe
        cached_score = self.fetch_existing_hash(depth_to_result)
        if cached_score is not None:
            return cached_score

        #  Terminal check
        outcome = self.detect_win_loss()
        terminal_score = self.fit_to_ai_metrics(outcome, depth_to_result)
        if terminal_score is not None:
            self.store_in_transposition_table(terminal_score, depth_to_result)
            return terminal_score

        # Initialise best score
        best_score = -INF if isMax else INF

        # Identify the side to move and generate its legal moves
        current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
        current_symbol = self.get_player_symbol(current_player)
        opponent_symbol = self.get_player_symbol(1 - current_player)
        possible_moves = self.get_possible_moves(current_player)



        # Loop through children
        for move in possible_moves:
            cloned_child = self.clone_instance()
            cloned_child.board[move[0]][move[1]] = current_symbol
            cloned_child.implement_flips(move[0], move[1],
                                         current_symbol, opponent_symbol)
            cloned_child.total_moves += 1
            cloned_child.last_moved = current_player

            score = cloned_child.minimax_with_alpha_beta_pruning(
                not isMax, depth_to_result + 1, alpha, beta)

            if isMax:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)

            if alpha >= beta:  # α-β cut-off
                break

        # 6) Cache and return
        self.store_in_transposition_table(best_score, depth_to_result)
        return best_score

        # To be used to prevent our alpha beta minimax from going till the end
        # and get slowed down instead we can try and use a heuristic function to look for what seems like a better position
        # only needed when game size is greater than 3 otherwise exhaustive search does the trick

    def calculate_coin_parity_heuristics(self,curr_player):
        player_piece= self.get_player_symbol(curr_player)
        opp_piece = self.get_player_symbol(1 - curr_player)
        curr_player_coins = self.get_player_piece_count(player_piece)
        opp_coins = self.get_player_piece_count(opp_piece)
        total_coins = curr_player_coins + opp_coins
        # should never happen since sum of coins would never be 0 not even at start
        # just handling for safety
        if total_coins == 0:
            print("Error encountered")
            return
        return (100 * (curr_player_coins - opp_coins))/total_coins

    # only considering actual mobility
    # no consideration of potential mobility
    # architectural trade-off made here
    def calculate_mobility_heuristics(self, curr_player):
        opp_code = 1 - curr_player
        curr_play_move_count = len(self.get_possible_moves(curr_player))
        opp_move_count = len(self.get_possible_moves(opp_code))

        if curr_play_move_count + opp_move_count == 0:
            # avoid div-by-zero – completely locked board has zero mobility
            return 0.0

        return 100.0 * (curr_play_move_count - opp_move_count) / (curr_play_move_count + opp_move_count)


    # only considering actual stability
    # no consideration of potential stability
    # architectural trade-off made here
    # only checking unbroken streaks from confirmed stable pieces which are corner pieces
    def calculate_stability_heuristics(self, curr_player):
        board_size = self.get_board_size()
        opp_code = 1 - curr_player
        curr_player_symbol = self.get_player_symbol(curr_player)
        opp_symbol = self.get_player_symbol(opp_code)
        curr_player_score = 0
        opp_score = 0
        board_corner_positions = [(0, 0), (0, board_size - 1), (board_size -1, 0), (board_size -1, board_size -1)]
        for corner_x, corner_y in board_corner_positions:
            cell = self.board[corner_x][corner_y]
            # current_player has a piece on this corner
            if cell == ".":
                # Nothing to do. We can skip
                continue
            elif cell == curr_player_symbol:
                curr_player_score += 1
                curr_player_score += self.calculate_uninterrupted_streak_from_corner(curr_player_symbol,corner_x, corner_y)
            # opponent symbol case
            else:
                opp_score += 1
                opp_score += self.calculate_uninterrupted_streak_from_corner(opp_symbol,corner_x, corner_y)
        total = curr_player_score + opp_score
        if total == 0:  # no stable discs on the board
            return 0.0

        # signed percentage advantage  ∈ [-100, +100]
        return 100.0 * (curr_player_score - opp_score) / total



    def calculate_uninterrupted_streak_from_corner(self, player_symbol, corner_x, corner_y):
        board_size = self.get_board_size()
        player_score = 0
        for dx, dy in self.get_differential_corner_coordinates([corner_x, corner_y]):
            x, y = corner_x + dx, corner_y + dy
            while 0 <= x < board_size and 0 <= y < board_size:
                # we only go on till we find an uninterrupted streak
                if self.board[x][y] == player_symbol:
                    player_score += 1
                else:
                    break
                x += dx
                y += dy
        return player_score



    # helper method to make small moves from corners
    def get_differential_corner_coordinates(self, move):
        board_size = self.get_board_size()
        x, y = move
        if move == [0, 0]:
            return [(1,0), (0,1), (1,1)]
        if move == [board_size-1, board_size-1]:
            return [(-1,0), (0,-1), (-1,-1)]
        if move == [board_size-1, 0]:
            return [(-1,0), (0,1), (-1,1)]
        if move == [0, board_size-1]:
            return [(1,0), (0,-1), (1,-1)]

    # research papers show this to be the most prominent heuristic. So actual, potential and unlikely
    # have all been considered
    def calculate_corner_capture_heuristics(self, curr_player):
        opp_code = 1 - curr_player
        curr_player_symbol = self.get_player_symbol(curr_player)
        opp_symbol = self.get_player_symbol(opp_code)
        curr_player_score = 0
        opp_score = 0
        board_corner_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
        curr_player_legal_moves = self.get_possible_moves(curr_player)
        opp_legal_moves = self.get_possible_moves(opp_code)
        for corner_x, corner_y in board_corner_positions:
            cell = self.board[corner_x][corner_y]
            if cell == curr_player_symbol:
                curr_player_score += CAPTURED_CORNER_WEIGHT
            elif cell == opp_symbol:
                opp_score += CAPTURED_CORNER_WEIGHT
            elif (corner_x, corner_y) in curr_player_legal_moves:
                curr_player_score += POTENTIAL_CORNER_WEIGHT
            elif (corner_x, corner_y) in opp_legal_moves:
                opp_score += POTENTIAL_CORNER_WEIGHT
            else:
                curr_player_score += UNLIKELY_CORNER_WEIGHT / 2
                opp_score += UNLIKELY_CORNER_WEIGHT / 2
        total = curr_player_score + opp_score
        # if conditioning purely to avoid division by 0 errors
        return 0.0 if total == 0 else 100.0 * (curr_player_score - opp_score) / total



    def heuristically_evaluate_board(self):
        current_player = self.current_player()
        coin_parity_heuristics = othello_camp(self.calculate_coin_parity_heuristics(current_player))
        mobility_heuristics = othello_camp(self.calculate_mobility_heuristics(current_player))
        stability_heuristics = othello_camp(self.calculate_stability_heuristics(current_player))
        corner_capture_heuristics = othello_camp(self.calculate_corner_capture_heuristics(current_player))
        return (COIN_PARITY_HEURISTIC_MULTIPLIER * coin_parity_heuristics) + (MOBILITY_HEURISTIC_MULTIPLIER * mobility_heuristics) + (STABILITY_HEURISTIC_MULTIPLIER * stability_heuristics) + (CORNER_CAPTURE_HEURISTIC_MULTIPLIER * corner_capture_heuristics)


    # heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
    # method to get a numerical metric for the next possible move with alpha beta pruning
    # aim to prune branches where alpha >= beta to diminish search space
    # we add a depth_to_result parameter to allow for setting of search depths
    def heuristic_minimax_with_alpha_beta_pruning(self, isMax, max_ai_search_depth, depth_to_result, alpha=-INF, beta=INF):
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
            best_score = -INF if isMax else INF
            # get the current turn player's symbol
            current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
            opponent_player = 1 - current_player
            current_symbol = self.get_player_symbol(current_player)
            opponent_symbol = self.get_player_symbol(opponent_player)
            possible_moves = self.get_possible_moves(current_player)

            for move in possible_moves:
                clone_game_instance = self.clone_instance()
                clone_game_instance.board[move[0]][move[1]] = current_symbol
                clone_game_instance.increment_total_move_count()
                # we need the flip in Othello
                clone_game_instance.implement_flips(move[0], move[1], current_symbol,
                                                    opponent_symbol)
                # Recursively call minimax for the next player's turn
                score = clone_game_instance.heuristic_minimax_with_alpha_beta_pruning(not isMax, max_ai_search_depth - 1,
                                                                       depth_to_result + 1, alpha, beta)

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







    # trade off here. We are assuming the board state to be the same when opponent makes the move
    # and checking if there is anything they can do the
    # making it more complicated would make it an N^2 operation. We keep it N for now
    def check_immediate_result(self, possible_moves):
        player = self.current_player()
        opponent = 1 - player
        player_symbol = self.get_player_symbol(player)
        opponent_symbol = self.get_player_symbol(opponent)

        for move in possible_moves:
            cloned_instance = self.clone_instance()
            cloned_instance.board[move[0]][move[1]] = player_symbol
            cloned_instance.implement_flips(move[0], move[1], player_symbol, opponent_symbol)
            cloned_instance.total_moves += 1
            cloned_instance.last_moved = player

            result = cloned_instance.detect_win_loss()  # 1 = Black wins, -1 = White wins, None = unfinished
            if (result == 1 and player_symbol == MOVE_B) or \
                    (result == -1 and player_symbol == MOVE_W):
                return move  # play the winning move

        # No immediate wins found so let us try and block any immediate wins for our opponent
        for move in possible_moves:
            cloned_instance = self.clone_instance()
            cloned_instance.board[move[0]][move[1]] = opponent_symbol
            # implementing flips for opponent
            cloned_instance.implement_flips(move[0], move[1], opponent_symbol, player_symbol)
            cloned_instance.total_moves += 1
            cloned_instance.last_moved = opponent
            result = cloned_instance.detect_win_loss()
            if (result == 1 and opponent_symbol == MOVE_B) or \
                    (result == -1 and opponent_symbol == MOVE_W):
                # block their winning move
                return move
        return None

    # this method decides whether to call minimax/ab pruning or
    def ai_skip_move_ab_flow_adjuster(self, cloned_child, current_player, isMax, depth_to_result, alpha, beta):
        opponent_symbol = cloned_child.get_player_symbol(1 - current_player)
        if len(cloned_child.get_possible_moves(opponent_symbol)) != 0:
            cloned_child.last_moved = current_player
            score = cloned_child.minimax_with_alpha_beta_pruning(
                not isMax, depth_to_result + 1, alpha, beta)
        else:
            score = cloned_child.minimax_with_alpha_beta_pruning(
                isMax, depth_to_result + 1, alpha, beta)
        return score

    def ai_skip_move_ab_heuristic_flow_adjuster(self, cloned_child, current_player, isMax, depth_to_result, alpha, beta):
        opponent_symbol = cloned_child.get_player_symbol(1 - current_player)
        if len(cloned_child.get_possible_moves(opponent_symbol)) != 0:
            cloned_child.last_moved = current_player
            score = cloned_child.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
                not isMax, depth_to_result + 1, alpha, beta)
        else:
            score = cloned_child.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
                isMax, depth_to_result + 1, alpha, beta)
        return score

    # basically using the move evaluation found in the previous step to choose an optimal move by evaluating
    # for each move possible given current empty spaces
    def select_optimal_ai_move_with_temperature_control(self):
        start_time = time.time()
        temperature_control = self.get_temperature_control()
        current_board_size = self.get_board_size()
        ai_player_code = self.ai_player_code
        ai_symbol = self.get_player_symbol(ai_player_code)
        opponent_symbol = self.get_player_symbol(1 - ai_player_code)
        possible_moves = self.get_possible_moves(ai_player_code)
        if not possible_moves:  # forced pass
            zero_policy = np.zeros(self.size * self.size, np.float32)
            return None, zero_policy
        # choosing the lowest abs value possible initially
        best_score = -INF
        best_follow_up_move = None
        move_score_list = []
        max_possible_moves = current_board_size*current_board_size
        for next_move in possible_moves:
            # trying each of the list of possible empty space given the current board state
            cloned_instance = self.clone_instance()
            cloned_instance.board[next_move[0]][next_move[1]] = ai_symbol
            cloned_instance.implement_flips(next_move[0], next_move[1], ai_symbol, opponent_symbol)
            cloned_instance.increment_total_move_count()
            opp_possible_moves = cloned_instance.get_possible_moves(1-ai_player_code)
            ai_adjusted_player_assignment = None
            if len(opp_possible_moves) != 0:
                ai_adjusted_player_assignment = False
            else:
                ai_adjusted_player_assignment = True
            cloned_instance.last_moved = ai_player_code
            score = 0
            remaining = max_possible_moves - cloned_instance.total_moves
            # Slightly different logic than tictactoe possible moves might jump up and down and do not directly correspond
            # to the number of remaining pieces
            if current_board_size <= 4 or remaining <= 12:
                score = cloned_instance.minimax_with_alpha_beta_pruning(ai_adjusted_player_assignment, 1, -INF, INF)

            else:
                # without iterative deepening
                # score = self.heuristic_minimax_with_alpha_beta_pruning(False,search_depth, 1, -INF, INF)

                # with iterative deepening
                score = cloned_instance.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
                    isMax=ai_adjusted_player_assignment,  # if you're simulating the opponent's move
                    max_ply=self.max_iterative_depth,  # depth limit — can tweak based on board size/time
                    depth_to_result=1  # always starts from 1
                )
            move_score_list.append((next_move, score))
        probability_distribution = self.generate_probability_distribution_with_temperature(move_score_list,
                                                                                           temperature_control)
        self.selective_print(f"Probability distribution is - {probability_distribution}")
        probability_based_idx = np.random.choice(len(move_score_list), p=probability_distribution)
        best_follow_up_move = move_score_list[probability_based_idx][0]
        best_score = move_score_list[probability_based_idx][1]
        self.selective_print(f"Evaluation Score of Position is {best_score}")
        policy_map = self.generate_flattened_policy_board_map_for_neural_net(move_score_list=move_score_list,
                                                                             probability_distribution=probability_distribution)
        end_time = time.time()
        print(f"AI move computation took {end_time - start_time:.3f} seconds.")
        return best_follow_up_move, policy_map

    # to be implemented
    # Formula used in the Alpha Go paper prioritizing visit count as the most viable metric πₐ ∝ N(s, a)¹/τ
    # Long story short, ove a decent number of simulations visit count is the best metric to look for
    def select_optimal_ai_move_mcts(self):
        start_time = time.time()
        # this will help us not to use too much load in doing and undoing on the
        # original instance and just perform everything on a clone
        simulated_mode = self.get_game_mode()
        cloned_instance = self.clone_instance()
        current_player = cloned_instance.ai_player_code
        # choosing the lowest abs value possible initially
        best_score = -INF
        best_follow_up_move = None
        mcts = MctsOthello(root=None, game_instance=cloned_instance)
        # variable which assigns different number of max runs based on self or vs human play
        max_runs = MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS_OTHELLO if simulated_mode == False else MIN_GAME_SIM_BENCHMARK_MCTS
        mcts.commence_mcts_for_selfplay(max_runs=max_runs)
        parent_node = mcts.get_root()
        children = parent_node.get_children()
        # just added for safety
        if not children:
            raise RuntimeError("MCTS produced no children – check rollout budget.")
        # ───────── DEBUG PRINT: show root statistics once per AI move ─────────
        print("\n=== ROOT AFTER SEARCH FINISHED ===")
        for mv, node in children.items():
            print("[root] move", mv,
                  "visits =", node.get_visits(),
                  "wins   =", node.get_wins())
        print("==================================\n")
        # ───────────────────────────────────────────────────────────────────────

        move_score_list = [(move, children[move].get_visits()) for move in children]
        # basically getting max based on move visit counts
        best_move = max(move_score_list, key=lambda x: x[1])[0]
        total_visits = sum(score for move, score in move_score_list)
        prob_distribution = None
        if total_visits == 0:  # safety fallback
            prob_distribution = [1.0 / len(move_score_list)] * len(move_score_list)
        else:
            prob_distribution = [score / total_visits for move, score in move_score_list]
        policy_map = self.generate_flattened_policy_board_map_for_neural_net(move_score_list, prob_distribution)
        print("id(self) =", id(self))
        print("id(root.state) =", id(mcts.root.state))
        end_time = time.time()
        print(f"AI move computation took {end_time - start_time:.3f} seconds.")
        return best_move, policy_map

    def make_ai_move(self,ai_type):
        # Neural Network needs the board state prior to the move, in a flattened form
        # so it can be used as a vector/list/tensor along with the policy map for that corresponding state
        pre_move_flattened_state_2d = "".join(str(cell) for row in self.board for cell in row)
        ai_player_code = self.current_player()
        opponent_code = 1 - ai_player_code
        best_move = None
        policy_map = None
        ai_player_symbol = self.get_player_symbol(ai_player_code)
        opponent_symbol = self.get_player_symbol(opponent_code)
        if ai_type == ALPHA_BETA_PRUNING:
            best_move, policy_map = self.select_optimal_ai_move_with_temperature_control()
            if best_move is None:  # no legal moves → pass
                self.last_moved = self.current_player()  # mark who just “moved”
                return  # skip placement / flips
            # -----------
        if ai_type in [MCTS, MCTS_NN]:
            # this check is very very very important and has been added to prevent
            # missing immediate wins and losses
            conclusive_result = self.check_immediate_result(self.get_possible_moves(ai_player_code))
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
        # we only need this if it is playing against humans otherwise this serves no benefit
        if not self.simulation_mode:
            self.display_board()
            time.sleep(0.5)
        self.implement_flips(best_move[0], best_move[1], ai_player_symbol, opponent_symbol)
        self.display_board()
        self.last_moved = ai_player_code
        print(f"AI played {best_move[0], best_move[1]}")

    def make_pseudo_random_move(self):
        current_player = self.current_player()
        possible_moves = self.get_possible_moves(current_player)

        # Pass: no legal moves
        if not possible_moves:
            return None

        chosen_move = self.check_immediate_result(possible_moves)
        if chosen_move is None:
            chosen_move = random.choice(possible_moves)

        player_symbol = self.get_player_symbol(current_player)
        opp_symbol = self.get_player_symbol(1 - current_player)

        # Place disc and flip
        row, column = chosen_move
        self.board[row][column] = player_symbol
        self.implement_flips(row, column, player_symbol, opp_symbol)

        # Bookkeeping
        self.total_moves += 1
        self.last_moved = current_player

        return chosen_move


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

    #  to run the game and link above methods together
    def run_game(self):
        simulation_mode = self.get_game_mode()
        ai_type = self.get_AI_type()
        # end_result of the game - 1, 0 and -1 indicating
        # victory for x, draw and defeat for x respectively
        result = None
        self.selective_print("Game has now begun")
        self.display_board()
        game_ongoing = True
        if simulation_mode:
            if ai_type == MCTS:
                game_ongoing = True
                while game_ongoing:
                    result = self.detect_win_loss()
                    # this check we have added since simulation loop can be entered
                    if result is not None:
                        game_ongoing = False
                        break
                    self.make_pseudo_random_move()
                    self.ai_player_code = self.current_player()
                    result = self.detect_win_loss()
                    if result is not None:
                        # if we don't break here it goes on to alternate the code even if the game has ended
                        break
                    self.ai_player_code = self.current_player()
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
                    # we can't simply alternate code like we did in tictactoe
                    # will have to switch
                    self.ai_player_code = self.current_player()
                    if result is not None:
                        game_ongoing = False
                        self.selective_print(result_map[result])
            else:
                pass
        # vs Human
        else:
            # Human vs Human
            if self.ai_player_code is None:
                self.selective_print("Human vs Human mode")
                while game_ongoing:
                    current_player = self.current_player()
                    self.selective_print(f"{self.get_player_symbol(current_player)} to move")
                    self.human_make_move()
                    result = self.detect_win_loss()
                    result_map = self.fetch_result_map()
                    if result is not None:
                        game_ongoing = False
                        self.selective_print(result_map[result])
            # # Human vs AI
            else:
                # would indicate the order
                ai_player_order = self.get_AI_player_code()
                self.selective_print(f"Human vs AI mode - {ai_type}")
                while game_ongoing:
                    # again can't do simple move count thingy
                    if self.current_player() == ai_player_order:
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

    def heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
            self, isMax, max_ply, depth_to_result,
            alpha=-INF, beta=INF):
        cached_score = self.fetch_existing_hash(depth_to_result)
        if cached_score is not None:
            return cached_score

        outcome = self.detect_win_loss()
        terminal_score = self.fit_to_ai_metrics(outcome, depth_to_result)
        if terminal_score is not None:
            self.store_in_transposition_table(terminal_score, depth_to_result)
            return terminal_score

        if max_ply == 0:
            static_score = self.heuristically_evaluate_board()
            self.store_in_transposition_table(static_score, depth_to_result)
            return static_score

        # Iterative deepening with aspiration-window
        prev_score = 0
        initial_window = 1
        best_score = -INF if isMax else INF
        ordered_moves = []

        for depth in range(1, max_ply + 1):
            window = initial_window
            alpha = prev_score - window
            beta = prev_score + window

            while True:
                window_low, window_high = alpha, beta
                current_move_scores = []
                current_best_score = -INF if isMax else INF

                player_code = self.ai_player_code if isMax else 1 - self.ai_player_code
                player_symbol = self.get_player_symbol(player_code)
                opp_symbol = self.get_player_symbol(1 - player_code)

                move_list = (self.get_possible_moves(player_code) if not ordered_moves
                             else [move for move, score in ordered_moves])

                for move in move_list:
                    cloned_child = self.clone_instance()
                    cloned_child.board[move[0]][move[1]] = player_symbol
                    cloned_child.implement_flips(move[0], move[1],
                                                 player_symbol, opp_symbol)
                    cloned_child.total_moves += 1
                    cloned_child.last_moved = player_code

                    score = cloned_child.heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
                        not isMax, depth - 1, depth_to_result + 1,
                        alpha, beta)

                    current_move_scores.append((move, score))

                    if isMax:
                        current_best_score = max(current_best_score, score)
                        alpha = max(alpha, current_best_score)
                    else:
                        current_best_score = min(current_best_score, score)
                        beta = min(beta, current_best_score)

                    if alpha >= beta:  # αβ cut-off
                        break

                # order moves for next iteration
                ordered_moves = sorted(current_move_scores,
                                       key=lambda x: x[1],
                                       reverse=isMax)
                best_score = current_best_score

                # Aspiration-window success?
                if window_low < current_best_score < window_high:
                    prev_score = current_best_score
                    break

                # Otherwise widen the window and repeat
                window *= ASPIRATION_WINDOW_MULTIPLIER
                if window > ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT:
                    alpha, beta = -INF, INF
                else:
                    alpha = prev_score - window
                    beta = prev_score + window

        self.store_in_transposition_table(best_score, depth_to_result)
        return best_score
