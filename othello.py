# Let us represent players with player 1 by 0 and player 2 by 1
# default setting to be against human player, and size and win_length to conventional 3*3
from copy import deepcopy

from boardgame import BoardGame
from common_utils import set_starting_othello_board
from constant_strings import TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS, MOVE_B, MOVE_W, OTHELLO_BOARD_SIZE, DIRECTIONS


class Othello(BoardGame) :
    def __init__(self, size=OTHELLO_BOARD_SIZE, temperature_control = TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS,vs_human = True, ai_player_code = None, simulation_mode = False, ai_type = None):
        super().__init__(size=size, vs_human=vs_human, ai_player_code=ai_player_code, ai_type=ai_type,
                         simulation_mode=simulation_mode)
        self.board = set_starting_othello_board()
        # Note this has been set for simulation purposes
        self.search_depth = 2
        self.temperature_control = temperature_control
        self.logging_mode = True
        # attribute to control who made the last move
        self.last_moved = 0
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
        cloned.search_depth = self.search_depth
        cloned.central_heuristic_evaluation_map = deepcopy(self.central_heuristic_evaluation_map)
        # Disable logging for simulation
        cloned.logging_mode = False
        return cloned

    # get the amount of temperature control needed. Comes into picture during alpha beta pruning based engine
    def get_temperature_control(self):
        return self.temperature_control

    def set_temperature_control(self, custom_temperature):
        self.temperature_control = custom_temperature

    def get_last_player(self):
        return self.last_moved

    def set_last_player(self, player_code):
        return self.last_moved

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
            self.display_board()
            # increment total move counter by 1
            self.increment_total_move_count()
            self.implement_flip(current_board_state, move_coordinates[0], move_coordinates[1], player_symbol, opponent_symbol)



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


    def implement_flip(self,board, x, y, player_symbol, opponent_symbol):
        # now that move is made we need to check for flips and modify board
        flip_candidates = self.get_flip_candidates(board, x, y, player_symbol, opponent_symbol)
        if len(flip_candidates) == 0:
            self.selective_print("No flips needed!")
        for flip_candidates in flip_candidates:
            x,y = flip_candidates
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




