from abc import ABC, abstractmethod

from common_utils import board_hash
from constant_strings import CONCLUSIVE_RESULT_MULTIPLIER


class BoardGame(ABC):
    def __init__(self, size, vs_human, ai_player_code, ai_type, simulation_mode = False):
        self.size = size
        self.vs_human = vs_human
        self.ai_player_code = ai_player_code
        # to select the type of AI bot to play against
        self.ai_type = ai_type
        # board can be set differently in different games so better to keep as None
        self.board = None
        self.total_moves = 0
        # we can use this to store game_states and moves played
        self.move_list  = []
        self.simulation_mode = simulation_mode
        # transposition tables
        # two different transposition tables set up purely for separation of concerns
        # ab pruning
        self.transposition_table = {}
        # mcts
        self.mcts_transposition_table = {}

    def get_board_size(self):
        return self.size

    # to get AI player code
    def get_AI_player_code(self):
        return self.ai_player_code

    # To get the current board_state
    def get_current_board_state(self):
        return self.board

    def increment_total_move_count(self):
        self.total_moves += 1
        return self.total_moves

    def decrement_total_move_count(self):
        self.total_moves -= 1
        return self.total_moves

    # to prevent unnecessary prints by simulation instance
    def selective_print(self, *args, **kwargs):
            if not getattr(self, "logging_mode", True):
                return
            print(*args, **kwargs)

    # check if attempted position is occupied or not
    def is_occupied(self,move_coordinates):
        current_board_state = self.get_current_board_state()
        # attempted position is not empty, so we shouldn't allow the move
        if current_board_state[move_coordinates[0]][move_coordinates[1]] != ".":
            self.selective_print("The position is occupied try a different position!")
            return True
        return False

    # To display the board to the user
    def display_board(self):
        current_board_state = self.get_current_board_state()
        self.selective_print("\nCurrent board:")
        for row in current_board_state:
            self.selective_print(" | ".join(cell if cell != '' else ' ' for cell in row))

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
                self.selective_print("Please provide only and exactly 2 co-ordinates, indexed from 0 indicating where u want ur move to be placed")
        return result

    # Our detect win loss returns 1, -1 and 0 when player 1 wins,player 2 wins and draw respectively
    # using it directly in minimax can cause issues since when AI is player 2 it would want to maximize its score
    # but doing it based on the detect_win_loss function would give wrong results
    def generate_win_loss_metrics_wrt_AI(self, outcome):
        ai_player_code = self.get_AI_player_code()
        if outcome is None:
            return None
        if outcome == 0:
            return 0
        if ai_player_code == 1:
            if outcome in [-1,1]:
                return -1 * outcome * CONCLUSIVE_RESULT_MULTIPLIER
            else:
                return outcome * CONCLUSIVE_RESULT_MULTIPLIER
        else:
            return outcome * CONCLUSIVE_RESULT_MULTIPLIER

    # ab pruning
    # lookup for current position for the current_depth in the transposition table
    def fetch_existing_hash(self, depth_to_result):
        current_board_state = self.get_current_board_state()
        # compute hash key
        key = board_hash(current_board_state, self.current_player())
        # --- TRANSPOSE TABLE: check for cached score ---
        cached = self.transposition_table.get(key)
        if cached is not None:
            # basically we only need to use it if the cached evaluation
            # was done at a depth greater than the current one or equal to it
            # otherwise it is not trust worthy
            if cached["depth"] >= depth_to_result:
                return cached["score"]
        return None

    # ab pruning only
    # store the current game info in transposition table
    def store_in_transposition_table(self, score, depth_to_result):
        cached = self.fetch_existing_hash(depth_to_result)
        key = board_hash(self.get_current_board_state(), self.current_player())
        # Only store if no entry exists OR current depth is deeper/equal
        if cached is None:
            self.transposition_table[key] = {
                "depth": depth_to_result,
                "score": score
            }

    def fit_to_ai_metrics(self, outcome,depth_to_result):
        ai_adjusted_outcome = self.generate_win_loss_metrics_wrt_AI(outcome)
        if ai_adjusted_outcome is not None:
            # Prefer faster win / slower loss
            if ai_adjusted_outcome > 0:
                score = ai_adjusted_outcome - depth_to_result
            elif ai_adjusted_outcome < 0:
                score = ai_adjusted_outcome + depth_to_result
            else:
                score = ai_adjusted_outcome
            return score

    # clear the transposition table for the current game instance
    def clear_transposition_table(self):
        self.transposition_table.clear()


    @abstractmethod
    def current_player(self):
        pass

    @abstractmethod
    def clone_instance(self):
        pass

    @abstractmethod
    def detect_win_loss(self):
        pass

    @abstractmethod
    def human_make_move(self):
        pass

    @abstractmethod
    def run_game(self):
        pass
