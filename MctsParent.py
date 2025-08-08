from abc import abstractmethod

from Neural_Net_Utils import prepare_neural_net_instance
from Node import Node
from common_utils import board_hash


class MctsParent:
    def __init__(self, root, game_instance):
        self.root = Node(state=game_instance)
        # just a hashed version of the current board state when MCTS is called
        self.hashed_root = board_hash(game_instance.get_current_board_state(), game_instance.current_player())
        # removed the hardcoded game name
        self.neural_net = prepare_neural_net_instance(game=game_instance.game_name, size = game_instance.get_board_size(), ai_type = game_instance.get_AI_type())
        # # # Cache for batch predictions to avoid repeated tensor conversions
        # self._prediction_cache = {}
        self.mcts_transposition_table = game_instance.mcts_transposition_table


    def get_root(self):
        return self.root

    def get_neural_net(self):
        return self.neural_net

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def expansion(self, parent):
        pass

    @abstractmethod
    def exploitation(self, parent):
        pass

    @abstractmethod
    def backtracking(self, child, value):
        pass

    # MIN_GAME_SIM_BENCHMARK_MCTS used for simulations runs
    # MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS used for vs human play
    def commence_mcts_for_selfplay(self, max_runs):
        for number in range(max_runs):
            parent = self.selection()
            child = self.expansion(parent) or parent
            value = self.exploitation(child)
            self.backtracking(child, value)