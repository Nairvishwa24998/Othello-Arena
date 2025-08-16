import time
from abc import abstractmethod

import numpy as np

from Neural_Net_Utils import prepare_neural_net_instance, flattened_board_to_tensor
from Node import Node
from common_utils import board_hash
from constant_strings import GAME_TICTACTOE, GAME_OTHELLO


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
        # added for testing
        self.pending_batch = []  # stores (leaf_node, encoded_board)
        self.BATCH_SIZE = 64  # tweak as needed

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
            # old version
            # parent = self.selection()
            # child = self.expansion(parent) or parent
            # value = self.exploitation(child)
            # self.backtracking(child, value)
            # new version
            parent = self.selection()
            child = self.expansion(parent) or parent
            game_name = child.state.game_name
            turn_to_move = None
            if game_name == GAME_OTHELLO:
                turn_to_move = child.state.current_player()
            if not getattr(child, "expanded_by_nn", False):
                flat_str = "".join(
                    str(cell) for row in child.state.board for cell in row
                )
                encoded = flattened_board_to_tensor(state_str=flat_str,game_name=game_name, turn_to_move=turn_to_move)
                self.pending_batch.append((child, encoded))
            else:
                # already has NN value (from TT) → rollout / heuristic
                value = self.exploitation(child)
                self.backtracking(child, value)
            # ── run batch when full or on final simulation ──────────────
            if len(self.pending_batch) >= self.BATCH_SIZE or number == max_runs - 1:
                if self.pending_batch:  # safeguard
                    batch = np.stack([b for _, b in self.pending_batch], axis=0)
                    start = time.perf_counter()
                    policy_batch, value_batch = self.neural_net.fast_predict(batch)
                    print("batch_time =", time.perf_counter() - start)
                    for (leaf, _), v in zip(self.pending_batch, value_batch):
                        # old
                        # leaf.expanded_by_nn = True  # mark done
                        # self.backtracking(leaf, float(v))  # propagate
                        # because of batching a group skips exploitation and gets the predicted values directly
                        child_to_move = leaf.state.current_player()
                        parent_to_move = leaf.parent.player_to_move if leaf.parent else child_to_move
                        q = float(v) if parent_to_move == child_to_move else -float(v)
                        self.backtracking(leaf, q)

                    self.pending_batch.clear()

