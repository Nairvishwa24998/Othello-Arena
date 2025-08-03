import random
import numpy as np

from Neural_Net_Utils import flattened_board_to_tensor, prepare_neural_net_instance
from Node import Node
from common_utils import board_hash, link_game_position_hash_to_pv
from constant_strings import MIN_GAME_SIM_BENCHMARK_MCTS, MCTS, MCTS_NN, GAME_TICTACTOE
from testing import board


class Mcts:
    def __init__(self, root, game_instance):
        self.root = Node(state=game_instance)
        # just a hashed version of the current board state when MCTS is called
        self.hashed_root = board_hash(game_instance.get_current_board_state(), game_instance.current_player())
        self.neural_net = prepare_neural_net_instance(game=GAME_TICTACTOE, size = game_instance.get_board_size(), ai_type = game_instance.get_AI_type())
        # # # Cache for batch predictions to avoid repeated tensor conversions
        # self._prediction_cache = {}
        self.mcts_transposition_table = game_instance.mcts_transposition_table

        # link_game_position_hash_to_pv(game_instance.mcts_transposition_table, self.hashed_root, )



    def get_root(self):
        return self.root

    def get_neural_net(self):
        return self.neural_net

    # new version
    def selection(self):
        current_node = self.root
        ai_type = current_node.state.ai_type
        while True:
            # If it is not None. Game is over, can't do selection. Perhaps should do backtracking
            if current_node.state.detect_win_loss() is not None:
                break
            # some possible moves from current position yet to be considered or given node is not fully expanded
            # so selection has to break for now for this node
            if len(current_node.children) < len(current_node.state.get_possible_moves()):
                break
            if not current_node.children:
                break
            # if none of the above case, we can do our UCB1 and choose the best one
            # we can use list. But dictionaries would allow us to use calculated ucb1 values
            # compute once per child
            # there are only two flows mcts or mcts_nn which can reach here
            resultant_values = {child: child.ucb1() for child in current_node.children.values()} if ai_type == MCTS \
                else {child: child.puct() for child in current_node.children.values()}

            best_confidence_value = max(resultant_values.values())

            best_children = [child for child, value in resultant_values.items()
                             if abs(value - best_confidence_value) < 1e-12]

            current_node = random.choice(best_children)

            # random tie-break so tree doesn’t freeze
            # basically allow almost random choices between top choices who are very very close to each other
        return current_node

    # with tt table
    def expansion(self, current_node):
        # we need to ensure each child being expanded into has a separate clone of the parent
        # current_node is a clone of the parent. but we need separate clones of the parent
        # for each child
        parent_board = current_node.get_state().get_current_board_state()
        ai_type = current_node.state.ai_type
        cloned_instance = current_node.state.clone_instance()
        children = current_node.get_children()
        possible_moves = current_node.state.get_possible_moves()
        player_turn = cloned_instance.determine_player_turn()
        contender_moves = []
        for move in possible_moves:
            if move not in children:
                contender_moves.append(move)
        # current node is fully expanded-all its possible moves have been added as Nodes
        # so can't expand further
        if len(contender_moves) == 0:
            return None
        move = random.choice(contender_moves)
        cloned_instance.board[move[0]][move[1]] = cloned_instance.get_player_symbol(player_turn)
        cloned_instance.increment_total_move_count()
        child_node = Node(cloned_instance, parent=current_node, move=move)
        if ai_type == MCTS_NN:
            hashed_board_key = board_hash(parent_board, player_turn)
            tt_value = self.mcts_transposition_table.get(hashed_board_key)
            # if tt_value is None:
            pre_move_flattened_state_2d = "".join(str(cell) for row in parent_board for cell in row)
            inp = flattened_board_to_tensor(pre_move_flattened_state_2d, game_name=GAME_TICTACTOE)[None, ...]
            neural_net = self.get_neural_net()
            policy_prediction, value_prediction = neural_net.model.predict(inp, verbose=0)
            # predicting and storing in cache for future
            tt_value = link_game_position_hash_to_pv(self.mcts_transposition_table, hashed_board_key, policy_prediction[0], value_prediction)

            flat = move[0] * cloned_instance.size + move[1]
            child_node.policy_prior = float(tt_value["policy"][flat])
        else:  # pure MCTS or no model yet
            # pretty useless as of now. Purely to avoid issues. Most cases should not even touch this line
            child_node.policy_prior = 1.0 / len(possible_moves)
        children[move] = child_node
        return child_node



    # # without tt table
    # def expansion(self, current_node):
    #     # we need to ensure each child being expanded into has a separate clone of the parent
    #     # current_node is a clone of the parent. but we need separate clones of the parent
    #     # for each child
    #     parent_board = current_node.get_state().get_current_board_state()
    #     ai_type = current_node.state.ai_type
    #     cloned_instance = current_node.state.clone_instance()
    #     children = current_node.get_children()
    #     possible_moves = current_node.state.get_possible_moves()
    #     player_turn = cloned_instance.determine_player_turn()
    #     contender_moves = []
    #     for move in possible_moves:
    #         if move not in children:
    #             contender_moves.append(move)
    #     # current node is fully expanded-all its possible moves have been added as Nodes
    #     # so can't expand further
    #     if len(contender_moves) == 0:
    #         return None
    #     move = random.choice(contender_moves)
    #     cloned_instance.board[move[0]][move[1]] = cloned_instance.get_player_symbol(player_turn)
    #     cloned_instance.increment_total_move_count()
    #     child_node = Node(cloned_instance, parent=current_node, move=move)
    #     if ai_type == MCTS_NN:
    #         # method after this takes a flattened board only
    #         # commented out for testing since too time consuming
    #         # pre_move_flattened_state_2d = "".join(str(cell) for row in parent_board for cell in row)
    #         # inp = flattened_board_to_tensor(pre_move_flattened_state_2d, game_name=GAME_TICTACTOE)[None, ...]  # helper you already have
    #         # neural_net = self.get_neural_net()
    #         # policy_prediction, value_prediction = neural_net.model.predict(inp, verbose=0)
    #         # flat = move[0] * cloned_instance.size + move[1]
    #         # child_node.policy_prior = float(policy_prediction[0][flat])
    #         # cache the policy+value prediction on the current node to avoid recomputation
    #         if not hasattr(current_node, "_cached_policy_value"):
    #             pre_move_flattened_state_2d = "".join(str(cell) for row in parent_board for cell in row)
    #             inp = flattened_board_to_tensor(pre_move_flattened_state_2d, game_name=GAME_TICTACTOE)[None, ...]
    #             neural_net = self.get_neural_net()
    #             policy_prediction, value_prediction = neural_net.model.predict(inp, verbose=0)
    #             current_node._cached_policy_value = (policy_prediction[0], value_prediction)
    #
    #         # reuse cached prediction
    #         policy_prediction, _ = current_node._cached_policy_value
    #         flat = move[0] * cloned_instance.size + move[1]
    #         child_node.policy_prior = float(policy_prediction[flat])
    #     else:  # pure MCTS or no model yet
    #         # pretty useless as of now. Purely to avoid issues. Most cases should not even touch this line
    #         child_node.policy_prior = 1.0 / len(possible_moves)
    #     children[move] = child_node
    #     return child_node

    # new one
    def exploitation(self, current_node):
        input_game_instance = current_node.state
        ai_type = input_game_instance.get_AI_type()
        if ai_type == MCTS_NN and self.neural_net:  # self.neural_net set in __init__
            board_str = "".join(str(c) for row in current_node.state.board for c in row)
            v = self.neural_net.model.predict(
                flattened_board_to_tensor(board_str, GAME_TICTACTOE)[None, ...],
                verbose=0,
            )[1][0]  # value scalar in [-1,1]
            return -float(v)  # flip perspective once (backtracking will flip again)
        simulation_instance = input_game_instance.clone_instance()
        # make it AI-vs-AI
        simulation_instance.set_to_simulation_mode()
        # whose turn is it?
        simulation_instance.ai_player_code = simulation_instance.current_player()
        if ai_type is None:  # default policy
            simulation_instance.set_AI_type(MCTS)
        # only two possible outcomes here MCTS or MCTS + NN. Alpha beta pruning doesn't even come here
        # this would be the no mcts case
        outcome = simulation_instance.rollout_pseudo_random()
        refined_outcome = -outcome
        return refined_outcome

    # new_version
    def backtracking(self, current_node, refined_outcome):
        ai_type = current_node.state.get_AI_type()
        while current_node is not None:
            current_node.visits += 1
            # for pure mcts case
            if ai_type == MCTS:
                current_node.wins += refined_outcome
            else:  # for neural net case
                current_node.backtracked_value += refined_outcome
            current_node = current_node.parent
            # what is win for a current node would be a loss for the parent node since turns flip
            refined_outcome = -refined_outcome

    # MIN_GAME_SIM_BENCHMARK_MCTS used for simulations runs
    # MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS used for vs human play
    def commence_mcts_for_selfplay(self, max_runs):
        for number in range(max_runs):
            parent = self.selection()
            child = self.expansion(parent) or parent
            value = self.exploitation(child)
            self.backtracking(child, value)
