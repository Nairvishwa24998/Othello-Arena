import random

from Node import Node
from constant_strings import MIN_GAME_SIM_BENCHMARK_MCTS, MCTS


class Mcts:
    def __init__(self,root, tictactoe_instance):
        self.root = Node(state=tictactoe_instance)

    def get_root(self):
        return self.root

    def selection(self):
        current_node = self.root
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
            ucb_values = {child: child.ucb1()  # compute once per child
                          for child in current_node.children.values()}

            best_ucb = max(ucb_values.values())

            best_children = [child for child, value in ucb_values.items()
                             if abs(value - best_ucb) < 1e-12]
            current_node = random.choice(best_children)

            # random tie-break so tree doesn’t freeze
            # basically allow almost random choices between top choices who are very very close to each other
        return current_node


    def expansion(self, current_node):
        # we need to ensure each child being expanded into has a separate clone of the parent
        # current_node is a clone of the parent. but we need separate clones of the parent
        # for each child
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
        # remember that dictionaries and lists are mutable objects in python
        # so when we save a dict as a variable and modify the variable
        # we modify the original dict as well
        children[move] = child_node
        return child_node

    # can also be called simulation
    def exploitation(self, current_node):
        input_game_instance = current_node.state
        simulation_instance = input_game_instance.clone_instance()
        # make it AI-vs-AI
        simulation_instance.set_to_simulation_mode()
        # whose turn is it?
        simulation_instance.ai_player_code = simulation_instance.current_player()
        if simulation_instance.get_AI_type() is None:  # default policy
            simulation_instance.set_AI_type(MCTS)
        # outcome = simulation_instance.run_game()
        outcome = simulation_instance.rollout_pseudo_random()
        # refined_outcome = simulation_instance.generate_win_loss_metrics_wrt_NN(outcome)
        refined_outcome = -outcome
        return refined_outcome

    def backtracking(self, current_node, refined_outcome):
        while current_node is not None:
            current_node.visits += 1
            current_node.wins += refined_outcome
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
