import random

from Node import Node
from constant_strings import MIN_GAME_SIMULATION_BENCHMARK_MCTS


class Mcts:
    def __init__(self,root, tictactoe_instance):
        self.root = Node(state=tictactoe_instance)

    def selection(self):
        current_node = self.root
        while True:
            # If it is not None. Game is over, can't do selection. Perhaps should do backtracking
            if current_node.state.detect_win_loss() is not None:
                break
            # some possible moves from current position yet to be considered
            if len(current_node.children) < len(current_node.state.get_possible_moves()):
                break
            if not current_node.children:
                break
            # if none of the above case, we can do our UCB1 and choose the best one
            parent_node = current_node.visits or 1
            best_ucb = max(child.ucb1(parent_node) for child in current_node.children.values())
            # random tie-break so tree doesn’t freeze
            # basically allow almost random choices between top choices who are very very close to each other
            best_children = [c for c in current_node.children.values()
                             if abs(c.ucb1(parent_node) - best_ucb) < 1e-12]
            current_node = random.choice(best_children)
        return current_node


    def expansion(self, current_node):
        cloned_instance = current_node.state
        children = current_node.get_children()
        possible_moves = current_node.state.get_possible_moves()
        player_turn = cloned_instance.determine_player_turn()
        contender_moves = []
        for move in possible_moves:
            if move not in children:
                contender_moves.append(move)
        # current node is fully expanded-all its possible moves have been added as Nodes
        if len(contender_moves) == 0:
            return None
        move = random.choice(contender_moves)
        cloned_instance.board[move[0]][move[1]] = cloned_instance.get_player_symbol(player_turn)
        cloned_instance.increment_total_move_count()
        child_node = Node(cloned_instance, parent=current_node, move=move)
        children[move] = child_node
        return child_node

    # can also be called simulation
    def exploitation(self, current_node):
        input_game_instance = current_node.state
        simulation_instance = input_game_instance.clone_instance()
        outcome = simulation_instance.run_game()
        refined_outcome = simulation_instance.generate_win_loss_metrics_wrt_NN(outcome)
        return refined_outcome

    def backtracking(self, current_node, refined_outcome):
        while current_node is not None:
            current_node.visits += 1
            current_node.wins += refined_outcome
            current_node = current_node.parent
            # what is win for a current node would be a loss for the parent node since turns flip
            refined_outcome = -refined_outcome

    def commence_mcts(self):
        for number in range(MIN_GAME_SIMULATION_BENCHMARK_MCTS):
            parent = self.selection()
            child = self.expansion(parent) or parent
            value = self.exploitation(child)
            self.backtracking(child, value)
































