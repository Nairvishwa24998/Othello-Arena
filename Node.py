import math

from constant_strings import UPPER_BOUND_CONFIDENCE_1_CONSTANT


# purely created for ease of traversal during the MCTS
class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        # could have been a list too but dictionary has better lookup time
        self.children = {}
        self.visits = 0
        self.wins = 0

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_visits(self):
        return self.visits

    def get_wins(self):
        return self.wins

    # to simply return the tictactoe instance
    def get_state(self):
        return self.state

    # we can potentially replace with PUCT1 when we add the Neural Net
    # mathematical formula for reference UCB = (w / n) + c * sqrt(log(N) / n)
    def ucb1(self):
        wins = self.get_wins()
        visits = self.get_visits()
        parent = self.get_parent()
        parent_visits = 0
        # otherwise would lead to dividing by 0 related errors
        if visits == 0:
            return math.inf
        # to prevent edge case where parent visit is 0 and causes error in log
        parent_visits = max(parent.get_visits(), 1) if parent else 1
        return (wins/visits) + UPPER_BOUND_CONFIDENCE_1_CONSTANT*math.sqrt(math.log(parent_visits)/visits)
