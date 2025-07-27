import math

from constant_strings import UPPER_BOUND_CONFIDENCE_1_CONSTANT, PUCT_CONSTANT


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
        # these two would aid us with the Neural Network
        self.backtracked_value = 0
        self.policy_prior = 0.0

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

    def get_accumulated_backtracked_value(self):
        return self.backtracked_value

    def get_policy_prior(self):
        return self.policy_prior

    # we can potentially replace with PUCT1 when we add the Neural Net
    # mathematical formula for reference UCB1 = (w / n) + c * sqrt(log(N) / n)
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
        # applied formula for UCB1
        return (wins/visits) + UPPER_BOUND_CONFIDENCE_1_CONSTANT*math.sqrt(math.log(parent_visits)/visits)

    # PUCT(s,a)=Q(s,a)+U(s,a)
    # U(s,a) =
    # policy upper bound confidence. to be used to get policy priors. replacement for ucb1 in selection process
    def puct(self):
        value = self.get_accumulated_backtracked_value()
        visits = self.get_visits()
        parent = self.get_parent()
        # to tackle the base case where we are touching the first node itself
        parent_visits = parent.get_visits() if parent is not None else 1
        exploitation_bonus = -math.inf
        exploration_bonus = -math.inf
        # exploitation_bonus = nn_predicted_value/visits
        # exploration_bonus = constant*policy_prior*(root of parent visits)/(1+node visits)
        # result = exploration_bonus + exploitation_bonus
        if visits == 0:
            exploitation_bonus = 0
        else:
            exploitation_bonus = value/visits
        exploration_bonus = PUCT_CONSTANT * self.policy_prior * math.sqrt(parent_visits)/(1 + visits)
        return exploration_bonus + exploitation_bonus


