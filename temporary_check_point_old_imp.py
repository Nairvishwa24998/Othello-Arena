# def iterative_deepening_support_ab_pruning(self, isMax, max_plies):
#     best_move = None
#     best_score = -math.inf if isMax else math.inf
#     ordered_moves = []
#     # we loop until the max depth, trying an ab pruning with heuristics upto the decided depth deepening approach at each turn
#     for depth in range(1, max_plies+1):
#         current_move_scores = []
#         curr_best_move = None
#         curr_best_score = -math.inf if isMax else math.inf
#         current_player = self.ai_player_code if isMax else 1 - self.ai_player_code
#         current_symbol = self.get_player_symbol(current_player)
#         # if first iteration no need to sort. You only go upto 1 depth and re-ordering is not relevant
#         # else get the move component out of ordered moves
#         possible_moves = self.get_possible_moves() if len(ordered_moves) == 0 else [move for move,score in ordered_moves]
#         for move in possible_moves:
#             self.board[move[0]][move[1]] = current_symbol
#             self.increment_total_move_count()
#             # calling the alg from the next
#             score = self.heuristic_minimax_with_alpha_beta_pruning(isMax= not isMax, max_ai_search_depth=depth-1, depth_to_result=1, alpha=-math.inf, beta=math.inf)
#             self.undo_last_move(move)
#             current_move_scores.append((move,score))
#             # this is just to keep track. Could easily have skipped this assignment and
#             # used the first element from ordered moves list
#             if isMax and score > curr_best_score:
#                 curr_best_score = score
#                 curr_best_move = move
#             elif not isMax and score < curr_best_score:
#                 curr_best_score = score
#                 curr_best_move = move
#         # if for max player would reverse and give highest first
#         # for min player won't reverse since isMax would be false
#         # and hence pick the smallest one
#         ordered_moves = sorted(current_move_scores, key = lambda x:x[1], reverse = isMax)
#         best_move = curr_best_move
#         best_score = curr_best_score
#         # basically found a win so stop iterating and pick that
#         if abs(best_score) == CONCLUSIVE_RESULT_MULTIPLIER:
#             break
#     return best_score