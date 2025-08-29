# KEPT FOR TESTING AND DEMO ONLY
import math

from constant_strings import ASPIRATION_WINDOW_MULTIPLIER, ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT


# WITH TT table

# method to get a numerical metric for the next possible move with alpha beta pruning
# aim to prune branches where alpha >= beta to diminish search space
# we add a depth_to_result parameter to allow for setting of search depths
def minimax_with_alpha_beta_pruning(ttt_game_instance, isMax, depth_to_result, alpha=-math.inf, beta=math.inf):
    # first check cache for matches
    cached_score = ttt_game_instance.fetch_existing_hash(depth_to_result)
    if cached_score is not None:
        return cached_score
    outcome = ttt_game_instance.detect_win_loss()
    terminal_score = ttt_game_instance.fit_to_ai_metrics(outcome, depth_to_result)
    if terminal_score is not None:
        # store terminal evaluation in TT table
        ttt_game_instance.store_in_transposition_table(terminal_score, depth_to_result)
        return terminal_score

    # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
    best_score = -math.inf if isMax else math.inf
    # get the current turn player's symbol
    current_player = ttt_game_instance.ai_player_code if isMax else 1 - ttt_game_instance.ai_player_code
    current_symbol = ttt_game_instance.get_player_symbol(current_player)
    possible_moves = ttt_game_instance.get_possible_moves()

    for move in possible_moves:
        ttt_game_instance.board[move[0]][move[1]] = current_symbol
        ttt_game_instance.increment_total_move_count()

        # Recursively call minimax for the next player's turn
        score = minimax_with_alpha_beta_pruning(ttt_game_instance, not isMax, depth_to_result + 1, alpha, beta)

        # Undo the move
        ttt_game_instance.undo_last_move(move)

        # Step 6: Update best score
        if isMax:
            best_score = max(best_score, score)
            alpha = max(best_score, alpha)
        else:
            best_score = min(best_score, score)
            beta = min(best_score, beta)

        if alpha >= beta:
            break

    ttt_game_instance.store_in_transposition_table(best_score, depth_to_result)
    return best_score

# method to get a numerical metric for the next possible move
def minimax(ttt_game_instance, isMax):
    # doesn't return anything if game is going on, so only return
    # if it actually has an outcome
    # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
    # the challenge is only when we have to build up recursively when there is no immediate final result
    outcome = ttt_game_instance.detect_win_loss()
    # this second layer explanation can be understood from documentation of the method
    ai_adjusted_outcome = ttt_game_instance.generate_win_loss_metrics_wrt_AI(outcome)
    if ai_adjusted_outcome is not None:
        return ai_adjusted_outcome
    # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
    best_score = -math.inf if isMax else math.inf
    # get the current turn player's symbol
    current_player = ttt_game_instance.ai_player_code if isMax else 1 - ttt_game_instance.ai_player_code
    current_symbol = ttt_game_instance.get_player_symbol(current_player)
    possible_moves = ttt_game_instance.get_possible_moves()

    for move in possible_moves:
        ttt_game_instance.board[move[0]][move[1]] = current_symbol
        ttt_game_instance.increment_total_move_count()

        # Recursively call minimax for the next player's turn
        score = minimax(ttt_game_instance,not isMax)

        # Undo the move
        ttt_game_instance.undo_last_move(move)

        # Step 6: Update best score
        if isMax:
            best_score = max(best_score, score)
        else:
            best_score = min(best_score, score)

    return best_score


# heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
# method to get a numerical metric for the next possible move with alpha beta pruning
# aim to prune branches where alpha >= beta to diminish search space
# we add a depth_to_result parameter to allow for setting of search depths
def heuristic_minimax_with_alpha_beta_pruning(ttt_game_instance, isMax, max_ai_search_depth, depth_to_result, alpha=-math.inf,
                                              beta=math.inf):
    # doesn't return anything if game is going on, so only return
    # if it actually has an outcome
    # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
    # the challenge is only when we have to build up recursively when there is no immediate final result
    outcome = ttt_game_instance.detect_win_loss()
    # this second layer explanation can be understood from documentation of the method
    ai_adjusted_outcome = ttt_game_instance.generate_win_loss_metrics_wrt_AI(outcome)
    if ai_adjusted_outcome is not None:
        # alpha beta pruning/minimax doesn't differentiate between quicker and longer wins,
        # if we can reduce the heuristic value according to how many moves before win
        if ai_adjusted_outcome > 0:
            return ai_adjusted_outcome - depth_to_result
        # basically, if we are losing, we prolong the result
        if ai_adjusted_outcome < 0:
            return ai_adjusted_outcome + depth_to_result
        return ai_adjusted_outcome

    # reached the max depth we set so we can just return the heuristic evaluation of the board
    if max_ai_search_depth == 0:
        return ttt_game_instance.heuristically_evaluate_board()

    # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
    best_score = -math.inf if isMax else math.inf
    # get the current turn player's symbol
    current_player = ttt_game_instance.ai_player_code if isMax else 1 - ttt_game_instance.ai_player_code
    current_symbol = ttt_game_instance.get_player_symbol(current_player)
    possible_moves = ttt_game_instance.get_possible_moves()

    for move in possible_moves:
        ttt_game_instance.board[move[0]][move[1]] = current_symbol
        ttt_game_instance.increment_total_move_count()

        # Recursively call minimax for the next player's turn
        score = heuristic_minimax_with_alpha_beta_pruning(ttt_game_instance, not isMax, max_ai_search_depth - 1, depth_to_result + 1,
                                                                            alpha, beta)

        # Undo the move
        ttt_game_instance.undo_last_move(move)

        # Step 6: Update best score
        if isMax:
            best_score = max(best_score, score)
            alpha = max(best_score, alpha)
        else:
            best_score = min(best_score, score)
            beta = min(best_score, beta)

        if alpha >= beta:
            break

    return best_score


# TT table added with aspiration window

# heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
# method to get a numerical metric for the next possible move with alpha beta pruning
# aim to prune branches where alpha >= beta to diminish search space
# we add a depth_to_result parameter to allow for setting of search depths
def heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(ttt_game_instance, isMax, max_ply, depth_to_result,
                                                                       alpha=-math.inf, beta=math.inf):
    # first check cache for matches
    cached_score = ttt_game_instance.fetch_existing_hash(depth_to_result)
    if cached_score is not None:
        return cached_score
    outcome = ttt_game_instance.detect_win_loss()
    terminal_score = ttt_game_instance.fit_to_ai_metrics(outcome, depth_to_result)
    if terminal_score is not None:
        # store terminal evaluation in TT
        ttt_game_instance.store_in_transposition_table(terminal_score, depth_to_result)
        return terminal_score

    if max_ply == 0:
        # horizon reached – evaluate statically
        static_score = ttt_game_instance.heuristically_evaluate_board()
        # store horizon evaluation in TT
        ttt_game_instance.store_in_transposition_table(static_score, depth_to_result)
        return static_score
    # we reach here means no immediate win
    prev_score = 0
    initial_window = 1
    best_score = -math.inf if isMax else math.inf
    ordered_moves = []
    for depth in range(1, max_ply + 1):
        # removing this line greatly increases speed but reduces search width
        # alpha, beta = -math.inf, math.inf
        window = initial_window
        alpha = prev_score - window
        beta = prev_score + window

        # we need to keep repeating alpha beta for the same depth again and again until the score
        # falls in the acceptable range
        # simply updating values for alpha beta in one pass and hoping it falls within the range from the next
        # depth is a luck thing so inherently unreliable
        while True:
            # NEW: remember the actual window we wanted to test against
            window_low = alpha
            window_high = beta
            current_move_scores = []
            curr_best_score = -math.inf if isMax else math.inf
            current_player = ttt_game_instance.ai_player_code if isMax else 1 - ttt_game_instance.ai_player_code
            current_symbol = ttt_game_instance.get_player_symbol(current_player)
            possible_moves = ttt_game_instance.get_possible_moves() if len(ordered_moves) == 0 else [move for move, score in
                                                                                                     ordered_moves]
            for move in possible_moves:
                ttt_game_instance.board[move[0]][move[1]] = current_symbol
                ttt_game_instance.increment_total_move_count()
                # Recursively call minimax for the next player's turn
                score = heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(ttt_game_instance, not isMax, depth - 1,
                                                                                                             depth_to_result + 1,
                                                                                                             alpha, beta)
                # Undo the move
                ttt_game_instance.undo_last_move(move)
                current_move_scores.append((move, score))
                # Step 6: Update best score
                if isMax:
                    curr_best_score = max(curr_best_score, score)
                    alpha = max(curr_best_score, alpha)
                else:
                    curr_best_score = min(curr_best_score, score)
                    beta = min(curr_best_score, beta)

                if alpha >= beta:
                    break
            ordered_moves = sorted(current_move_scores, key=lambda x: x[1], reverse=isMax)
            best_score = curr_best_score
            # Aspiration Window Check
            # this means values are in range, can go on to the next depth by breaking the while loop
            # NOTE DO NOT USE alpha beta directly instead of window_low and window_hi
            # reason is that alpha and beta are converging possibly and by the time you reach here
            # they have come very very close, Hence they become very different from the inital window size u wanted to compare to
            # so use window_lo and window_hi to store initial values of alpha and beta
            if window_low < curr_best_score < window_high:
                # Success: score fits inside window
                prev_score = curr_best_score
                break
            # Fail-High or Fail-Low → widen the window
            window *= ASPIRATION_WINDOW_MULTIPLIER
            # Have expanded the window quite a bit, perhaps most efficient
            # to search with the full width
            if window > ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT:
                alpha, beta = -math.inf, math.inf
            else:
                alpha = prev_score - window
                beta = prev_score + window

    # TT: store internal node evaluation after all depths
    ttt_game_instance.store_in_transposition_table(best_score, depth_to_result)
    return best_score
