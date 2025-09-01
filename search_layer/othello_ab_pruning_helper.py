from constant_strings import INF, ASPIRATION_WINDOW_MULTIPLIER, ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT


# method to get a numerical metric for the next possible move
def minimax(othello_game_instance, isMax):
    # doesn't return anything if game is going on, so only return
    # if it actually has an outcome
    # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
    # the challenge is only when we have to build up recursively when there is no immediate final result
    outcome = othello_game_instance.detect_win_loss()
    # this second layer explanation can be understood from documentation of the method
    ai_adjusted_outcome = othello_game_instance.generate_win_loss_metrics_wrt_AI(outcome)
    if ai_adjusted_outcome is not None:
        return ai_adjusted_outcome
    # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
    best_score = -INF if isMax else INF
    # get the current turn player's symbol
    current_player = othello_game_instance.ai_player_code if isMax else 1 - othello_game_instance.ai_player_code
    opponent_player = 1 - current_player
    current_symbol = othello_game_instance.get_player_symbol(current_player)
    opponent_symbol = othello_game_instance.get_player_symbol(opponent_player)
    possible_moves = othello_game_instance.get_possible_moves(current_player)

    for move in possible_moves:
        clone_game_instance = othello_game_instance.clone_instance()
        clone_game_instance.board[move[0]][move[1]] = current_symbol
        clone_game_instance.increment_total_move_count()
        # we need the flip in Othello
        clone_game_instance.implement_flips(move[0], move[1], current_symbol, opponent_symbol)

        # Recursively call minimax for the next player's turn
        score = minimax(clone_game_instance, not isMax)
        # Update best score
        if isMax:
            best_score = max(best_score, score)
        else:
            best_score = min(best_score, score)

    return best_score


def minimax_with_alpha_beta_pruning(
        othello_game_instance, isMax: bool, depth_to_result: int,
        alpha: float = -INF, beta: float = INF):
    # Transposition-table probe
    cached_score = othello_game_instance.fetch_existing_hash(depth_to_result)
    if cached_score is not None:
        return cached_score

    #  Terminal check
    outcome = othello_game_instance.detect_win_loss()
    terminal_score = othello_game_instance.fit_to_ai_metrics(outcome, depth_to_result)
    if terminal_score is not None:
        othello_game_instance.store_in_transposition_table(terminal_score, depth_to_result)
        return terminal_score

    # Initialise best score
    best_score = -INF if isMax else INF

    # Identify the side to move and generate its legal moves
    current_player = othello_game_instance.ai_player_code if isMax else 1 - othello_game_instance.ai_player_code
    current_symbol = othello_game_instance.get_player_symbol(current_player)
    opponent_symbol = othello_game_instance.get_player_symbol(1 - current_player)
    possible_moves = othello_game_instance.get_possible_moves(current_player)



    # added for TESTING
    # If no moves: either terminal (already handled above) or forced PASS
    if not possible_moves:
        # if opponent has any move, PASS (flip side to move, same board)
        if othello_game_instance.get_possible_moves(1 - current_player):
            # depth_to_result increases (one ply consumed), but no board change
            score = minimax_with_alpha_beta_pruning(
                othello_game_instance, not isMax, depth_to_result + 1, alpha, beta
            )
            othello_game_instance.store_in_transposition_table(score, depth_to_result)
            return score
        # else both sides have no moves ⇒ terminal would have returned above




    # Loop through children
    for move in possible_moves:
        cloned_child = othello_game_instance.clone_instance()
        cloned_child.board[move[0]][move[1]] = current_symbol
        cloned_child.implement_flips(move[0], move[1],
                                     current_symbol, opponent_symbol)
        cloned_child.total_moves += 1
        cloned_child.last_moved = current_player

        score = minimax_with_alpha_beta_pruning(cloned_child,
            not isMax, depth_to_result + 1, alpha, beta)

        if isMax:
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
        else:
            best_score = min(best_score, score)
            beta = min(beta, best_score)

        if alpha >= beta:  # α-β cut-off
            break

    # 6) Cache and return
    othello_game_instance.store_in_transposition_table(best_score, depth_to_result)
    return best_score

    # To be used to prevent our alpha beta minimax from going till the end
    # and get slowed down instead we can try and use a heuristic function to look for what seems like a better position
    # only needed when game size is greater than 3 otherwise exhaustive search does the trick


# heuristic aid added to evaluate scores for positions and prevent searching till terminal positions
# method to get a numerical metric for the next possible move with alpha beta pruning
# aim to prune branches where alpha >= beta to diminish search space
# we add a depth_to_result parameter to allow for setting of search depths
def heuristic_minimax_with_alpha_beta_pruning(othello_game_instance, isMax, max_ai_search_depth, depth_to_result, alpha=-INF, beta=INF):
    # doesn't return anything if game is going on, so only return
    # if it actually has an outcome
    # Basically, if the last move leads to a draw, loss or win, the result value(1,-1 or 0) itself is the value
    # the challenge is only when we have to build up recursively when there is no immediate final result
    outcome = othello_game_instance.detect_win_loss()
    # this second layer explanation can be understood from documentation of the method
    ai_adjusted_outcome = othello_game_instance.generate_win_loss_metrics_wrt_AI(outcome)
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
        return othello_game_instance.heuristically_evaluate_board()

    # min player is trying to minimize this score for max and max player is trying to maximize this score for themselves
    best_score = -INF if isMax else INF
    # get the current turn player's symbol
    current_player = othello_game_instance.ai_player_code if isMax else 1 - othello_game_instance.ai_player_code
    opponent_player = 1 - current_player
    current_symbol = othello_game_instance.get_player_symbol(current_player)
    opponent_symbol = othello_game_instance.get_player_symbol(opponent_player)
    possible_moves = othello_game_instance.get_possible_moves(current_player)
    # If no moves: either terminal (already handled above) or forced PASS
    if not possible_moves:
        if othello_game_instance.get_possible_moves(opponent_player):
            return heuristic_minimax_with_alpha_beta_pruning(
                othello_game_instance, not isMax,
                max_ai_search_depth,               # do NOT reduce search horizon on pass
                depth_to_result + 1, alpha, beta
            )
        # else both sides stuck → terminal was handled above
    for move in possible_moves:
        clone_game_instance = othello_game_instance.clone_instance()
        clone_game_instance.board[move[0]][move[1]] = current_symbol
        clone_game_instance.increment_total_move_count()
        # we need the flip in Othello
        clone_game_instance.implement_flips(move[0], move[1], current_symbol,
                                            opponent_symbol)
        # Recursively call minimax for the next player's turn
        score = heuristic_minimax_with_alpha_beta_pruning(clone_game_instance,not isMax, max_ai_search_depth - 1,
                                                                              depth_to_result + 1, alpha, beta)
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
def heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
        othello_game_instance, isMax, max_ply, depth_to_result,
        alpha=-INF, beta=INF):
    cached_score = othello_game_instance.fetch_existing_hash(depth_to_result)
    if cached_score is not None:
        return cached_score

    outcome = othello_game_instance.detect_win_loss()
    terminal_score = othello_game_instance.fit_to_ai_metrics(outcome, depth_to_result)
    if terminal_score is not None:
        othello_game_instance.store_in_transposition_table(terminal_score, depth_to_result)
        return terminal_score

    if max_ply == 0:
        static_score = othello_game_instance.heuristically_evaluate_board()
        othello_game_instance.store_in_transposition_table(static_score, depth_to_result)
        return static_score

    # Iterative deepening with aspiration-window
    prev_score = 0
    initial_window = 1
    best_score = -INF if isMax else INF
    ordered_moves = []

    for depth in range(1, max_ply + 1):
        window = initial_window
        alpha = prev_score - window
        beta = prev_score + window

        while True:
            window_low, window_high = alpha, beta
            current_move_scores = []
            current_best_score = -INF if isMax else INF

            player_code = othello_game_instance.ai_player_code if isMax else 1 - othello_game_instance.ai_player_code
            player_symbol = othello_game_instance.get_player_symbol(player_code)
            opp_symbol = othello_game_instance.get_player_symbol(1 - player_code)

            move_list = (othello_game_instance.get_possible_moves(player_code) if not ordered_moves
                         else [move for move, score in ordered_moves])

            # PASS handling
            if not move_list:
                if othello_game_instance.get_possible_moves(1 - player_code):
                    # do NOT shrink depth here; consume only ply count
                    return heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(
                        othello_game_instance, not isMax, max_ply, depth_to_result + 1, alpha, beta
                    )
                # else terminal already returned above

            for move in move_list:
                cloned_child = othello_game_instance.clone_instance()
                cloned_child.board[move[0]][move[1]] = player_symbol
                cloned_child.implement_flips(move[0], move[1],
                                             player_symbol, opp_symbol)
                cloned_child.total_moves += 1
                cloned_child.last_moved = player_code

                score = heuristic_minimax_with_alpha_beta_pruning_with_iterative_deepening(cloned_child,
                    not isMax, depth - 1, depth_to_result + 1,
                    alpha, beta)

                current_move_scores.append((move, score))

                if isMax:
                    current_best_score = max(current_best_score, score)
                    alpha = max(alpha, current_best_score)
                else:
                    current_best_score = min(current_best_score, score)
                    beta = min(beta, current_best_score)

                if alpha >= beta:  # αβ cut-off
                    break

            # order moves for next iteration
            ordered_moves = sorted(current_move_scores,
                                   key=lambda x: x[1],
                                   reverse=isMax)
            best_score = current_best_score

            # Aspiration-window success?
            if window_low < current_best_score < window_high:
                prev_score = current_best_score
                break

            # Otherwise widen the window and repeat
            window *= ASPIRATION_WINDOW_MULTIPLIER
            if window > ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT:
                alpha, beta = -INF, INF
            else:
                alpha = prev_score - window
                beta = prev_score + window

    othello_game_instance.store_in_transposition_table(best_score, depth_to_result)
    return best_score