CONCLUSIVE_RESULT_MULTIPLIER = 1000
TEMPERATURE_CONTROL_FOR_MAX_RANDOMNESS = 1000
# Setting an extremely small value since putting 0 gives division error
TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS = 1e-6
MAX_MOVE_COUNT_WITH_INITIAL_TEMPERATURE_CONTROL = 5
MIN_GAME_SIM_BENCHMARK = 1000
MIN_GAME_SIM_BENCHMARK_MCTS = 1500
MIN_GAME_SIM_VS_HUMAN_BENCHMARK_MCTS = 150
UPPER_BOUND_CONFIDENCE_1_CONSTANT = 2
# value has been chosen based on research papers where value is higher for games like go
# due to incredibly high branching factor but lower for games like chess and so on
PUCT_CONSTANT = 1
# tictactoe moves
MOVE_X = "X"
MOVE_O = "O"
# othello moves
MOVE_B = "B"
MOVE_W = "W"

MCTS_NN = "MCTS+NN"
MCTS = "MCTS"
ALPHA_BETA_PRUNING = "ALPHA_BETA_PRUNING"
GAME_TICTACTOE = "tictactoe"
GAME_OTHELLO = "othello"
NUMBER_OF_NN_CHANNELS = 3
ACTIVATION_TANH = "tanh"
NEURAL_NET_LEARNING_RATE = 1e-3
OTHELLO_BOARD_SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

ASPIRATION_WINDOW_MULTIPLIER = 2
# basically a value after which we just increase the search window size to full
ASPIRATION_WINDOW_FAILURE_UPPER_LIMIT = 50
# XLA Performance Optimization Settings
ENABLE_XLA_COMPILATION = True
XLA_BATCH_SIZE = 32
MCTS_PREDICTION_CACHE_SIZE = 1000