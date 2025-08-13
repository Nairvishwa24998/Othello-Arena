import numpy as np

d = np.load("game_data_board_size8_othello.npz", allow_pickle=True)
states, values = d["states"], d["values"]

def side_to_move_from_state(s):
    # s could be a flat 64-char string like '.BW...'
    # or an array/list; adapt as needed
    if isinstance(s, str):
        board = s
    else:
        board = "".join(np.array(s).reshape(-1).tolist())
    b = board.count('B')
    w = board.count('W')
    return 'B' if b == w else 'W'

# Count how many times we'd flip
flips = 0
for i in range(10000):  # sample for speed
    stm = side_to_move_from_state(states[i])
    if stm == 'W' and values[i] != 0:
        flips += 1
print("White-to-move non-draw samples in first 10k:", flips)

# If you use the flip, recompute a quick baseline:
import math
y = values[:100000].astype(np.float32)
stm = np.array([side_to_move_from_state(s) for s in states[:100000]])
y_stm = np.where(stm == 'B', y, -y)
mse_zero = np.mean((y - 0)**2)
mse_zero_stm = np.mean((y_stm - 0)**2)
print("Baseline MSE (no flip):", mse_zero, " | with flip:", mse_zero_stm)