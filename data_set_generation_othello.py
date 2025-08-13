import numpy as np
d = np.load("game_data_board_size8_othello.npz", allow_pickle=True)

for k in d.files:
    arr = d[k]
    print(k, type(arr), getattr(arr, "shape", None), getattr(arr, "dtype", None))
