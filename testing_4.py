import numpy as np

d = np.load("game_data_board_size8_othello.npz", allow_pickle=True)
states, values = d["states"], d["values"].astype(np.float32)

def side_to_move(s):
    # adapt if your state is not a string; this works for flat 64-char strings
    if not isinstance(s, str):
        s = "".join(np.array(s).reshape(-1).tolist())
    b = s.count('B'); w = s.count('W')
    return 'B' if b == w else 'W'

def disc_diff(s):
    if not isinstance(s, str):
        s = "".join(np.array(s).reshape(-1).tolist())
    return s.count('B') - s.count('W')

def empties(s):
    if not isinstance(s, str):
        s = "".join(np.array(s).reshape(-1).tolist())
    return s.count('.')

N = 100_000  # sample
stm = np.array([side_to_move(s) for s in states[:N]])
dd  = np.array([disc_diff(s)     for s in states[:N]], dtype=np.float32)
sign_stm = np.where(stm=='B', 1.0, -1.0)

# raw vs STM-adjusted disc advantage
dd_raw   = dd
dd_stm   = dd * sign_stm

def corr(a,b):
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum() / (np.sqrt((a*a).sum())*np.sqrt((b*b).sum()) + 1e-8))

print("corr(value, disc_diff raw)    :", corr(values[:N], dd_raw))
print("corr(value, disc_diff (STM))  :", corr(values[:N], dd_stm))

# Late-game check where the proxy is strongest
late = np.array([empties(s)<=10 for s in states[:N]])
print("LATE corr(raw):", corr(values[:N][late], dd_raw[late]))
print("LATE corr(STM):", corr(values[:N][late], dd_stm[late]))
