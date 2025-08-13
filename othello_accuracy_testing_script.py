# import numpy as np
# import tensorflow as tf
# from keras.src.saving import load_model
# from Neural_Net_Utils import flattened_board_to_tensor
#
# # ── paths & batch size ────────────────────────────────────────────────────
# MODEL_PATH = "weights_othello_8/othello-8.keras"
# NPZ_PATH   = "game_data_board_size8_othello.npz"
# BATCH      = 192               # adjust if your GPU is small
#
# # ── load model & dataset (memory-mapped) ──────────────────────────────────
# model  = load_model(MODEL_PATH)
# data   = np.load(NPZ_PATH, allow_pickle=True, mmap_mode="r")
# states = data["states"]
# pols   = data["policies"]
# vals   = data["values"]
#
# # detect channel layout expected by the model
# channels_last = (model.input_shape[-1] == 3)
#
# # ── streaming metrics ────────────────────────────────────────────────────
# policy_correct = 0
# value_metric   = tf.keras.metrics.Accuracy()
#
# for start in range(0, len(states), BATCH):
#     end           = start + BATCH
#     batch_states  = states[start:end]
#     batch_pols    = pols[start:end]
#     batch_vals    = vals[start:end]
#
#     # build input tensor batch on the fly
#     if channels_last:
#         X = np.empty((len(batch_states), 8, 8, 3), dtype=np.float32)
#     else:                             # channels_first
#         X = np.empty((len(batch_states), 3, 8, 8), dtype=np.float32)
#
#     for i, s in enumerate(batch_states):
#         board = flattened_board_to_tensor(s, "othello")  # (8,8,3)
#         if channels_last:
#             X[i] = board
#         else:
#             X[i] = np.transpose(board, (2, 0, 1))        # ➜ (3,8,8)
#
#     # forward pass (no huge logits array kept in RAM)
#     pol_pred, val_pred = model.predict_on_batch(X)
#
#     # policy top-1
#     true_moves = np.argmax(batch_pols, axis=1)
#     pred_moves = np.argmax(pol_pred,  axis=1)
#     policy_correct += np.sum(true_moves == pred_moves)
#
#     # value sign accuracy
#     true_signs = np.sign(batch_vals).astype(np.int8)
#     pred_signs = np.sign(val_pred).astype(np.int8)
#     value_metric.update_state(true_signs, pred_signs)
#
# # ── final report ──────────────────────────────────────────────────────────
# policy_top1      = 100 * policy_correct / len(states)
# value_sign_acc   = 100 * value_metric.result().numpy()
#
# print(f"📊 Policy top-1 accuracy : {policy_top1:6.2f} %")
# print(f"📊 Value sign accuracy   : {value_sign_acc:6.2f} %")



import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, MeanAbsoluteError

# --- Config ---
MODEL_PATH = "othello-8.keras"
DATA_PATH  = "game_data_board_size8_othello.npz"
BATCH_SIZE = 1024

# --- Load model ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loaded:", MODEL_PATH)

# --- Load data ---
d = np.load(DATA_PATH, allow_pickle=True)
states  = d["states"]                 # (N,) object of 64-char strings
policies = d["policies"].astype(np.float32)  # (N,64)
values   = d["values"].astype(np.float32)    # (N,)

# --- Helpers (match your Neural_Net_Utils.flattened_board_to_tensor) ---
def board_str_to_planes_3(s: str):
    # plane 0: B, plane 1: W, plane 2: turn (1 if Black-to-move, else 0)
    b = np.zeros((8,8), np.float32)
    w = np.zeros((8,8), np.float32)
    for i, ch in enumerate(s):
        r, c = divmod(i, 8)
        if ch == 'B': b[r,c] = 1.0
        elif ch == 'W': w[r,c] = 1.0
    # side to move: in your format, Black moves when counts are equal
    turn_plane = np.ones_like(b) if b.sum() == w.sum() else np.zeros_like(b)
    return np.stack([b, w, turn_plane], axis=-1)  # (8,8,3)

def gen():
    # yield batches lazily to keep RAM reasonable
    for s, p, v in zip(states, policies, values):
        x = board_str_to_planes_3(s)
        yield x, {"policy_logits": p, "value": np.array([v], dtype=np.float32)}  # value shape (1,)

# --- tf.data pipeline ---
output_signature = (
    tf.TensorSpec(shape=(8,8,3), dtype=tf.float32),
    {
        "policy_logits": tf.TensorSpec(shape=(64,), dtype=tf.float32),
        "value":         tf.TensorSpec(shape=(1,),  dtype=tf.float32),
    },
)
ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)\
                    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Metrics ---
top1  = CategoricalAccuracy(name="top1")               # works with logits
top3  = TopKCategoricalAccuracy(k=3, name="top3")
top5  = TopKCategoricalAccuracy(k=5, name="top5")
v_mae = MeanAbsoluteError(name="value_mae")

# --- Eval loop ---
for bx, by in ds:
    pred_policy, pred_value = model(bx, training=False)   # logits, tanh
    top1.update_state(by["policy_logits"], pred_policy)
    top3.update_state(by["policy_logits"], pred_policy)
    top5.update_state(by["policy_logits"], pred_policy)
    v_mae.update_state(by["value"], pred_value)

print(f"Policy Top-1: {top1.result().numpy():.4f}")
print(f"Policy Top-3: {top3.result().numpy():.4f}")
print(f"Policy Top-5: {top5.result().numpy():.4f}")
print(f"Value MAE   : {v_mae.result().numpy():.4f}")
