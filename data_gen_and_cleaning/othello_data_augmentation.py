import tensorflow as tf
from neural_net import Neural_Net
# from constant_strings import GAME_OTHELLO
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.src.optimizers import Adam
from keras.src.losses import CategoricalCrossentropy, MeanSquaredError
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from neural_net_utils import load_npz_dataset, obtain_train_test_validation_data, prepare_input_output
from constant_strings import GAME_OTHELLO

dataset = load_npz_dataset(8, GAME_OTHELLO)               # uses your NPZ w/ to_move
train_data, val_data, _ = obtain_train_test_validation_data(dataset)

train_X, train_Y = prepare_input_output(train_data, GAME_OTHELLO)  # (N,8,8,3), dict: policy/value
val_X,   val_Y   = prepare_input_output(val_data,   GAME_OTHELLO)


AUTOTUNE = tf.data.AUTOTUNE

# basically augment our data randomly along 7 directions
# Rotation 90°
# Rotation 180°
# Rotation 270°
# Horizontal reflection (flip left–right)
# Vertical reflection (flip top–bottom)
# Diagonal reflection (main diagonal)
# Diagonal reflection (anti-diagonal)

def d4_augment(x, y):
    # k ∈ {0..7}: rotate r times, then horizontal flip if k>=4
    k = tf.random.uniform((), 0, 8, dtype=tf.int32)
    r = k % 4
    flip = k >= 4

    x_aug = tf.image.rot90(x, r)
    pol2d = tf.reshape(y["policy_logits"], [8, 8])
    pol2d = tf.image.rot90(pol2d[..., tf.newaxis], r)[..., 0]

    if flip:
        x_aug = tf.image.flip_left_right(x_aug)
        pol2d = tf.image.flip_left_right(pol2d[..., tf.newaxis])[..., 0]

    y_aug = {
        "policy_logits": tf.reshape(pol2d, [64]),
        "value": y["value"],
    }
    return x_aug, y_aug

train_ds = (tf.data.Dataset
            .from_tensor_slices((train_X, train_Y))
            .shuffle(100_000)
            .map(d4_augment, num_parallel_calls=AUTOTUNE)
            .batch(256)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset
          .from_tensor_slices((val_X, val_Y))
          .batch(1024)
          .prefetch(AUTOTUNE))

# init + load your existing weights
nn = Neural_Net(game=GAME_OTHELLO, size=8)
nn.load("/path/to/othello-8.keras")  # <-- set this

# 2) compile (same losses as before; equal weights so val_loss = policy + value)
nn.model.compile(
    optimizer=Adam(1e-3),
    loss={
        "policy_logits": CategoricalCrossentropy(from_logits=True),
        "value": MeanSquaredError(),
    },
    loss_weights={"policy_logits": 1.0, "value": 1.0},
)

ckpt_dir = "checkpoints_othello8_d4"
cbs = [
    ModelCheckpoint(f"{ckpt_dir}/best.keras", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
]

# train on D4-augmented datasets
history = nn.model.fit(
# augmented train dataset
    train_ds,
    validation_data=val_ds,
    epochs=60,
    callbacks=cbs,
)

# 5) save the fine-tuned model
nn.save("othello-8_finetuned_d4.keras")
