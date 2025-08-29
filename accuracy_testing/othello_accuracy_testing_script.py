from neural_net import Neural_Net
from neural_net_utils import load_npz_dataset, obtain_train_test_validation_data, prepare_input_output
from constant_strings import GAME_OTHELLO
import numpy as np, tensorflow as tf
# from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, MeanAbsoluteError
from keras.src.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, MeanAbsoluteError
# load model
model = tf.keras.models.load_model("othello-8.keras", compile=False)

dataset = load_npz_dataset(8, GAME_OTHELLO)

# only val data is relevant
_, val_data, _ = obtain_train_test_validation_data(dataset)
val_X, val_Y = prepare_input_output(val_data, GAME_OTHELLO)   # uses your flattened_board_to_tensor (with turn_to_move)

#predict & metrics (logits are fine for these metrics)
pol_logits, val_pred = model.predict(val_X, batch_size=1024, verbose=1)

top1 = CategoricalAccuracy()
top3 = TopKCategoricalAccuracy(k=3)
top5 = TopKCategoricalAccuracy(k=5)
mae  = MeanAbsoluteError()

top1.update_state(val_Y["policy_logits"], pol_logits)
top3.update_state(val_Y["policy_logits"], pol_logits)
top5.update_state(val_Y["policy_logits"], pol_logits)
mae.update_state(val_Y["value"],         val_pred)

print(f"Policy Top-1: {top1.result().numpy():.4f}")
print(f"Policy Top-3: {top3.result().numpy():.4f}")
print(f"Policy Top-5: {top5.result().numpy():.4f}")
print(f"Value MAE   : {mae.result().numpy():.4f}")
