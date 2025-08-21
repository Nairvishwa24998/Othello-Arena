import numpy as np
import tensorflow as tf
from keras.src.saving import load_model

from Neural_Net_Utils import flattened_board_to_tensor

# Load the trained model
model = load_model("ttt-4.keras")

# Load the test dataset
data = np.load("../game_data_board_size4_ttt.npz", allow_pickle=True)
test_X = data["states"]
test_Y = {
    "policy_logits": data["policies"],
    "value": data["values"]
}

game_name = "tictactoe"
game_size = 4

# Preprocess X (convert strings to tensors)
test_X_tensor = np.array([
    flattened_board_to_tensor(state_str, game_name)
    for state_str in test_X
], dtype=np.float32)

# Predict on test set
policy_preds, value_preds = model.predict(test_X_tensor, verbose=0)

# --- Policy Accuracy ---
true_policy_moves = np.argmax(test_Y["policy_logits"], axis=1)
predicted_policy_moves = np.argmax(policy_preds, axis=1)
policy_accuracy = np.mean(true_policy_moves == predicted_policy_moves)

# --- Value Sign Accuracy ---
true_value_signs = np.round(test_Y["value"].flatten())
predicted_value_signs = np.round(value_preds.flatten())
value_sign_accuracy = np.mean(true_value_signs == predicted_value_signs)

print(f"Policy Accuracy: {policy_accuracy * 100:.2f}%")
print(f" Value Sign Accuracy: {value_sign_accuracy * 100:.2f}%")
