import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

from Neural_Net import Neural_Net
from constant_strings import GAME_OTHELLO, GAME_TICTACTOE, MCTS_NN, MCTS

# this allows us to get all the
npz_files = glob.glob("game_data_board_size*.npz")


# can be used to obtain the symbols for the game
def get_game_symbols(game_name):
    if game_name == GAME_TICTACTOE:
        return ["X", "O"]
    if game_name == GAME_OTHELLO:
        return ["B", "W"]

# can be used to get file names which in turn can be used to get the
def get_game_abbr(game_name):
    if game_name == GAME_TICTACTOE:
        return "ttt"
    if game_name == GAME_OTHELLO:
        return "othello"

# to help us convert input flattened string to a tensor 3D object, with all the planes needed

# we can use the same architecture as used in AlphaGo in terms of planes indicating a player's pieces
# but we don't need historic planes since repetitions are not an issue in tictactoe or othello
# so one plane for whose turn, another two positional planes is accurate
# we are giving three matrices here - first matrix with ones wherever X/Black's pieces are found
# second matrix with of 1s where O/ White pieces
def flattened_board_to_tensor(state_str: str, game_name):
    symbol_1, symbol_2 = get_game_symbols(game_name)
    # we get the square root of the length of the flattened input string
    # for example length of string would be 9 for a 3*3 board
    board_size = int(len(state_str) ** 0.5)
    # reshaping into a matrix format
    board = np.array(list(state_str)).reshape((board_size, board_size))
    # basically matrix indicates 1 where symbol is present 0 where not
    # older models did use both in one single matrix, but CNNs work much better with multiple
    first_symbol_plane = (board == symbol_1).astype(np.float32)
    second_symbol_plane = (board == symbol_2).astype(np.float32)

    # matrix plane for the turn
    first_player_turn = (first_symbol_plane.sum() == second_symbol_plane.sum())
    turn_plane = np.ones_like(first_symbol_plane) if first_player_turn else np.zeros_like(first_symbol_plane)

    return np.stack([first_symbol_plane, second_symbol_plane, turn_plane], axis=-1)


# can be used to load corresponding data sets for both othello and tictactoe
def load_npz_dataset(board_size: int, game_name):
    game_abbr = get_game_abbr(game_name)
    file_name = f"game_data_board_size{board_size}_{game_abbr}.npz"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"{file_name} not found")
    pack = np.load(file_name, allow_pickle=True)
    # Note that this will be a array of dictionaries
    states = pack["states"]
    policies = pack["policies"]
    values = pack["values"]
    board, policy, value = [], [], []
    for s,p,v in zip(states, policies, values):
        board.append(flattened_board_to_tensor(s, game_name))
        policy.append(p)
        value.append([v])  # Wrap value as list to match expected shape

    return (np.asarray(board, dtype=np.float32),
            np.asarray(policy, dtype=np.float32),
            np.asarray(value, dtype=np.float32))


# just a sample method to help viiew contents
def view_npz_sample_contents(filename):
    data = np.load(filename)
    for key in data.files:
        print("First few values:\n", data[key][:6])



# # helps with predictability
# def set_global_seed(seed=42):
#     np.random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)


# splitting the data into training, testing and validation
def obtain_train_test_validation_data(dataset):
    board_state, policy, value = dataset
    # First we split into train and temp
    board_train, board_temp, policy_train, policy_temp, value_train, value_temp = train_test_split(
        board_state, policy, value, test_size=0.3, random_state=42)

    # Then we split the temp into validation and testing
    board_val, board_test, policy_val, policy_test, value_val, value_test = train_test_split(
        board_temp, policy_temp, value_temp, test_size=0.5, random_state=42)

    print("Train size:", len(board_train))
    print("Validation size:", len(board_val))
    print("Test size:", len(board_test))

    # You can now proceed to use these datasets for training, validation, and evaluation
    return (board_train, policy_train, value_train), (board_val, policy_val, value_val), (
        board_test, policy_test, value_test)


# we can use this method to
def prepare_input_output(data, game_name):
    states, policies, values = data
    # we extract the input features from data and convert it from the flattened boaed state
    # to a format tensorflow like
    # We can remove the doub
    X = np.array(states, dtype=np.float32)

    y = {
        "policy_logits": np.array(policies, dtype=np.float32),
        "value": np.array(values, dtype=np.float32)
    }
    return X, y

def commence_neural_net_pipeline(game_name, game_size):
    # we need to change this later. Just using for example
    dataset = load_npz_dataset(game_size, game_name)
    train_data, validation_data, test_data = obtain_train_test_validation_data(dataset)
    train_X, train_Y = prepare_input_output(train_data, game_name)
    validation_X, validation_Y = prepare_input_output(validation_data, game_name)
    testing_X, testing_Y = prepare_input_output(test_data, game_name)
    neural_net = Neural_Net(game=game_name, size=game_size)
    # model attribute internally calls build model so we don't need to do it from hre
    neural_net.train_model((train_X, train_Y), (validation_X, validation_Y))
    # this is going to tbe value and policy prediction
    predictions = neural_net.predict(testing_X)
    test_loss = neural_net.model.evaluate(testing_X, testing_Y, verbose=1)
    print("Test Loss:", test_loss)
    # only added to support saving drive on colab
    neural_net.save("final_model_ttt_4x4.keras")

# can be used to set instance to the model class with prepared weights file
def prepare_neural_net_instance(game, size, ai_type):
    # we don't need a neural net for pure MCTS
    if ai_type == MCTS:
        return None
    neural_net = Neural_Net(game=game, size = size)
    file_name = os.path.join("weights_ttt_4", f"{game}-{size}.keras")
    neural_net.load(file_name)
    return neural_net

if __name__ == "__main__":
    commence_neural_net_pipeline(game_name=GAME_TICTACTOE, game_size=4)




# for file in npz_files:
#     data = np.load(file)
#     print("Loaded", file, "→ keys:", data.files)


