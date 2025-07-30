import os

import numpy as np

from keras import layers, models
from keras.src.losses import CategoricalCrossentropy, MeanSquaredError
from keras.src.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras.src.optimizers import Adam

from constant_strings import NUMBER_OF_NN_CHANNELS, ACTIVATION_TANH, NEURAL_NET_LEARNING_RATE


class Neural_Net:
    # we choose 64 coz that is a good number given the game complexity
    # and available computational power
    # more complex games like go might need
    # performed worse
    # def __init__(self, game: str, size: int, primary_filter_count: int = 64, policy_filter_count: int = 2, value_filter_count: int = 1, residual_layer_count: int = 4, primary_kernel_size = 3, secondary_kernel_size =1):
    # performer better
    def __init__(self, game: str, size: int, primary_filter_count: int = 128, policy_filter_count: int = 2,
                 value_filter_count: int = 1, residual_layer_count: int = 12, primary_kernel_size=3,
                 secondary_kernel_size=1):
        self.game = game
        # size of the game
        self.size = size
        # 256 would be ideal and used by alpha zero, but 64 should be sufficient
        self.primary_filter_count = primary_filter_count
        # Refer paper. They use the same
        self.policy_filter_count = policy_filter_count
        # Refer paper. They use the same
        self.value_filter_count = value_filter_count
        # number of residual layers
        self.residual_layer_count = residual_layer_count
        # the size of the kernel patch that would go all across the board 3*3 is quite nice and standard
        self.primary_kernel_size = primary_kernel_size
        # a smaller kernel is used in the policy and value parts
        self.secondary_kernel_size = secondary_kernel_size
        # the model of the Neural Net
        self.model = self.build_model()

    # Getter methods in line with encapsulation
    def get_game(self):
        return self.game

    def get_size(self):
        return self.size

    def get_primary_filter_count(self):
        return self.primary_filter_count

    def get_policy_filter_count(self):
        return self.policy_filter_count

    def get_value_filter_count(self):
        return self.value_filter_count

    def get_residual_layer_count(self):
        return self.residual_layer_count

    def get_primary_kernel_size(self):
        return self.primary_kernel_size

    def get_secondary_kernel_size(self):
        return self.secondary_kernel_size


    # convolutional bundling to capture the repeated code blocks used
    # skip_layer which defaults to None but can be used
    def set_up_minimal_convolutional_bundle(self, filters, kernel_size,input_layer, skip_layer = None):
        # padding same allows the kernel patch to same makes sure the edges of the input are not missed
        # use bias set to false since we are going to do a batch normalization next and that works by calculating mean and
        # would effectively remove any bias used
        resultant_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same",
                                        use_bias=False)(input_layer)
        resultant_layer = layers.BatchNormalization()(resultant_layer)
        if skip_layer is not None:
            resultant_layer = layers.Add()([skip_layer, resultant_layer])
        resultant_layer = layers.ReLU()(resultant_layer)
        return resultant_layer

    # establish link between prior layer(usually convolutional body and the policy head
    def setup_policy_head_link(self, prior_layer):
        policy_layer_filter_count = self.get_policy_filter_count()
        game_board_size = self.get_size()
        secondary_kernel_size = self.get_secondary_kernel_size()
        # since policy layer's output is supposed to be a board of probabilities of the same size as the
        # size of the board
        policy_layer_output_units = game_board_size * game_board_size
        policy_head_resultant_layer = self.set_up_minimal_convolutional_bundle(filters = policy_layer_filter_count, kernel_size=secondary_kernel_size, input_layer=prior_layer)
        policy_head_resultant_layer = layers.Flatten()(policy_head_resultant_layer)
        # think as the dense head of the policy layer
        policy_logits = layers.Dense(units=policy_layer_output_units, name = "policy_logits")(policy_head_resultant_layer)
        # link for reference as to why from_logits=True and therefore excluding the softmax operation in the last model layer is more numerically stable for the loss calculation.
        # https://stackoverflow.com/questions/66454675/why-is-computing-the-loss-from-logits-more-numerically-stable
        # above comment is the reason why we skipped an activation function with logits
        # logits are more stable and use activation function later when checking losses
        return policy_logits

    # might be a bit too shallow
    # perhaps add two versions for convenience
    # establish link between prior layer(usually convolutional body and the value head
    def setup_value_head_link(self, prior_layer):
        value_layer_output_units = 1
        value_layer_filter_count = self.get_value_filter_count()
        secondary_kernel_size = self.get_secondary_kernel_size()
        value_head_resultant_layer = self.set_up_minimal_convolutional_bundle(filters=value_layer_filter_count, kernel_size=secondary_kernel_size, input_layer=prior_layer)
        value_head_resultant_layer = layers.Flatten()(value_head_resultant_layer)
        # think as the dense head of the value layer
        value_prediction = layers.Dense(units = value_layer_output_units, activation=ACTIVATION_TANH, name = "value")(value_head_resultant_layer)
        return value_prediction

    # assembles our model
    def build_model(self):
        game_board_size = self.get_size()
        primary_filter_count = self.get_primary_filter_count()
        primary_kernel_size = self.get_primary_kernel_size()
        residual_layer_count = self.get_residual_layer_count()
        # number of nn channels is set
        input_layer = layers.Input(shape=(game_board_size, game_board_size, NUMBER_OF_NN_CHANNELS))
        # pre-residual stacking convolutional layer setup
        resultant_layer = self.set_up_minimal_convolutional_bundle(filters=primary_filter_count, kernel_size=primary_kernel_size, input_layer=input_layer)
         # stacking residual layers
        for layer_index in range(residual_layer_count):
            resultant_layer = self.add_residual_block(resultant_layer)
        # Policy Head set up
        policy_logits = self.setup_policy_head_link(prior_layer=resultant_layer)
        # Value Head set up
        value_prediction = self.setup_value_head_link(prior_layer=resultant_layer)
        return models.Model(inputs=input_layer, outputs=[policy_logits, value_prediction])



    # EXAMPLE taken FROM ALPHA_GO_PAPER that we can use with lesser filters
    # Each residual block applies the following modules sequentially to its input:
    # 1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    # 2. Batch normalisation
    # 3. A rectifier non-linearity
    # 4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    # 5. Batch normalisation
    # 6. A skip connection that adds the input to the block
    # 7. A rectifier non-linearity

    # core idea is quite simple
    # multiple convolutional layers can often lead to over-smoothening and
    # diminishing/exploding gradients.To tackle that residual blocks are used
    # they contain skip layers where the input is added to the final output to prevent
    # diminishing gradient loss
    def add_residual_block(self, output_from_previous_layers):
        skip_layer = output_from_previous_layers
        primary_filter_count = self.primary_filter_count
        primary_kernel_size = self.get_primary_kernel_size()
        initial_convolutional_bundle = self.set_up_minimal_convolutional_bundle(filters=primary_filter_count,kernel_size=primary_kernel_size,input_layer=output_from_previous_layers)
        final_convolutional_bundle = self.set_up_minimal_convolutional_bundle(filters=primary_filter_count, kernel_size=primary_kernel_size, input_layer=initial_convolutional_bundle, skip_layer=skip_layer)
        return final_convolutional_bundle


    # after layering residual nets, we follow the dual-res. It is lighter and does the job
    # just note to be kept in mind

    # slightly worse
    # def train_model(self, training_bundle,validation_bundle, epochs = 10, batch_size = 128, ckpt_dir: str = "checkpoints", save_best_only: bool = True):
    #     # to generate folder to create file in the given folder
    #     os.makedirs(ckpt_dir, exist_ok=True)
    #     ckpt_path = os.path.join(
    #         ckpt_dir, "epoch{epoch:02d}-val_loss{val_loss:.4f}.h5"
    #     )
    #     checkpoint_cb = ModelCheckpoint(
    #         ckpt_path,
    #         monitor="val_loss",
    #         verbose=1,
    #         save_best_only=save_best_only,
    #         save_weights_only=False,
    #     )
    #     self.model.compile(
    #         optimizer=Adam(learning_rate=NEURAL_NET_LEARNING_RATE),
    #         loss = {
    #             "policy_logits": CategoricalCrossentropy(from_logits=True),
    #             "value" : MeanSquaredError()
    #            },
    #         # basically treat both values equally when adjusting weights
    #         loss_weights={"policy_logits": 1.0, "value": 1.0})
    #     train_X, train_Y = training_bundle
    #     self.model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size,validation_data=validation_bundle,callbacks=[checkpoint_cb])

        # after layering residual nets, we follow the dual-res. It is lighter and does the job
        # just note to be kept in mind
    def train_model(
                self,
                training_bundle,
                validation_bundle,
                epochs: int = 100,  # upper bound – ES will halt earlier
                batch_size: int = 128,
                ckpt_dir: str = "checkpoints",
                save_best_only: bool = True,
        ):
            # this part solely to keep saving good model snapshots
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                # could have named it anything reallt just chose this since the naming is somehwat descriptive
                ckpt_dir, "epoch{epoch:02d}-val{val_loss:.4f}.keras"
            )

            checkpoint_callback = ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                verbose=1,
                save_best_only=save_best_only,
                save_weights_only=False,
            )
            learning_rate_modifier_callback = ReduceLROnPlateau(
                factor=0.5,
                # wait 3 epochs with no val-loss gain
                patience=3,
                min_lr=1e-5,
                verbose=1,
            )

            early_stopping_callback = EarlyStopping(
                # stop after 10 stagnant epochs
                patience=10,
                restore_best_weights=True,
                verbose=1,
            )
            self.model.compile(
                optimizer=Adam(learning_rate=NEURAL_NET_LEARNING_RATE),  # e.g. 1e-3
                loss={
                    "policy_logits": CategoricalCrossentropy(from_logits=True),
                    "value": MeanSquaredError(),
                },
                loss_weights={"policy_logits": 1.0, "value": 1.0},
            )
            train_X, train_Y = training_bundle
            val_X, val_Y = validation_bundle
            self.model.fit(
                train_X,
                train_Y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(val_X, val_Y),
                shuffle=True,
                callbacks=[checkpoint_callback, learning_rate_modifier_callback, early_stopping_callback],
                verbose=2,
            )
    def predict(self, board_tensor):
        # verbose is simply so it doesn't overwhelm the logs with console logs for each epoch
        # both approaches do the same thing - TensorFlow needs a 4 needs
        # approach-1
        # board_tensor  = board_tensor[None, ...]
        # prediction = self.model.predict(board_tensor[None, ...], verbose=0)

        # approach-2
        # batched_input = np.expand_dims(board_tensor, axis=0)
        prediction = self.model.predict(board_tensor, verbose=0)
        return prediction[0].squeeze(), prediction[1].squeeze()

    # helps us save the finished model weights to a file path
    def save(self, path):
        self.model.save(path)

    # allows us to load a model from a given path
    def load(self, path):
        self.model = models.load_model(path)



