import os

import tensorflow as tf
from keras.layers import Dense, Lambda, Input, Concatenate
from keras.models import Model
from keras.optimizers import *

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_predict):
    """Custom Huber loss function."""
    err = y_true - y_predict

    cond = tf.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * tf.square(err)
    L1 = HUBER_LOSS_DELTA * (tf.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return tf.reduce_mean(loss)


class DQN(object):
    """
    Constructs the Q-network model. Supports optional dueling architecture.
    """
    def __init__(self, state_size, action_size, net_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = net_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.discount_state = arguments['discount_state']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.dueling = arguments['dueling']
        self.optimizer_model = arguments['optimizer']
        self.model = self._build_model()
        self.model_ = self._build_model()

    def _build_model(self):

        input_shape = (self.state_size,)

        x = Input(shape=input_shape)

        if self.dueling:
            # a series of fully connected layer for estimating V(s)

            y11 = Dense(self.num_nodes, activation='relu')(x)
            y12 = Dense(self.num_nodes, activation='relu')(y11)
            y13 = Dense(1, activation="linear")(y12)

            # a series of fully connected layer for estimating A(s,a)

            y21 = Dense(self.num_nodes, activation='relu')(x)
            y22 = Dense(self.num_nodes, activation='relu')(y21)
            y23 = Dense(self.action_size, activation="linear")(y22)

            w = Concatenate(axis=-1)([y13, y23])

            # combine V(s) and A(s,a) to get Q(s,a)
            z = Lambda(lambda a: tf.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - tf.reduce_mean(a[:, 1:], keepdims=True),
                       output_shape=(self.action_size,))(w)
        else:
            # a series of fully connected layer for estimating Q(s,a)

            y1 = Dense(self.num_nodes, activation='relu')(x)
            y2 = Dense(self.num_nodes, activation='relu')(y1)
            z = Dense(self.action_size, activation="linear")(y2)

        model = Model(inputs=x, outputs=z)

        # Set optimizer based on configuration
        optimizer = None
        if self.optimizer_model == 'Adam':
            optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(learning_rate=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        if optimizer is not None:
            model.compile(loss=huber_loss, optimizer=optimizer)
        else:
            print('Optimizer not assigned!')

        # Load the model for test mode
        if self.test:
            if not os.path.isfile(self.weight_backup):
                raise ValueError(
                    "There is no weights folders in this directory, but test mode is enabled (via options --test)! "
                    "Either set load model weights or do not use --test.")
            else:
                model.load_weights(self.weight_backup)
        return model

    def train(self, x, y, sample_weight=None, epochs=1, verbose=None):
        """
        Trains the model on a batch of data.
        """
        self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)

    def predict(self, state, target=False, verbose=None):
        """
        Predicts Q-values for given state. Uses target network if specified.
        """
        if target:
            return self.model_.predict(state, verbose=verbose)
        else:
            return self.model.predict(state, verbose=verbose)

    def predict_one_sample(self, state, target=False, verbose=None):
        """
        Predicts Q-values for a single state input.
        """
        return self.predict(state.reshape(1, self.state_size), target=target, verbose=verbose).flatten()

    def update_target_model(self):
        """
        Update the target model's weights to match the primary model's weights.
        """
        self.model_.set_weights(self.model.get_weights())

    def save_weights(self):
        """
        Save the model weights to a specified file.
        """
        self.model.save(self.weight_backup)
        # self.model.save(self.weight_backup, include_optimizer=False)
