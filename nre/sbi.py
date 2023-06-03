import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle


class nre():
    def __init__(self):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout

        self.optimizer = tf.keras.optimizers.legacy.Adam(lr=1e-3)

    def basic_model(
            self, input_dim, output_dim, layer_sizes, activation,
            drop_val, output_activation):
        
        a0 = self.Inputs(shape=(input_dim,))
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation=output_activation)(a0)
        self.model = self.Model(inputs, outputs)

    def _train_step(self, params, truth):

            r"""
            This function is used to calculate the loss value at each epoch and
            adjust the weights and biases of the neural networks via the
            optimizer algorithm.
            """

            with tf.GradientTape() as tape:
                prediction = tf.transpose(self.model(params, training=True))[0]
                truth = tf.convert_to_tensor(truth)
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(truth, prediction)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients,
                        self.model.trainable_variables))
                return loss

    def training(self, epochs, data, labels, early_stop, batch_size=32):

        data_train, data_test, labels_train, labels_test = \
                train_test_split(data, labels, test_size=0.2)
        
        train_dataset = np.hstack([data_train, labels_train[:, np.newaxis]]).astype(np.float32)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.batch(batch_size)

        loss_history = []
        test_loss_history = []
        c = 0
        for i in range(epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()

            for x in train_dataset:
                loss = self._train_step(x[:, :-1], x[:, -1]).numpy()
                epoch_loss_avg.update_state(loss)
            loss_history.append(epoch_loss_avg.result())

            test_pred = tf.transpose(self.model(data_test, training=True))[0]
            loss_test = tf.keras.losses.BinaryCrossentropy(from_logits=False)(labels_test, test_pred)

            print('Epoch: {:d}, Loss: {:.4f}, Test Loss: {:.4f}'.format(i, loss, loss_test))

            test_loss_history.append(loss_test)

            if early_stop:
                c += 1
                if i == 0:
                    minimum_loss = test_loss_history[-1]
                    minimum_epoch = i
                    minimum_model = None
                else:
                    if test_loss_history[-1] < minimum_loss:
                        minimum_loss = test_loss_history[-1]
                        minimum_epoch = i
                        minimum_model = self.model
                        c = 0
                if minimum_model:
                    if c == round((epochs/100)*2):
                        print('Early stopped. Epochs used = ' + str(i) +
                                '. Minimum at epoch = ' + str(minimum_epoch))
                        return minimum_model, data_test, labels_test
        return self.model, data_test, labels_test
