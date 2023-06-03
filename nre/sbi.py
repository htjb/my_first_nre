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

    def build_model(
            self, input_dim, output_dim, layer_sizes, activation,
            drop_val=0, output_activation='sigmoid'):
        
        a0 = self.Inputs(shape=(input_dim,))
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation=output_activation)(a0)
        self.model = self.Model(inputs, outputs)
    
    def build_simulations(self, simulation_func, prior_function, n=10000):

        self.simulation_func = simulation_func
        self.prior_function = prior_function

        theta = self.prior_function(n)

        # generate lots of simulations 
        sims, params = [], []
        for i in range(len(theta)):
            sims.append(self.simulation_func(*theta[i]))
            params.append(theta[i])
        sims = np.array(sims)
        params = np.array(params)

        idx = np.arange(0, len(sims), 1)
        shuffle(idx)
        mis_labeled_params = params[idx]

        data = []
        for i in range(len(sims)):
            data.append([*sims[i], *params[i], 1])
            data.append([*sims[i], *mis_labeled_params[i], 0])
        data = np.array(data)

        shuffle(idx)
        self.data = data[idx, :-1]
        self.labels = data[idx, -1]

    def training(self, epochs, early_stop=True, batch_size=32):

        data_train, data_test, labels_train, labels_test = \
                train_test_split(self.data, self.labels, test_size=0.2)
        
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
    
    def __call__(self, true_y, prior_prob, iters=2000):
        """Draw samples from the nre"""

        self.samples = self.prior_function(iters)

        prior_probability = prior_prob(self.samples)

        posterior_value = []
        r_values = []
        for i in range(len(prior_probability)):
            params = tf.convert_to_tensor(np.array([[*true_y, *self.samples[i]]]).astype('float32'))
            r = self.model(params).numpy()[0]
            r_values.append(r)
            posterior_value.append(r*prior_probability[i])
        self.posterior_value = np.array(posterior_value).T[0]
        self.r_values = np.array(r_values).T[0]