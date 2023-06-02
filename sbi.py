import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle


class network_models():
    def __init__(self):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout

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
        model = self.Model(inputs, outputs)
        return model

def _train_step(model, params, truth):

        r"""
        This function is used to calculate the loss value at each epoch and
        adjust the weights and biases of the neural networks via the
        optimizer algorithm.
        """

        with tf.GradientTape() as tape:
            prediction = tf.transpose(model(params, training=True))[0]
            truth = tf.convert_to_tensor(truth)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(truth, prediction)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients,
                    model.trainable_variables))
            return loss

def training(epochs, data, labels, model, early_stop, batch_size=32):

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
            loss = _train_step(model, x[:, :-1], x[:, -1]).numpy()
            epoch_loss_avg.update_state(loss)
        loss_history.append(epoch_loss_avg.result())

        test_pred = tf.transpose(model(data_test, training=True))[0]
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
                    minimum_model = model
                    c = 0
            if minimum_model:
                if c == round((epochs/100)*2):
                    print('Early stopped. Epochs used = ' + str(i) +
                            '. Minimum at epoch = ' + str(minimum_epoch))
                    return minimum_model, data_test, labels_test
    return model, data_test, labels_test

# parameters are a and b
# some independent variable x
# and y define as y = ax+ b

x = np.linspace(10, 20, 10)
true_a, true_b = 5, 2
true_y = true_a*x + true_b

np.savetxt('true_y.txt', true_y)
np.savetxt('true_params.txt', [true_a, true_b])

def simulation(x, a, b):
    return a*x + b

# generate lots of simulations 
sims, params = [], []
for i in range(10000):
    a = np.random.uniform(0, 10)
    b = np.random.uniform(0, 10)
    sims.append(simulation(x, a, b))
    params.append([a, b])
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
data = data[idx]

classifier = network_models().basic_model(len(true_y)+2, 1, 3*[2*(len(true_y)+2)], 
            'sigmoid',
            0, 'sigmoid')

optimizer = tf.keras.optimizers.legacy.Adam(lr=1e-3)

model, data_test, labels_test = training(2000, data[:, :-1], data[:, -1], classifier, True, batch_size=100)

model.save('trained_nre.h5')

test_predictions = tf.transpose(model(data_test, training=True))[0]

pos, negatives, confused = 0, 0, 0
for i in range(len(test_predictions)):
    if test_predictions[i] > 0.7:
        if labels_test[i] == 1:
            pos += 1
        else:
            negatives += 1
    if test_predictions[i] < 0.3:
        if labels_test[i] == 0:
            pos += 1
        else:
            negatives += 1
    else:
        confused += 1

print(pos, negatives, confused)
    
