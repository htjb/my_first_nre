import numpy as np
from random import shuffle
from nre.sbi import nre
import tensorflow as tf


x = np.linspace(10, 20, 10)
true_a, true_b = 5, 2
true_y = true_a*x + true_b

np.savetxt('true_y.txt', true_y)
np.savetxt('true_params.txt', [true_a, true_b])

def simulation(a, b):
    return a*x + b

def prior():
    a = np.random.uniform(0, 10)
    b = np.random.uniform(0, 10)
    return [a, b]

nre_instance = nre()

nre_instance.build_simulations(simulation, prior)

nre_instance.build_model(len(true_y)+2, 1, 3*[2*(len(true_y)+2)], 
            'sigmoid')

model, data_test, labels_test = nre_instance.training(2000, batch_size=100)

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
    
