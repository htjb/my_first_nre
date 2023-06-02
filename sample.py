import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

def prior():
    a = np.random.uniform(0, 10, 1)
    b = np.random.uniform(0, 10, 1)
    return [a, b]

def data_model(x, a, b):
    return a*x + b

model = keras.models.load_model('trained_nre.h5',
                compile=False)
true_y = np.loadtxt('true_y.txt')
true_params = np.loadtxt('true_params.txt')

x = np.linspace(10, 20, 10)

prior_prob = 1/10*1/10

iters = 2000

samples = []
posterior_value = []
for i in range(iters):
    samples.append(prior())
    params = tf.convert_to_tensor(np.array([[*true_y, *samples[-1]]]).astype('float32'))
    r = model(params)
    posterior_value.append(r*prior_prob)
samples = np.array(samples)
posterior_value = np.array(posterior_value)

im = plt.scatter(samples[:, 0], samples[:, 1], cmap='inferno', c=posterior_value)
plt.colorbar(im, label='Posterior Prob.')
plt.axvline(true_params[0])
plt.axhline(true_params[1])
plt.xlabel('a')
plt.ylabel('b')
plt.savefig('nre_example.png')
plt.show()
