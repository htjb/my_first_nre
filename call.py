import numpy as np
from random import shuffle
from nre.sbi import nre
import tensorflow as tf
import matplotlib.pyplot as plt

# define the real toy data
x = np.linspace(10, 20, 10)
true_a, true_b = 5, 2
true_params = [true_a, true_b]
true_y = true_a*x + true_b

# save the real toy data
np.savetxt('true_y.txt', true_y)
np.savetxt('true_params.txt', [true_a, true_b])

# define the simulation function and the prior function
def simulation(a, b):
    return a*x + b
def prior():
    a = np.random.uniform(0, 10)
    b = np.random.uniform(0, 10)
    return [a, b]

# initialise the nre
nrei = nre()
# build the simulation data given the prior function and the simulation func
nrei.build_simulations(simulation, prior)
# build the neural network
nrei.build_model(len(true_y)+2, 1, 3*[2*(len(true_y)+2)], 'sigmoid')
# train the neural network
model, data_test, labels_test = nrei.training(2000, batch_size=100)
# save teh model
model.save('testing.h5')

# analytic prior probability to get posterior probability
prior_prob = 1/10*1/10

# generate samples from the nre and calculate their posterior probability
nrei(true_y, prior_prob)

# plot the results
im = plt.scatter(nrei.samples[:, 0], nrei.samples[:, 1], cmap='inferno', c=nrei.posterior_value)
plt.colorbar(im, label='Posterior Prob.')
plt.axvline(true_params[0])
plt.axhline(true_params[1])
plt.xlabel('a')
plt.ylabel('b')
#plt.savefig('nre_example_test.png')
plt.show()