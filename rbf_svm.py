import keras
import sys
import os
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import numpy as np
import random
import data_sets
import loss_functions
import custom_layer
import itertools






random.seed(4)
np.random.seed(4)
tf.random.set_seed(4)
metric = keras.metrics.Accuracy()
X_RANGE, Y_RANGE, X_RANGE_VARIABILITY = 5, 5, 0.3
epochs = 10
no_samples = 20
gamma = 3
regularization = 0.001
learning_rate = 0.1
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
means ,labels = data_sets.generate_circular_dataset(size=no_samples, inner_diameter=0.5, outer_diamater=0.9)
covariance = np.diag(np.full(2, 0.005))
covariances = data_sets.generate_covariance_matrices(means, covariance, increasing=False)


model = keras.models.Sequential([
        keras.layers.Input(shape=(2,)),
        custom_layer.CustomKernelDense(1, activation='linear', kernel_regularizer= tf.keras.regularizers.L2(regularization), gamma = gamma)
    ])
    
loss_func = loss_functions.CustomLossWithWeights(model)
model.compile(optimizer=optimizer, loss=loss_func)

#normalize data between -1 and 1
means = 2 * (means-np.min(means)) / (np.max(means)-np.min(means)) - 1



losses = []
accuracies = []
for epoch in range(epochs+1):
    data = zip(means, labels, covariances)
    print(f"\n Start of training epoch {epoch}")
    for mean, y_true, covariance in data:
        mean = np.expand_dims(mean, axis=0)
        with tf.GradientTape() as tape:
            y_pred = model(mean, training=True)
            loss = model.loss((y_true, covariance), y_pred )
            loss += model.losses  
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        metric.update_state(tf.math.sign(y_pred), tf.reshape(y_true, [1]))
        
    
    y_preds = model(means, training=False)
    metric.update_state(tf.math.sign(y_preds), labels)
    acc = metric.result()
    losses.append(tf.squeeze(loss))
    accuracies.append(acc)
    print(f"Loss: {loss} and accuracy: {acc}")
    metric.reset_state()
    
y_preds = model(means, training=False)
metric.update_state(tf.math.sign(y_preds), labels)
print(f"final accuracy: {metric.result()}")
losses_final = []
for mean, label, covariance, y_pred in zip(means, labels, covariances, y_preds):
    loss = model.loss((label, covariance), y_pred)
    losses_final.append(tf.squeeze(loss))
loss = np.mean(losses_final)
plt.plot(range(len(losses)), losses)
plt.plot(range(len(losses)), accuracies)
plt.legend(['loss', 'accuracy'])
plt.show()
data_sets.plot_samples_with_uncertainty_and_boundary(model, means, labels, covariances, 2, 2)


