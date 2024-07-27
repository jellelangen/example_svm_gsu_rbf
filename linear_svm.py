import keras
import sys
import os
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import numpy as np

import data_sets
import loss_functions
import custom_layer


X_RANGE, Y_RANGE, X_RANGE_VARIABILITY = 10, 10, 0.3
epochs = 100
no_samples = 6
means, labels = data_sets.generate_linear_svm_toy_dataset(no_samples, X_RANGE, Y_RANGE, X_RANGE_VARIABILITY)

covariances = data_sets.generate_covariance_matrices(means, [[0.015,0],[0,0.04]], increasing=True)

#normalize data
means = 2 * (means-np.min(means)) / (np.max(means)-np.min(means)) - 1


#svm
model = keras.models.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(1, activation='linear', kernel_regularizer= tf.keras.regularizers.L2(0.01))
])


# optimizer = keras.optimizers.Adam()
optimizer = keras.optimizers.SGD(learning_rate=0.01)
loss_func = loss_functions.CustomLossWithWeights(model)
metric = keras.metrics.Accuracy()
model.compile(optimizer=optimizer, loss=loss_func)
losses = []
accuracies = []
for epoch in range(epochs+1):
    data = zip(means, labels, covariances)
    print(f"\n Start of training epoch {epoch}")
    for mean, y_true, covariance in data:
        mean = mean.reshape(1,2)
        with tf.GradientTape() as tape:
            
            y_pred = model(mean, training=True)
            loss = loss_func((y_true, covariance), y_pred )
            loss += model.losses
        
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        metric.update_state(tf.math.sign(y_pred/tf.sqrt(tf.reduce_sum(tf.square(model.trainable_weights[0])))), y_true)

    train_acc = metric.result()
    losses.append(tf.squeeze(loss))
    accuracies.append(train_acc)

    # if epoch % (epochs//10) == 0:
    print(f"epoch {epoch} - accuracy {train_acc} and loss: {loss}")
    metric.reset_state()



plt.plot(range(len(losses)), losses)
plt.plot(range(len(losses)), accuracies)
plt.legend(['loss', 'accuracy'])
plt.show()

data_sets.plot_samples_with_uncertainty_and_boundary(model, means, labels, covariances, 1, 1)
