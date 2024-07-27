import keras
import sys
import os
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import numpy as np
import random
import data_sets
import loss_functions
import two_moons
import custom_layer
import itertools


def build_model(optimizer="sgd", learning_rate=0.01, gamma=7, regularization=0.01, loss_func=loss_functions.CustomLossWithWeights):
    #svm
    model = keras.models.Sequential([
        keras.layers.Input(shape=(2,)),
        custom_layer.CustomKernelDense(1, activation='linear', kernel_regularizer= tf.keras.regularizers.L2(regularization), gamma = gamma)
    ])
    np.random.seed(4)
    tf.random.set_seed(4)
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    loss_func = loss_func(model)
    model.compile(optimizer=optimizer, loss=loss_func)
    return model


def test_rbf_svm(model, optimizer, learning_rate, gamma, epochs, regularization):
    random.seed(4)
    np.random.seed(4)
    tf.random.set_seed(4)
    metric = keras.metrics.Accuracy()
    X_RANGE, Y_RANGE, X_RANGE_VARIABILITY = 5, 5, 0.3
    epochs = epochs
    no_samples = 20
    # means, labels = data_sets.generate_linear_svm_toy_dataset(no_samples, X_RANGE, Y_RANGE, X_RANGE_VARIABILITY)
    # means, labels = data_sets.generate_non_linear_svm_toy_dataset(no_samples, X_RANGE, Y_RANGE)
    # means, labels = two_moons.get_two_moons_dataset(100, 0.1)
    means ,labels = data_sets.generate_circular_dataset(size=no_samples, inner_diameter=0.5, outer_diamater=0.9)
    covariance = np.diag(np.full(2, 0.005))
    covariances = data_sets.generate_covariance_matrices(means, covariance, increasing=False)
    print(covariance)

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
            
        train_acc = metric.result()
        losses.append(tf.squeeze(loss))
        accuracies.append(train_acc)
        print(f"Loss: {loss} and accuracy: {metric.result()}")
        metric.reset_state()
        
    y_preds = model(means, training=False)

    metric.update_state(tf.math.sign(y_preds), labels)
    losses_final = []
    for mean, label, covariance, y_pred in zip(means, labels, covariances, y_preds):
        loss = model.loss((label, covariance), y_pred)
        losses_final.append(tf.squeeze(loss))
    loss = np.mean(losses_final)
    # plt.plot(range(len(losses)), losses)
    # plt.plot(range(len(losses)), accuracies)
    # plt.legend(['loss', 'accuracy'])
    # plt.show()
    # data_sets.plot_samples_with_uncertainty_and_boundary(model, means, labels, covariances, 2, 2)
    data_sets.plot_samples_with_uncertainty(means, labels, covariances, 2, 2)
    return(loss, metric.result())
    # data_sets.plot_linear_data(means, labels, X_RANGE, Y_RANGE)
    
    


#gridsearch

# optimizers = ["adam", "sgd"]
# learning_rates = [0.1, 0.01, 0.001, 0.0001]
# gammas = [10, 1, 0.1, 0.01, 0.001]
# epochss = [100]
# regularizations = [1, 0.1, 0.01, 0.001, 0.0001]
optimizers = ["sgd"]
learning_rates = [0.1]
gammas = [0.5]
epochss = [1]
regularizations = [0.1]


results = [["optimizer", "learning_rate", "gamma", "epochs", "regularization", "loss", "accuracy"]]
for opt, lr, gm, ep, reg in itertools.product(optimizers, learning_rates, gammas, epochss, regularizations):
    model = build_model(optimizer=opt, learning_rate=lr, gamma=gm, regularization=reg)
    result = test_rbf_svm(model=model, optimizer=model.optimizer, learning_rate=lr, gamma=gm, epochs=ep, regularization=reg)
    print(f"optimizer: {opt}, learning_rate: {lr}, gamma: {gm}, epochs: {ep}, regularization: {reg}, loss: {result[0]}, accuracy: {result[1]}")
    results.append([opt, lr, gm, ep, reg, result[0], result[1]])

# with open("results.csv", "w") as f:
#     for row in results:
#         f.write(",".join(map(str, row)) + "\n")

# visualize loss landscape
# def evaluate_loss(model, weights, means, labels, covariances):
#     model.set_weights(weights)
#     losses = []
#     for mean, y_true, covariance in zip(means, labels, covariances):
#         mean = mean.reshape(1, 2)
#         y_pred = model(mean, training=False)
#         loss = model.loss((y_true, covariance), y_pred)
#         loss += model.losses
#         losses.append(tf.squeeze(loss).numpy())
#     return np.mean(losses)

# model = build_model(gamma = 5)

# w1_range = np.linspace(-2, 2, 50)
# w2_range = np.linspace(-2, 2, 50)
# W1, W2 = np.meshgrid(w1_range, w2_range)
# Loss = np.zeros_like(W1)

# biasses = np.linspace(-2, 2, 500)
# test_bias = np.zeros_like(biasses)

# no_samples = 20
# means ,labels = data_sets.generate_circular_dataset(no_samples, 8, 10)
# covariances = data_sets.generate_covariance_matrices(means, [[0.005,0],[0,0.005]], increasing=False)
# means = 2 * (means-np.min(means)) / (np.max(means)-np.min(means)) - 1
# initial_weights = model.get_weights()
# bias = -0.001


# # for weights
# for i in range(W1.shape[0]):
#     for j in range(W1.shape[1]):
#         weights = [np.array([[W1[i, j]], [W2[i, j]]]), np.array([bias])]
#         Loss[i, j] = evaluate_loss(model, weights, means, labels, covariances)  
#         print(f"Loss: {Loss[i, j]} at w1: {model.trainable_weights[0][0]} and w2: {model.trainable_weights[0][1]} and bias: {model.trainable_weights[1].numpy()}")

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(W1, W2, Loss, cmap='viridis')
# ax.set_xlabel('Weight 1')
# ax.set_ylabel('Weight 2')
# ax.set_zlabel('Loss')
# ax.set_title('Loss Landscape')
# plt.show()
# data_sets.plot_samples_with_uncertainty(means, labels, covariances, 2, 2)

#for biasses
# for i in range(test_bias.shape[0]):
#         weights = [np.array([[0], [0]]), np.array([biasses[i]])]
#         test_bias[i] = evaluate_loss(model, weights, means, labels, covariances)  
#         print(f"Loss: {test_bias[i]} at w1: {model.trainable_weights[0][0]} and w2: {model.trainable_weights[0][1]} and bias: {model.trainable_weights[1].numpy()}")

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# ax.plot(biasses, test_bias)
# ax.set_xlabel('Bias')
# ax.set_ylabel('Loss')
# ax.set_title('Loss Landscape')
# plt.show()