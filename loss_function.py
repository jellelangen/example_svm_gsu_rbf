import tensorflow as tf
import numpy as np



epsilon = 1e-10
pi_value = tf.constant(3.141592653589793)
class CustomLossWithWeights:
    def __init__(self, model):
        self.model = model
    
    def compute_dx_i(self, y_true, y_pred):
        y_true = tf.where(y_true == 0, tf.cast(-1, tf.float32), y_true) #convert label from 0 to -1 to work with tf
        return 1 - (y_true * y_pred)

    def mulcovweights(self, covariance_matrix, weights):
        # weights_reshaped = tf.reshape(weights, shape=(-1, 1))

        intermediate_product = 2 * tf.matmul(weights, covariance_matrix, transpose_a = True)
        
        #d_sigma = intermediate * weights
        d_sigma = tf.math.sqrt(tf.matmul(intermediate_product, weights))
        d_sigma_scalar = tf.squeeze(d_sigma)

        return d_sigma_scalar

    def __call__(self, y_true, y_pred):
        epsilon = 0.00001
        label = y_true[0] 
        covariance = y_true[1] 
        dx_i = self.compute_dx_i(label, y_pred)
        
        d_sigma = self.mulcovweights(covariance, self.model.trainable_weights[0])
        d_sigma = tf.maximum(d_sigma, epsilon) #if d_sigma is zero add a small epsilon.
        exponent = tf.math.exp(- ( tf.math.pow(dx_i,2) / tf.math.pow(d_sigma,2) ) )
        first_term = (dx_i/2) * (tf.math.erf( dx_i / d_sigma ) + 1)
        second_term = ( d_sigma / ( 2 * tf.math.sqrt(pi_value) ) ) * exponent
        
        loss = first_term + second_term
        return loss

class CustomLossWithWeightsNoModel(tf.keras.losses.Loss):
    def __init__(self, layer, covariance_dim, covariance, name="custom_loss_with_weights_no_model"):
        super().__init__(name=name)
        self.layer = layer
        self.covariance_dim = covariance_dim
        self.covariance = covariance
        

    def compute_dx_i(self, y_true, y_pred):
        y_true = tf.where(y_true == 0, tf.cast(-1, tf.float32), y_true)  # convert label from 0 to -1 to work with tf
        return 1 - (y_true * y_pred)

    def mulcovweights(self, covariance_matrix, weights):
        intermediate_product = 2 * tf.matmul(weights, covariance_matrix, transpose_a=True)
        d_sigma = tf.math.sqrt(tf.matmul(intermediate_product, weights))
        d_sigma_scalar = tf.squeeze(d_sigma)
        return d_sigma_scalar

    def call(self, y_true, y_pred):
        epsilon = 0.00001
        label = y_true

        covariance = np.diag(np.full(self.covariance_dim, self.covariance)).astype(np.float32)

        dx_i = self.compute_dx_i(label, y_pred)

        weights = self.layer.weights[0]

        d_sigma = self.mulcovweights(covariance, weights)
        d_sigma = tf.maximum(d_sigma, epsilon)  # if d_sigma is zero add a small epsilon.
        exponent = tf.math.exp(-(tf.math.pow(dx_i, 2) / tf.math.pow(d_sigma, 2)))
        first_term = (dx_i / 2) * (tf.math.erf(dx_i / d_sigma) + 1)
        second_term = (d_sigma / (2 * tf.math.sqrt(pi_value))) * exponent

        loss = first_term + second_term
        return loss