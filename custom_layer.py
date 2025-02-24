from keras import activations
from keras import backend
from keras import constraints
from keras import dtype_policies
from keras import initializers
from keras import ops
from keras import quantizers
from keras import regularizers
# from keras.api_export import keras_export
# from keras.layers.input_spec import InputSpec
from keras import InputSpec #replaced commented out line above
# from keras.layers.layer import Layer
from keras.layers import Layer #replaced commented out line above
import tensorflow as tf

# @keras_export("keras.layers.Dense") #commented out to deal with import on line 9 which was causing issues
# class CustomKernelDense(Layer):
#     """Just your regular densely-connected NN layer.

#     `Dense` implements the operation:
#     `output = activation(dot(input, kernel) + bias)`
#     where `activation` is the element-wise activation function
#     passed as the `activation` argument, `kernel` is a weights matrix
#     created by the layer, and `bias` is a bias vector created by the layer
#     (only applicable if `use_bias` is `True`).

#     Note: If the input to the layer has a rank greater than 2, `Dense`
#     computes the dot product between the `inputs` and the `kernel` along the
#     last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
#     For example, if input has dimensions `(batch_size, d0, d1)`, then we create
#     a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
#     of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
#     `batch_size * d0` such sub-tensors). The output in this case will have
#     shape `(batch_size, d0, units)`.

#     Args:
#         units: Positive integer, dimensionality of the output space.
#         activation: Activation function to use.
#             If you don't specify anything, no activation is applied
#             (ie. "linear" activation: `a(x) = x`).
#         use_bias: Boolean, whether the layer uses a bias vector.
#         kernel_initializer: Initializer for the `kernel` weights matrix.
#         bias_initializer: Initializer for the bias vector.
#         kernel_regularizer: Regularizer function applied to
#             the `kernel` weights matrix.
#         bias_regularizer: Regularizer function applied to the bias vector.
#         activity_regularizer: Regularizer function applied to
#             the output of the layer (its "activation").
#         kernel_constraint: Constraint function applied to
#             the `kernel` weights matrix.
#         bias_constraint: Constraint function applied to the bias vector.
#         lora_rank: Optional integer. If set, the layer's forward pass
#             will implement LoRA (Low-Rank Adaptation)
#             with the provided rank. LoRA sets the layer's kernel
#             to non-trainable and replaces it with a delta over the
#             original kernel, obtained via multiplying two lower-rank
#             trainable matrices. This can be useful to reduce the
#             computation cost of fine-tuning large dense layers.
#             You can also enable LoRA on an existing
#             `Dense` layer by calling `layer.enable_lora(rank)`.

#     Input shape:
#         N-D tensor with shape: `(batch_size, ..., input_dim)`.
#         The most common situation would be
#         a 2D input with shape `(batch_size, input_dim)`.

#     Output shape:
#         N-D tensor with shape: `(batch_size, ..., units)`.
#         For instance, for a 2D input with shape `(batch_size, input_dim)`,
#         the output would have shape `(batch_size, units)`.
#     """

#     def __init__(
#         self,
#         units,
#         gamma=1,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         lora_rank=None,
#         **kwargs,
#     ):
#         super().__init__(activity_regularizer=activity_regularizer, **kwargs)
#         self.units = units
#         self.gamma = gamma
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.lora_rank = lora_rank
#         self.lora_enabled = False
#         self.input_spec = InputSpec(min_ndim=2)
#         self.supports_masking = True

#     def build(self, input_shape):
#         self.centers = self.add_weight(
#             name='centers',
#             shape=(input_shape[-1], self.units),
#             initializer='uniform',
#             trainable=True)

#         if self.activation is None:
#             self.activation = lambda x: x

#     @property
#     def kernel(self):
#         if not self.built:
#             raise AttributeError(
#                 "You must build the layer before accessing `kernel`."
#             )
#         if self.lora_enabled:
#             return self._kernel + ops.matmul(
#                 self.lora_kernel_a, self.lora_kernel_b
#             )
#         return self._kernel

#     def call(self, inputs):
#         #https://github.com/PetraVidnerova/rbf_keras/blob/master/rbflayer.py
#         # x = ops.matmul(inputs, self.kernel) #replace by different kernel 
#         input_expand = tf.expand_dims(inputs, axis=1)
#         centers_expand = tf.expand_dims(self.centers, axis=0)
#         diff = input_expand - centers_expand
#         l2_dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
#         x = tf.exp(-self.gamma * l2_dist_sq)
#         # if self.bias is not None:
#         #     x = ops.add(x, self.bias)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x

#     def compute_output_shape(self, input_shape):
#         output_shape = list(input_shape)
#         output_shape[-1] = self.units
#         return tuple(output_shape)

#     def enable_lora(
#         self, rank, a_initializer="he_uniform", b_initializer="zeros"
#     ):
#         if self.kernel_constraint:
#             raise ValueError(
#                 "Lora is incompatible with kernel constraints. "
#                 "In order to enable lora on this layer, remove the "
#                 "`kernel_constraint` argument."
#             )
#         if not self.built:
#             raise ValueError(
#                 "Cannot enable lora on a layer that isn't yet built."
#             )
#         if self.lora_enabled:
#             raise ValueError(
#                 "lora is already enabled. "
#                 "This can only be done once per layer."
#             )
#         self._tracker.unlock()
#         self.lora_kernel_a = self.add_weight(
#             name="lora_kernel_a",
#             shape=(self.kernel.shape[0], rank),
#             initializer=initializers.get(a_initializer),
#             regularizer=self.kernel_regularizer,
#         )
#         self.lora_kernel_b = self.add_weight(
#             name="lora_kernel_b",
#             shape=(rank, self.kernel.shape[1]),
#             initializer=initializers.get(b_initializer),
#             regularizer=self.kernel_regularizer,
#         )
#         self._kernel.trainable = False
#         self._tracker.lock()
#         self.lora_enabled = True
#         self.lora_rank = rank

#     def save_own_variables(self, store):
#         # Do nothing if the layer isn't yet built
#         if not self.built:
#             return
#         # The keys of the `store` will be saved as determined because the
#         # default ordering will change after quantization
#         kernel_value, kernel_scale = self._get_kernel_with_merged_lora()
#         store["0"] = kernel_value
#         if self.use_bias:
#             store["1"] = self.bias
#         if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
#             store["2"] = kernel_scale

#     def load_own_variables(self, store):
#         if not self.lora_enabled:
#             self._check_load_own_variables(store)
#         # Do nothing if the layer isn't yet built
#         if not self.built:
#             return
#         # The keys of the `store` will be saved as determined because the
#         # default ordering will change after quantization
#         self._kernel.assign(store["0"])
#         if self.use_bias:
#             self.bias.assign(store["1"])
#         if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
#             self.kernel_scale.assign(store["2"])
#         if self.lora_enabled:
#             self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
#             self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

#     def get_config(self):
#         base_config = super().get_config()
#         config = {
#             "units": self.units,
#             "activation": activations.serialize(self.activation),
#             "use_bias": self.use_bias,
#             "kernel_initializer": initializers.serialize(
#                 self.kernel_initializer
#             ),
#             "bias_initializer": initializers.serialize(self.bias_initializer),
#             "kernel_regularizer": regularizers.serialize(
#                 self.kernel_regularizer
#             ),
#             "bias_regularizer": regularizers.serialize(self.bias_regularizer),
#             "kernel_constraint": constraints.serialize(self.kernel_constraint),
#             "bias_constraint": constraints.serialize(self.bias_constraint),
#         }
#         if self.lora_rank:
#             config["lora_rank"] = self.lora_rank
#         return {**base_config, **config}

#     def _check_load_own_variables(self, store):
#         all_vars = self._trainable_variables + self._non_trainable_variables
#         if len(store.keys()) != len(all_vars):
#             if len(all_vars) == 0 and not self.built:
#                 raise ValueError(
#                     f"Layer '{self.name}' was never built "
#                     "and thus it doesn't have any variables. "
#                     f"However the weights file lists {len(store.keys())} "
#                     "variables for this layer.\n"
#                     "In most cases, this error indicates that either:\n\n"
#                     "1. The layer is owned by a parent layer that "
#                     "implements a `build()` method, but calling the "
#                     "parent's `build()` method did NOT create the state of "
#                     f"the child layer '{self.name}'. A `build()` method "
#                     "must create ALL state for the layer, including "
#                     "the state of any children layers.\n\n"
#                     "2. You need to implement "
#                     "the `def build_from_config(self, config)` method "
#                     f"on layer '{self.name}', to specify how to rebuild "
#                     "it during loading. "
#                     "In this case, you might also want to implement the "
#                     "method that generates the build config at saving time, "
#                     "`def get_build_config(self)`. "
#                     "The method `build_from_config()` is meant "
#                     "to create the state "
#                     "of the layer (i.e. its variables) upon deserialization.",
#                 )
#             raise ValueError(
#                 f"Layer '{self.name}' expected {len(all_vars)} variables, "
#                 "but received "
#                 f"{len(store.keys())} variables during loading. "
#                 f"Expected: {[v.name for v in all_vars]}"
#             )

#     """Quantization-related methods"""

#     def quantized_build(self, input_shape, mode):
#         input_dim = input_shape[-1]
#         if mode == "int8":
#             self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
#             self._kernel = self.add_weight(
#                 name="kernel",
#                 shape=(input_dim, self.units),
#                 initializer="zeros",
#                 dtype="int8",
#                 trainable=False,
#             )
#             self.kernel_scale = self.add_weight(
#                 name="kernel_scale",
#                 shape=(self.units,),
#                 initializer="ones",
#                 trainable=False,
#             )

#     def quantized_call(self, inputs):
#         @ops.custom_gradient
#         def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):
#             def grad_fn(*args, upstream=None):
#                 if upstream is None:
#                     (upstream,) = args
#                 float_kernel = ops.divide(
#                     ops.cast(kernel, dtype=self.compute_dtype),
#                     kernel_scale,
#                 )
#                 inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
#                 return (inputs_grad, None, None)

#             inputs, inputs_scale = self.inputs_quantizer(inputs)
#             x = ops.matmul(inputs, kernel)
#             # De-scale outputs
#             x = ops.cast(x, self.compute_dtype)
#             x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
#             return x, grad_fn

#         x = matmul_with_inputs_gradient(
#             inputs,
#             ops.convert_to_tensor(self._kernel),
#             ops.convert_to_tensor(self.kernel_scale),
#         )
#         if self.lora_enabled:
#             lora_x = ops.matmul(inputs, self.lora_kernel_a)
#             lora_x = ops.matmul(lora_x, self.lora_kernel_b)
#             x = ops.add(x, lora_x)
#         if self.bias is not None:
#             x = ops.add(x, self.bias)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x

#     def quantize(self, mode):
#         import gc

#         # Prevent quantization of the subclasses
#         if type(self) is not CustomKernelDense:
#             raise NotImplementedError(
#                 f"Layer {self.__class__.__name__} does not have a `quantize()` "
#                 "method implemented."
#             )
#         self._check_quantize_args(mode, self.compute_dtype)
#         if mode == "int8":
#             if backend.standardize_dtype(self._kernel.dtype) == "int8":
#                 raise ValueError("`quantize` can only be done once per layer.")
#             # Configure `self.inputs_quantizer`
#             self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
#             # Quantize `self._kernel` to int8 and compute corresponding scale
#             kernel_value, kernel_scale = quantizers.abs_max_quantize(
#                 self._kernel, axis=0
#             )
#             kernel_scale = ops.squeeze(kernel_scale, axis=0)
#             self._tracker.unlock()
#             self._untrack_variable(self._kernel)
#             kernel_shape = self._kernel.shape
#             del self._kernel
#             self._kernel = self.add_weight(
#                 name="kernel",
#                 shape=kernel_shape,
#                 # Prevent adding a large constant to the computation graph
#                 initializer=lambda shape, dtype: kernel_value,
#                 dtype="int8",
#                 trainable=False,
#             )
#             self.kernel_scale = self.add_weight(
#                 name="kernel_scale",
#                 shape=(self.units,),
#                 # Prevent adding a large constant to the computation graph
#                 initializer=lambda shape, dtype: kernel_scale,
#                 trainable=False,
#             )
#             self._tracker.lock()
#         else:
#             NotImplementedError(
#                 "Invalid quantization mode. Expected 'int8'. "
#                 f"Received: mode={mode}"
#             )

#         # Set new dtype policy
#         if not isinstance(
#             self.dtype_policy, dtype_policies.QuantizedDTypePolicy
#         ):
#             quantized_dtype = f"{mode}_from_{self.dtype_policy.name}"
#             self.dtype_policy = dtype_policies.get(quantized_dtype)

#         # Release memory manually because sometimes the backend doesn't
#         gc.collect()

    # def _get_kernel_with_merged_lora(self):
    #     if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
    #         kernel_value = self._kernel
    #         kernel_scale = self.kernel_scale
    #         if self.lora_enabled:
    #             # Dequantize & quantize to merge lora weights into int8 kernel
    #             # Note that this is a lossy compression
    #             kernel_value = ops.divide(kernel_value, kernel_scale)
    #             kernel_value = ops.add(
    #                 kernel_value,
    #                 ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
    #             )
    #             kernel_value, kernel_scale = quantizers.abs_max_quantize(
    #                 kernel_value, axis=0
    #             )
    #             kernel_scale = ops.squeeze(kernel_scale, axis=0)
    #         return kernel_value, kernel_scale
    #     return self.kernel, None
class CustomKernelDense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors). The output in this case will have
    shape `(batch_size, d0, units)`.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `Dense` layer by calling `layer.enable_lora(rank)`.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        gamma = 1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.gamma = gamma
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
            self.quantized_build(
                input_shape, mode=self.dtype_policy.quantization_mode
            )
        else:
            self._kernel = self.add_weight(
                name="kernel",
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel
    
    def kernelFunction(self, inputs):
        inputs_exp = tf.expand_dims(inputs, -1)
        kernel_exp = tf.expand_dims(self.kernel, 0)
        distance = tf.norm(inputs_exp - kernel_exp, axis=1)
        return tf.exp(-self.gamma * tf.square(distance))
    
    def call(self, inputs):
        x = self.kernelFunction(inputs)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(self.kernel.shape[0], rank),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel.shape[1]),
            initializer=initializers.get(b_initializer),
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        kernel_value, kernel_scale = self._get_kernel_with_merged_lora()
        store["0"] = kernel_value
        if self.use_bias:
            store["1"] = self.bias
        if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
            store["2"] = kernel_scale

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        self._kernel.assign(store["0"])
        if self.use_bias:
            self.bias.assign(store["1"])
        if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
            self.kernel_scale.assign(store["2"])
        if self.lora_enabled:
            self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
            self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return {**base_config, **config}

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )

    """Quantization-related methods"""

    def quantized_build(self, input_shape, mode):
        input_dim = input_shape[-1]
        if mode == "int8":
            self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
            self._kernel = self.add_weight(
                name="kernel",
                shape=(input_dim, self.units),
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            self.kernel_scale = self.add_weight(
                name="kernel_scale",
                shape=(self.units,),
                initializer="ones",
                trainable=False,
            )

    def quantized_call(self, inputs):
        @ops.custom_gradient
        def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):
            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=self.compute_dtype),
                    kernel_scale,
                )
                inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
                return (inputs_grad, None, None)

            inputs, inputs_scale = self.inputs_quantizer(inputs)
            x = ops.matmul(inputs, kernel)
            # De-scale outputs
            x = ops.cast(x, self.compute_dtype)
            x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
            return x, grad_fn

        x = matmul_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(self._kernel),
            ops.convert_to_tensor(self.kernel_scale),
        )
        if self.lora_enabled:
            lora_x = ops.matmul(inputs, self.lora_kernel_a)
            lora_x = ops.matmul(lora_x, self.lora_kernel_b)
            x = ops.add(x, lora_x)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def quantize(self, mode):
        import gc

        # Prevent quantization of the subclasses
        if type(self) is not CustomKernelDense:
            raise NotImplementedError(
                f"Layer {self.__class__.__name__} does not have a `quantize()` "
                "method implemented."
            )
        self._check_quantize_args(mode, self.compute_dtype)
        if mode == "int8":
            if backend.standardize_dtype(self._kernel.dtype) == "int8":
                raise ValueError("`quantize` can only be done once per layer.")
            # Configure `self.inputs_quantizer`
            self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
            # Quantize `self._kernel` to int8 and compute corresponding scale
            kernel_value, kernel_scale = quantizers.abs_max_quantize(
                self._kernel, axis=0
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            self._tracker.unlock()
            self._untrack_variable(self._kernel)
            kernel_shape = self._kernel.shape
            del self._kernel
            self._kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                # Prevent adding a large constant to the computation graph
                initializer=lambda shape, dtype: kernel_value,
                dtype="int8",
                trainable=False,
            )
            self.kernel_scale = self.add_weight(
                name="kernel_scale",
                shape=(self.units,),
                # Prevent adding a large constant to the computation graph
                initializer=lambda shape, dtype: kernel_scale,
                trainable=False,
            )
            self._tracker.lock()
        else:
            NotImplementedError(
                "Invalid quantization mode. Expected 'int8'. "
                f"Received: mode={mode}"
            )

        # Set new dtype policy
        if not isinstance(
            self.dtype_policy, dtype_policies.QuantizedDTypePolicy
        ):
            quantized_dtype = f"{mode}_from_{self.dtype_policy.name}"
            self.dtype_policy = dtype_policies.get(quantized_dtype)

        # Release memory manually because sometimes the backend doesn't
        gc.collect()

    def _get_kernel_with_merged_lora(self):
        if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
            kernel_value = self._kernel
            kernel_scale = self.kernel_scale
            if self.lora_enabled:
                # Dequantize & quantize to merge lora weights into int8 kernel
                # Note that this is a lossy compression
                kernel_value = ops.divide(kernel_value, kernel_scale)
                kernel_value = ops.add(
                    kernel_value,
                    ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
                )
                kernel_value, kernel_scale = quantizers.abs_max_quantize(
                    kernel_value, axis=0
                )
                kernel_scale = ops.squeeze(kernel_scale, axis=0)
            return kernel_value, kernel_scale
        return self.kernel, None
