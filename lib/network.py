import tensorflow as tf


class Network:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.
    """

    @classmethod
    def build(cls, num_inputs=2, layers=None, activation='tanh', num_outputs=1):
        """
        Build a PINN model for Burgers' equation with input shape (t, x) and output shape u(t, x).

        Args:
            num_inputs: number of input variables. Default is 2 for (t, x).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outputs: number of output variables. Default is 1 for u(t, x).

        Returns:
            keras network model
        """

        # input layer
        if layers is None:
            layers = [32, 16, 32]
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                                      kernel_initializer='he_normal')(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
                                        kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
