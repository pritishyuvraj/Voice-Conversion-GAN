import tensorflow as tf


class PixelShuffler:
    def __init__(self, shuffle_size=2, name=None):
        self.shuffle_size = shuffle_size
        self.name = name

    def forward(self, inputs):
        n = tf.shape(inputs)[0]
        w = tf.shape(inputs)[1]
        c = inputs.get_shape().as_list()[2]

        out_c = c // self.shuffle_size
        out_w = w * self.shuffle_size

        outputs = tf.reshape(tensor=inputs, shape=[n, out_w, out_c], name=self.name)

        return outputs


class Residual1DBlock:
    """
    Creates a residual convolutional block for the model generator
    """

    def __init__(self, filters, kernel_size, strides, name_prefix, activation=None, kernel_init=None):
        """
        :param filters: Defines number of out channels
        :param kernel_size: Defines the dimensions of a filter
        :param strides: Stride to be used in CNN layers
        :param activation: Activation function to be used by the CNN layers
        :param kernel_init: Initialization function to be used to initialize filters for CNN layers
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = 'same'
        self.name_prefix = name_prefix
        self.activation = activation
        self.kernel_init = kernel_init
        self.epsilon = 1e-06

    def forward(self, inputs):
        """
        Builds the forward pass graph for the residual1D_block
        """
        h1 = tf.layers.conv1d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                              strides=self.strides, padding=self.padding, activation=self.activation,
                              kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_conv')

        h1_norm = tf.contrib.layers.instance_norm(inputs=h1, epsilon=self.epsilon, activation_fn=None,
                                                  name=self.name_prefix + 'h1_norm')

        h1_gates = tf.layers.conv1d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                                    strides=self.strides, padding=self.padding, activation=self.activation,
                                    kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_gates')

        h1_norm_gates = tf.contrib.layers.instance_norm(inputs=h1_gates, epsilon=self.epsilon, activation_fn=None,
                                                        name=self.name_prefix + 'h1_norm_gates')

        h1_glu = tf.multiply(x=h1_norm, y=tf.sigmoid(h1_norm_gates), name=self.name_prefix + 'h1_glu')

        h2 = tf.layers.conv1d(inputs=h1_glu, filters=self.filters, kernel_size=self.kernel_size,
                              strides=self.strides, padding=self.padding, activation=self.activation,
                              kernel_initializer=self.kernel_init, name=self.name_prefix + 'h2_conv')

        h2_norm = tf.contrib.layers.instance_norm(inputs=h2, epsilon=self.epsilon, activation_fn=None,
                                                  name=self.name_prefix + 'h2_norm_gates')

        h3 = inputs + h2_norm

        return h3


class Downsample1DBlock:
    """
    Creates a downsample 1d block for the model generator
    """
    def __init__(self, filters, kernel_size, strides, activation=None, kernel_init=None):
        """
        :param filters: Defines number of out channels
        :param kernel_size: Defines the dimensions of a filter
        :param strides: Stride to be used in CNN layers
        :param activation: Activation function to be used by the CNN layers
        :param kernel_init: Initialization function to be used to initialize filters for CNN layers
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = 'same'
        self.name_prefix = 'downsample1D_block_'
        self.activation = activation
        self.kernel_init = kernel_init
        self.epsilon = 1e-06

    def forward(self, inputs):
        """
        Builds the forward pass graph for the downsample1D_block
        """

        h1 = tf.layers.conv1d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                              strides=self.strides, padding=self.padding, activation=self.activation,
                              kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_conv')

        h1_norm = tf.contrib.layers.instance_norm(inputs=h1, epsilon=self.epsilon, activation_fn=None,
                                                  name=self.name_prefix + 'h1_norm')

        h1_gates = tf.layers.conv1d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                                    strides=self.strides, padding=self.padding, activation=self.activation,
                                    kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_gates')

        h1_norm_gates = tf.contrib.layers.instance_norm(inputs=h1_gates, epsilon=self.epsilon, activation_fn=None,
                                                        name=self.name_prefix + 'h1_norm_gates')

        h1_glu = tf.multiply(x=h1_norm, y=tf.sigmoid(h1_norm_gates), name=self.name_prefix + 'h1_glu')

        return h1_glu


class Downsample2DBlock:
    """
    Creates a downsample 2d block for the model generator
    """
    def __init__(self, filters, kernel_size, strides, name_prefix, activation=None, kernel_init=None):
        """
        :param filters: Defines number of out channels
        :param kernel_size: Defines the dimensions of a filter
        :param strides: Stride to be used in CNN layers
        :param activation: Activation function to be used by the CNN layers
        :param kernel_init: Initialization function to be used to initialize filters for CNN layers
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = 'same'
        self.name_prefix = name_prefix
        self.activation = activation
        self.kernel_init = kernel_init
        self.epsilon = 1e-06

    def forward(self, inputs):
        """
        Builds the forward pass graph for the downsample2D_block
        """

        h1 = tf.layers.conv2d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                              strides=self.strides, padding=self.padding, activation=self.activation,
                              kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_conv')

        h1_norm = tf.contrib.layers.instance_norm(inputs=h1, epsilon=self.epsilon, activation_fn=None,
                                                  name=self.name_prefix + 'h1_norm')

        h1_gates = tf.layers.conv2d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                                    strides=self.strides, padding=self.padding, activation=self.activation,
                                    kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_gates')

        h1_norm_gates = tf.contrib.layers.instance_norm(inputs=h1_gates, epsilon=self.epsilon, activation_fn=None,
                                                        name=self.name_prefix + 'h1_norm_gates')

        h1_glu = tf.multiply(x=h1_norm, y=tf.sigmoid(h1_norm_gates), name=self.name_prefix + 'h1_glu')

        return h1_glu


class Upsample1DBlock:
    """
    Creates a upsample 1d block for the model generator
    """
    def __init__(self, filters, kernel_size, strides, name_prefix, shuffle_size=2, activation=None, kernel_init=None):
        """
        :param filters: Defines number of out channels
        :param kernel_size: Defines the dimensions of a filter
        :param strides: Stride to be used in CNN layers
        :param activation: Activation function to be used by the CNN layers
        :param kernel_init: Initialization function to be used to initialize filters for CNN layers
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = 'same'
        self.name_prefix = name_prefix
        self.activation = activation
        self.kernel_init = kernel_init
        self.epsilon = 1e-06
        self.shuffle_size = shuffle_size

    def forward(self, inputs):
        """
        Builds the forward pass graph for the upsample1D_block
        """
        h1 = tf.layers.conv1d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                              strides=self.strides, padding=self.padding, activation=self.activation,
                              kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_conv')

        h1_shuffle = PixelShuffler(shuffle_size=self.shuffle_size,
                                   name=self.name_prefix + 'h1_shuffle').forward(inputs=h1)

        h1_norm = tf.contrib.layers.instance_norm(inputs=h1_shuffle, epsilon=self.epsilon, activation_fn=None,
                                                  name=self.name_prefix + 'h1_norm')

        h1_gates = tf.layers.conv1d(inputs=inputs, filters=self.filters, kernel_size=self.kernel_size,
                                    strides=self.strides, padding=self.padding, activation=self.activation,
                                    kernel_initializer=self.kernel_init, name=self.name_prefix + 'h1_gates')

        h1_shuffle_gates = PixelShuffler(shuffle_size=self.shuffle_size,
                                         name=self.name_prefix + 'h1_shuffle_gates').forward(inputs=h1_gates)

        h1_norm_gates = tf.contrib.layers.instance_norm(inputs=h1_shuffle_gates, epsilon=self.epsilon,
                                                        activation_fn=None, name=self.name_prefix + 'h1_norm_gates')

        h1_glu = tf.multiply(x=h1_norm, y=tf.sigmoid(h1_norm_gates), name=self.name_prefix + 'h1_glu')

        return h1_glu
