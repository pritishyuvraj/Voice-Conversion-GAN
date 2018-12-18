import tensorflow as tf
from layers import Downsample2DBlock
import numpy as np


class Discriminator:
    """
    Discriminator network
    """

    def forward(self, inputs, reuse=False, scope_name='discriminator'):

        inputs = tf.expand_dims(inputs, -1)

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            h1 = tf.layers.conv2d(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], padding='same',
                                  activation=None, kernel_initializer=None, name='h1_conv')

            h1_gates = tf.layers.conv2d(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], padding='same',
                                        activation=None, kernel_initializer=None, name='h1_gates_conv')

            h1_glu = tf.multiply(x=h1, y=tf.sigmoid(h1_gates), name='h1_glu')

            # Downsample
            d1 = Downsample2DBlock(filters=256, kernel_size=[3, 3], strides=[2, 2],
                                   name_prefix='downsample2d_block1_').forward(inputs=h1_glu)
            d2 = Downsample2DBlock(filters=512, kernel_size=[3, 3], strides=[2, 2],
                                   name_prefix='downsample2d_block2_').forward(inputs=d1)
            d3 = Downsample2DBlock(filters=1024, kernel_size=[6, 3], strides=[1, 2],
                                   name_prefix='downsample2d_block3_').forward(inputs=d2)

            # Compute output
            out = tf.layers.dense(inputs=d3, units=1, activation=tf.nn.sigmoid)

            return out


if __name__ == '__main__':
    tf.reset_default_graph()
    array = np.random.randn(1, 24, 128)
    discriminator = Discriminator().forward(array, reuse=False, scope_name='disc1')
    discriminator1 = Discriminator().forward(
        array, reuse=False, scope_name='disc2')
    print(discriminator.shape)
