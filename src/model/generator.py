import tensorflow as tf
from layers import Residual1DBlock, Upsample1DBlock, Downsample1DBlock
import numpy as np


class Generator:
    """
    Gated CNN generator network
    """

    def forward(self, inputs, reuse=False, scope_name='gated_cnn_generator'):

        inputs = tf.transpose(inputs, perm=[0, 2, 1], name='transposed_input')

        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            h1 = tf.layers.conv1d(inputs=inputs, filters=128, kernel_size=15, strides=1, padding='same',
                                  activation=None, kernel_initializer=None, name='h1_conv')

            h1_gates = tf.layers.conv1d(inputs=inputs, filters=128, kernel_size=15, strides=1, padding='same',
                                        activation=None, kernel_initializer=None, name='h1_gates')

            h1_glu = tf.multiply(x=h1, y=tf.sigmoid(h1_gates), name='h1_glu')

            # Downsample
            d1 = Downsample1DBlock(
                filters=256, kernel_size=5, strides=2, name_prefix="downsample1D_block_1_").forward(inputs=h1_glu)
            d2 = Downsample1DBlock(
                filters=512, kernel_size=5, strides=2, name_prefix="downsample1D_block_2_").forward(inputs=d1)

            # Residual Blocks
            r1 = Residual1DBlock(filters=1024, kernel_size=3, strides=1,
                                 name_prefix='residual1d_block1_').forward(inputs=d2)

            r2 = Residual1DBlock(filters=1024, kernel_size=3, strides=1,
                                 name_prefix='residual1d_block2_').forward(inputs=r1)
            r3 = Residual1DBlock(filters=1024, kernel_size=3, strides=1,
                                 name_prefix='residual1d_block3_').forward(inputs=r2)
            r4 = Residual1DBlock(filters=1024, kernel_size=3, strides=1,
                                 name_prefix='residual1d_block4_').forward(inputs=r3)
            r5 = Residual1DBlock(filters=1024, kernel_size=3, strides=1,
                                 name_prefix='residual1d_block5_').forward(inputs=r4)
            r6 = Residual1DBlock(filters=1024, kernel_size=3, strides=1,
                                 name_prefix='residual1d_block6_').forward(inputs=r5)
            # Upsample
            u1 = Upsample1DBlock(filters=1024, kernel_size=15, strides=1, shuffle_size=2,
                                 name_prefix='upsample1d_block1_').forward(inputs=r6)
            u2 = Upsample1DBlock(filters=512, kernel_size=5, strides=1, shuffle_size=2,
                                 name_prefix='upsample1d_block2_').forward(inputs=u1)

            # Compute output
            out1 = tf.layers.conv1d(
                inputs=u2, filters=24, kernel_size=15, strides=1, activation=None, padding='same', name='out1_conv')

            out2 = tf.transpose(out1, perm=[0, 2, 1], name='transposed_output')

            return out2


if __name__ == '__main__':
    tf.reset_default_graph()
    array = np.random.randn(1, 24, 128)
    print("initial -> ", array.shape)
    generator = Generator().forward(array, reuse=False, scope_name="gen1")
    generator = Generator().forward(array, reuse=False, scope_name="gen2")
