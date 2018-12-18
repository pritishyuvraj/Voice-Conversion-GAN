import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from losses import l1_loss, l2_loss
import numpy as np


class GAN:
    def __init__(self, num_features, mode='train'):
        self.num_features = num_features
        # [batch_size, num_features, num_frames]
        self.input_shape = [None, num_features, None]
        tf.reset_default_graph()
        self.build_model()
        self.optimizerIntializer()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        # Inputs
        self.input_A = tf.placeholder(
            tf.float32, shape=self.input_shape, name="input_A")
        self.input_B = tf.placeholder(
            tf.float32, shape=self.input_shape, name="input_B")

        # Generated Fake Samples
        self.fake_A = tf.placeholder(
            dtype=tf.float32, shape=self.input_shape, name="fake_A")
        self.fake_B = tf.placeholder(
            dtype=tf.float32, shape=self.input_shape, name="fake_B")

        # Generator
        self.generator_B = Generator().forward(inputs=self.input_A,
                                               reuse=False,
                                               scope_name="generator_A2B")
        self.cycle_A = Generator().forward(inputs=self.generator_B,
                                           reuse=False,
                                           scope_name="generator_B2A")

        self.generator_A = Generator().forward(inputs=self.input_B,
                                               reuse=True,
                                               scope_name="generator_B2A")
        self.cycle_B = Generator().forward(inputs=self.generator_A,
                                           reuse=True,
                                           scope_name="generator_A2B")

        self.generator_A_identity = Generator().forward(inputs=self.input_A,
                                                        reuse=True,
                                                        scope_name="generator_B2A")
        self.generator_B_identity = Generator().forward(inputs=self.input_B,
                                                        reuse=True,
                                                        scope_name="generator_A2B")

        self.discriminator_A_fake = Discriminator().forward(inputs=self.generator_A,
                                                            reuse=False,
                                                            scope_name="discriminator_A")
        self.discriminator_B_fake = Discriminator().forward(inputs=self.generator_B,
                                                            reuse=False,
                                                            scope_name="discriminator_B")
        print("Shapes are -> ", self.input_A.shape, self.cycle_A.shape,
              self.input_B.shape, self.cycle_B.shape, self.generator_A_identity.shape, self.generator_B_identity.shape)
        # Cycle Loss
        self.cycleLoss = l1_loss(y=self.input_A, y_hat=self.cycle_A) + \
            l1_loss(y=self.input_B, y_hat=self.cycle_B)

        # Idenity Loss
        self.identityLoss = l1_loss(y=self.input_A, y_hat=self.generator_A_identity) + \
            l1_loss(y=self.input_B, y_hat=self.generator_B_identity)

        self.lambdaCycle = tf.placeholder(dtype=tf.float32,
                                          shape=None,
                                          name='lambdaCycle')
        self.lambdaIndentity = tf.placeholder(dtype=tf.float32,
                                              shape=None,
                                              name='lambdaIndentity')

        # Generator Loss
        self.generatorLossA2B = l2_loss(y=tf.ones_like(
            self.discriminator_B_fake), y_hat=self.discriminator_B_fake)
        self.generatorLossB2A = l2_loss(y=tf.ones_like(
            self.discriminator_A_fake), y_hat=self.discriminator_A_fake)

        self.generatorLoss = self.generatorLossA2B + self.generatorLossB2A + \
            self.lambdaCycle * self.cycleLoss + self.identityLoss * self.lambdaIndentity

        # Discriminator Loss
        self.discriminatorInputA = Discriminator().forward(inputs=self.input_A,
                                                           reuse=True,
                                                           scope_name='discriminator_A')
        self.discriminatorInputB = Discriminator().forward(inputs=self.input_B,
                                                           reuse=True,
                                                           scope_name='discriminator_B')
        self.discriminatorFakeA = Discriminator().forward(inputs=self.fake_A,
                                                          reuse=True,
                                                          scope_name='discriminator_A')
        self.discriminatorFakeB = Discriminator().forward(inputs=self.fake_B,
                                                          reuse=True,
                                                          scope_name='discriminator_B')

        self.discriminatorLossInputA = l2_loss(y=tf.ones_like(
            self.discriminatorInputA), y_hat=self.discriminatorInputA)
        self.discriminatorLossFakeA = l2_loss(y=tf.zeros_like(
            self.discriminatorFakeA), y_hat=self.discriminatorFakeA)
        self.DiscriminatorLossA = (
            self.discriminatorLossInputA + self.discriminatorLossFakeA) / 2

        self.discriminatorLossInputB = l2_loss(y=tf.ones_like(
            self.discriminatorInputB), y_hat=self.discriminatorInputB)
        self.discriminatorLossFakeB = l2_loss(y=tf.zeros_like(
            self.discriminatorFakeB), y_hat=self.discriminatorFakeB)
        self.discriminatorLossB = (
            self.discriminatorLossInputB + self.discriminatorLossFakeB) / 2

        self.discriminatorLoss = self.DiscriminatorLossA + self.discriminatorLossB

        trainable_variables = tf.trainable_variables()
        self.discriminatorVars = [
            var for var in trainable_variables if 'discriminator' in var.name]
        self.generatorVars = [
            var for var in trainable_variables if 'generator' in var.name]
        # for var in trainable_variables:
        #     print(var.name)

    def optimizerIntializer(self):
        self.generatorLearningRate = tf.placeholder(
            tf.float32, None, name='generatorLearningRate')
        self.discriminatorLearningRate = tf.placeholder(
            tf.float32, None, name='discriminatorLearningRate')
        self.discriminatorOptimizer = tf.train.AdamOptimizer(learning_rate=self.discriminatorLearningRate,
                                                             beta1=0.5).minimize(self.discriminatorLoss)
        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.generatorLearningRate, beta1=0.5).minimize(self.generatorLoss)

    def train_model(self, input_A, input_B, lambdaCycle, lambdaIndentity, generatorLearningRate, discriminatorLearningRate):
        generatorA, generatorB, generatorLoss, _ = self.sess.run([self.generator_A,
                                                                  self.generator_B,
                                                                  self.generatorLoss,
                                                                  self.generator_optimizer],
                                                                 feed_dict={self.lambdaCycle: lambdaCycle,
                                                                            self.lambdaIndentity: lambdaIndentity,
                                                                            self.input_A: input_A,
                                                                            self.input_B: input_B,
                                                                            self.generatorLearningRate: generatorLearningRate})
        discriminatorLoss, _ = self.sess.run([self.discriminatorLoss,
                                              self.discriminatorOptimizer],
                                             feed_dict={self.input_A: input_A,
                                                        self.input_B: input_B,
                                                        self.discriminatorLearningRate: discriminatorLearningRate,
                                                        self.fake_A: generatorA,
                                                        self.fake_B: generatorB})
        return generatorLoss, discriminatorLoss


if __name__ == '__main__':
    cycleGAN = GAN(24)
    input_A = np.random.randn(1, 24, 128)
    input_B = np.random.randn(1, 24, 128)
    generatorLoss, discriminatorLoss = cycleGAN.train_model(input_A=input_A,
                                                            input_B=input_B,
                                                            lambdaCycle=10,
                                                            lambdaIndentity=5,
                                                            generatorLearningRate=0.0002,
                                                            discriminatorLearningRate=0.0001)
    print(generatorLoss, discriminatorLoss)
