import tensorflow as tf


def l1_loss(y, y_hat):
    return tf.reduce_mean(tf.abs(y - y_hat))


def l2_loss(y, y_hat):
    return tf.reduce_mean(tf.square(y - y_hat))
