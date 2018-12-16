import tensorflow as tf


class GAN:
    def __init__(self, num_features, mode='train'):
        self.num_features = num_features
        self.input_shape = [None, num_features, None]  # [batch_size, num_features, num_frames]

    def build_model(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass