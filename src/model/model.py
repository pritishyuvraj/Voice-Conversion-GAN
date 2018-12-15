import tensorflow as tf


class GAN:
    def __init__(self, num_features, mode='train'):
        self.num_features = num_features
        self.input_shape = [None, num_features, None]  # [batch_size, num_features, num_frames]

    def build_model(self):
        '''
        Build tensorflow graph here sess.run() call is here
        Call helper function train.py to train the model
        :return:
        '''
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train_model(self):
        '''
        Model training here -- both train and validation data
        Definition of cycle loss is here rather than in losses.py
        :return:
        '''
        pass

    def evaluate_model(self):
        '''
        Prediction for test data
        :return:
        '''
        pass