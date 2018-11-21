import os
import numpy as np
import argparse
import torch
import time
import pickle
from trainingDataset import trainingDataset


class CycleGANTraining:
    def __init__(self, logf0s_normalization, mcep_normalization, coded_sps_A_norm, coded_sps_B_norm):
        self.num_epochs = 5000
        self.mini_batch_size = 1
        self.dataset_A = self.loadPickleFile(coded_sps_A_norm)
        self.dataset_B = self.loadPickleFile(coded_sps_B_norm)

    def train(self):

        for epoch in range(self.num_epochs):
            start_time_epoch = time.time()

            dataset = trainingDataset(datasetA=self.dataset_A,
                                      datasetB=self.dataset_B,
                                      n_frames=128)
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.mini_batch_size,
                                                       shuffle=True,
                                                       drop_last=False)
            for i, (dataset_A, dataset_B) in enumerate(train_loader):
                print("Iteration {} datasetA: {}, datasetB: {}".format(
                      i, dataset_A.shape, dataset_B.shape))

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset")

    logf0s_normalization_default = '../cache/logf0s_normalization.npz'
    mcep_normalization_default = '../cache/mcep_normalization.npz'
    coded_sps_A_norm = '../cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm = '../cache/coded_sps_B_norm.pickle'

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm)
    parser.add_argument('--coded_sps_B_norm', type=str,
                        help="mcep norm for data B", default=coded_sps_B_norm)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    coded_sps_A_norm = argv.coded_sps_A_norm
    coded_sps_B_norm = argv.coded_sps_B_norm

    # Check whether following cached files exists
    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print(
            "Cached files do not exist, please run the program preprocess_training.py first")

    cycleGAN = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                coded_sps_A_norm=coded_sps_A_norm,
                                coded_sps_B_norm=coded_sps_B_norm)
    cycleGAN.train()
