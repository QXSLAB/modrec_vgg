from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
from skorch.callbacks import EpochScoring, PrintLog
from skorch.callbacks import BatchScoring
from skorch.utils import data_from_dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import norm
from dask.distributed import Client
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pickle
import copy
import math


warnings.filterwarnings("ignore")

mods = ['BPSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK',
        'PAM4', 'PAM8', 'PSK8', 'QAM16', 'QAM64', 'QPSK']
class_num = len(mods)


def import_from_mat(data, size):
    features = []
    labels = []
    for mod in mods:
        real = np.array(data[mod].real[:size])
        imag = np.array(data[mod].imag[:size])
        signal = np.concatenate([real, imag], axis=1)
        features.append(signal)
        labels.append(mods.index(mod) * np.ones([size, 1]))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def load_data():

    print("loading data")

    data = scipy.io.loadmat(
        "D:/batch100000_symbols128_sps8_baud1_snr5.dat",
    )
    features, labels = import_from_mat(data, 100000)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    X = features
    y = labels.reshape(-1)

    # class_num = 10
    # X, y = make_classification(100, 2048,
    #                            n_informative=5,
    #                            n_classes=class_num,
    #                            random_state=0)
    # X = X.astype(np.float32)
    # y = y.astype(np.int64)

    return X, y


class VGGNet(nn.Module):

    print("Define VGG Net")

    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),  # batch, 64, 1024
            nn.MaxPool1d(2),  # batch, 64, 512
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),  # batch, 64, 512
            nn.MaxPool1d(2),  # batch, 64, 256
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),  # batch, 64, 256
            nn.MaxPool1d(2),  # batch, 64, 128
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),  # batch, 64, 128
            nn.MaxPool1d(2),  # batch, 64, 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),  # batch, 64, 64
            nn.MaxPool1d(2),  # batch, 64, 32
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),  # batch, 64, 32
            nn.MaxPool1d(2),  # batch, 64, 16
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),  # batch, 64, 16
            nn.MaxPool1d(2),  # batch, 64, 8
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, class_num),
        )

    def forward(self, x, **kwargs):
        x = x.reshape((x.size(0), 2, -1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def forward1(self, x, **kwargs):
        x = x.reshape((x.size(0), 2, -1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class SaveBestParam(Checkpoint):

    """Save best model state"""

    def save_model(self, net):
        self.best_model_dict = copy.deepcopy(
            net.module_.state_dict()
        )
        self.best_confusion_matrix = copy.deepcopy(
            net.history[-1, "confusion_matrix"]
        )


class StopRestore(EarlyStopping):

    """Early Stop and Restore best module state"""

    def on_epoch_end(self, net, **kwargs):
        # super().on_epoch_end(net, **kwargs)
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
        if self.misses_ == self.patience:
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)

            best_cp = net.get_params()['callbacks__best']
            net.module_.load_state_dict(best_cp.best_model_dict)
            print("Best Model State Restored")
            print("Best Confusion Matrix:\n {0}".format(
                best_cp.best_confusion_matrix))

            raise KeyboardInterrupt


class Score_ConfusionMatrix(EpochScoring):
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        EpochScoring.on_epoch_end(self, net, dataset_train, dataset_valid)

        X_test, y_test = data_from_dataset(dataset_valid)
        y_pred = net.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sample_num = np.sum(cm, axis=1)
        sample_num = np.tile(sample_num, (sample_num.size, 1))
        cm = cm * (1/np.transpose(sample_num))
        cm = np.round(cm*100)
        cm1 = cm.astype(np.int8)
        history = net.history
        history.record("confusion_matrix", cm1)


class CheckNanInf(BatchScoring):

    def on_batch_end(self, net, X, y, training, **kwargs):
        if training:
            train_loss = net.history[-1, 'batches', -1, 'train_loss']
            print("train loss")
            print(train_loss)
            print("module forward")
            print(net.module_.forward(X.cuda())[0])
            print("module forward1")
            print(net.module_.forward1(X.cuda())[0])
            print("net forward")
            print(net.forward(X)[0])
            print("net predict prob")
            print(net.predict_proba(X)[0])
            print("\n")
            if math.isnan(train_loss):
                print(X.shape)

def train():

    disc = VGGNet()

    cp = SaveBestParam(dirname='best')
    early_stop = StopRestore(patience=10000)
    nan_batch = CheckNanInf('accuracy',
                            lower_is_better=False,
                            on_train=True)
    score = Score_ConfusionMatrix(scoring="accuracy", lower_is_better=False)
    pt = PrintLog(keys_ignored="confusion_matrix")
    net = NeuralNetClassifier(
        disc,
        max_epochs=10000,
        lr=0.01,
        device='cuda',
        callbacks=[('best', cp),
                   ('early', early_stop),
                   ('nan', nan_batch)],
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False
    )
    net.set_params(callbacks__valid_acc=score)
    net.set_params(callbacks__print_log=pt)

    X, y = load_data()
    net.fit(X, y)
    print(1)

    # param_dist = {
    #     'lr': [0.05, 0.01, 0.005],
    # }
    #
    # search = RandomizedSearchCV(net,
    #                             param_dist,
    #                             cv=StratifiedKFold(n_splits=3),
    #                             n_iter=3,
    #                             verbose=10,
    #                             scoring='accuracy')
    #
    # X, y = load_data()
    #
    # # search.fit(X, y)
    #
    # Client("127.0.0.1:8786")  # create local cluster
    #
    # with joblib.parallel_backend('dask'):
    #     search.fit(X, y)
    #
    # with open('result.pkl', 'wb') as f:
    #     pickle.dump(search, f)


if __name__ == "__main__":
    train()
