from modrec_vgg import VGGNet, SaveBestParam, StopRestore, Score_ConfusionMatrix
import pickle

with open('result.pkl', 'rb') as f:
    res = pickle.load(f)

print(1)