import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict, List, Union
from utils import seed_worker
import random


___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

def get_in_the_wild_loader(seed:int,
                           config:dict,
                           dataset: str,
                           fewshot=False,
                           use_name=False,
                           shots=None):
    database_path = "/home/mikiya/data/in_the_wild/"

    import csv
    file = os.path.join(database_path, 'meta.csv')
    d_meta = {}
    file_list = []
    data_list0 = []
    data_list1 = []

    name_group = {}
    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            key, name, label = line
            # file_list.append(os.path.join(database_path,key))
            if label == 'bona-fide':
                data_list1.append(os.path.join(database_path,key))
                d_meta[os.path.join(database_path, key)] = 1
            else:
                data_list0.append(os.path.join(database_path, key))
                d_meta[os.path.join(database_path, key)] = 0
            if name not in name_group:
                name_group[name] = []
            name_group[name].append(os.path.join(database_path,key))

    if use_name:
        data_list0 = []
        data_list1 = []
        # names_list = ['Barack Obama', 'Donald Trump', 'Bill Clinton', 'JFK', 'Mark Zuckerberg']
        names_list = ['Donald Trump']
        for i in names_list:
            name_data = name_group[i]
            for j in name_data:
                if d_meta[j] == 1:
                    data_list1.append(j)
                else:
                    data_list0.append(j)



    if len(data_list0) <= len(data_list1):
        data_list1 = random.sample(data_list1,len(data_list0))
    else:
        data_list0 = random.sample(data_list0,len(data_list1))
    file_list = data_list0 + data_list1
    dataset = AudioDataset(list_IDs=file_list,
                               labels=d_meta,
                               base_dir=None,
                               transform=False)

    gen = torch.Generator()
    gen.manual_seed(seed)
    data_loader = DataLoader(dataset,
                             batch_size=config["batch_size"],
                             shuffle=True,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)
    print("no. in-the-wild files:", len(file_list))

    if fewshot:
        assert shots is not None
        trn_list = data_list0[:shots//2] + data_list1[:shots//2]
        eval_list = data_list0[-100:] + data_list1[-100:]
        trnset = AudioDataset(list_IDs=trn_list,
                                   labels=d_meta,
                                   base_dir=None,
                                   transform=False)

        gen = torch.Generator()
        gen.manual_seed(seed)
        trn_loader = DataLoader(trnset,
                                 batch_size=config["batch_size"],
                                 shuffle=True,
                                 drop_last=False,
                                 pin_memory=True,
                                 worker_init_fn=seed_worker,
                                 generator=gen)
        evalset = AudioDataset(list_IDs=eval_list,
                                  labels=d_meta,
                                  base_dir=None,
                                  transform=False)

        gen = torch.Generator()
        gen.manual_seed(seed)
        eval_loader = DataLoader(evalset,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=False,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)

        return trn_loader, eval_loader

    return data_loader


def get_custom_loader(seed: int,
                      config: dict,
                      dataset: str):


    if dataset == 'elevenlabs':
        database_path = './data/elevenlabs/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'openai':
        database_path = './data/openai/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'flashspeech':
        database_path = './data/FlashSpeech/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'voicebox':
        database_path = './data/VoiceBox/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'xtts':
        database_path = './data/xTTS/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'neuralspeech3':
        database_path = './data/NeuralSpeech3/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'valle':
        database_path = './data/VALLE/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'prompttts2':
        database_path = './data/PromptTTS2/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'audiogen':
        database_path = './data/AudioGen/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'audiobox':
        database_path = './data/AudioBox/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)



    dataset = AudioDataset(list_IDs=file_custom,
                               labels=d_label_custom,
                               base_dir=None,
                               transform=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    data_loader = DataLoader(dataset,
                             batch_size=config["batch_size"],
                             shuffle=True,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)
    print("no. custom files:", len(file_custom))

    return data_loader

def get_wavefake_loader(database_path: str,
                        seed: int,
                        config: dict,
                        dataset: str):
    # database_path = '/u/erdos/students/xli/data/wavefake/'
    d_trn_label, trn_file = genWavefake_list(database_path, is_train=True)
    trn_set = AudioDataset(list_IDs=trn_file,
                               labels=d_trn_label,
                               base_dir=None,
                               transform=True)
    print("no. wavefake train files:", len(trn_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(trn_set,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=gen)

    d_dev_label, dev_file = genWavefake_list(database_path)
    dev_set = AudioDataset(list_IDs=dev_file,
                                labels=d_dev_label,
                                base_dir=None,
                                transform=False)

    print("no. wavefake dev files:", len(dev_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    dev_loader = DataLoader(dev_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)

    d_eval_label, eval_file = genWavefake_list(database_path, is_eval=True)
    eval_set = AudioDataset(list_IDs=eval_file,
                                labels=d_eval_label,
                                base_dir=None,
                                transform=False)

    print("no. wavefake eval files:", len(eval_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)

    d_few_label, few_file = genWavefake_list(database_path, is_train=True, few_shot=True)
    few_set = AudioDataset(list_IDs=few_file,
                               labels=d_few_label,
                               base_dir=None,
                               transform=False)

    print("no. wavefake few-shot files:", len(few_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    few_loader = DataLoader(few_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    return train_loader, dev_loader, eval_loader, few_loader



def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    # d_label_trn, file_train = genSpoof_list(is_train=True,is_eval=False)
    d_label_trn, file_train = genLibriSeVoc_list(database_path, is_train=True, is_eval=False)
    print("no. training files:", len(file_train))

    train_set = AudioDataset(list_IDs=file_train,
                                 labels=d_label_trn,
                                 base_dir=None)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    # d_label_dev, file_dev = genSpoof_list(is_train=False, is_eval=False)
    d_label_dev, file_dev = genLibriSeVoc_list(database_path, is_train=False, is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = AudioDataset(list_IDs=file_dev,
                               labels=d_label_dev,
                               base_dir=None,
                               transform=False)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # d_label_eval, file_eval = genSpoof_list(is_train=False,is_eval=True)
    d_label_eval, file_eval = genLibriSeVoc_list(database_path, is_train=False, is_eval=True)
    print("no. test files:", len(file_eval))

    eval_set = AudioDataset(list_IDs=file_eval,
                                labels=d_label_eval,
                                base_dir=None,
                                transform=False)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def genLibriSeVoc_list(data_path, is_train=False, is_eval=False, budget=1.0):
    d_meta = {}
    data_list0 = []
    data_list1 = []
    data_list = []
    folders = ['diffwave',  'gt',  'melgan',  'parallel_wave_gan',  'wavegrad',  'wavenet',  'wavernn']
    if is_train:
        file = './train.txt'
    elif is_eval:
        file = './test.txt'
    else:
        file = './dev.txt'

    for folder in folders:
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if folder == 'gt':
                    key = os.path.join(data_path, folder, line)
                    data_list1.append(key)
                    d_meta[key] = 1
                else:
                    line = line.replace(".wav", "_gen.wav")
                    key = os.path.join(data_path, folder, line)
                    data_list0.append(key)
                    d_meta[key] = 0

    if is_train:
        random.shuffle(data_list0)
        data_list = data_list0[:int(budget*len(data_list0))] + data_list1[:int(budget*len(data_list1))]

    else:
        random.shuffle(data_list0)
        data_list0 = data_list0[:len(data_list1)]
        data_list = data_list0 + data_list1
        random.shuffle(data_list)

    return d_meta, data_list

def genWavefake_list(data_path, is_train=False, is_eval=False, few_shot=False, shots=500, budget=1.0):
    d_meta = {}
    data_list0 = []
    data_list1 = []

    ## Get wavefake
    folders = ['ljspeech_melgan',
               'ljspeech_parallel_wavegan',
               'ljspeech_multi_band_melgan',
               'ljspeech_full_band_melgan',
               'ljspeech_waveglow',
                'ljspeech_hifiGAN']

    for i in range(len(folders)):
        file_list = os.listdir(os.path.join(data_path, folders[i]))
        if is_train:
            file_list = file_list[:int(0.7*len(file_list))]
        elif is_eval:
            file_list = file_list[int(0.8*len(file_list)):]
        else:
            file_list = file_list[int(0.7*len(file_list)):int(0.8*len(file_list))]

        for line in file_list:
            key = os.path.join(data_path,folders[i],line)
            data_list0.append(key)
            d_meta[key] = 0

    # Get LJSpeech
    file_list = os.listdir('./data/LJSpeech-1.1/wavs/')
    if is_train:
        file_list = file_list[:int(0.7*len(file_list))]
    elif is_eval:
        file_list = file_list[int(0.8*len(file_list)):]
    else:
        file_list = file_list[int(0.7 * len(file_list)):int(0.8 * len(file_list))]

    for line in file_list:
        key = os.path.join('./data/LJSpeech-1.1/wavs/',line)
        data_list1.append(key)
        d_meta[key] = 1

    if is_train and few_shot:
        data_list0 = random.sample(data_list0, shots//2)
        data_list1 = random.sample(data_list1,shots//2)


    if is_train:
        random.shuffle(data_list0)
        data_list = data_list0[:int(budget*len(data_list0))] + data_list1[:int(budget*len(data_list1))]

    else:
        random.shuffle(data_list0)
        data_list0 = data_list0[:len(data_list1)]  # Sampling to match label 1 count
        # Combine and shuffle the final dataset
        data_list = data_list0 + data_list1
        random.shuffle(data_list)


    return d_meta, data_list

def genCustom_list(data_path, fake=True):
    d_meta = {}
    data_list = []
    file_list = os.listdir(data_path)
    for line in file_list:
        key = os.path.join(data_path, line)
        data_list.append(key)
        if fake:
            d_meta[key] = 0
        else:
            d_meta[key] = 1
    return d_meta, data_list

 

def add_gaussian_noise(waveform, noise_level=0.005, probability=0.5):
    """
    Add Gaussian noise to the waveform with a given probability.

    Parameters:
    - waveform: the original audio signal, a PyTorch tensor
    - noise_level: the standard deviation of the Gaussian noise
    - probability: the probability that noise will be added
    """
    if torch.rand(1).item() < probability:
        # Generate Gaussian noise
        noise = np.random.randn(waveform.size) * noise_level

        # Add noise to the waveform
        noisy_waveform = waveform + noise
        return noisy_waveform
    else:
        # Return the original waveform if not adding noise
        return waveform


def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 96000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class AudioDataset(Dataset):
    def __init__(self, list_IDs, labels, base_dir, transform=True):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 96000  # take ~4 sec audio (96000 samples) 24kHz sample rate
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = sf.read(str(key))
        X, _ = librosa.load(str(key), sr=24000)
        X = librosa.util.normalize(X)
        X_pad = pad_random(X, self.cut)
        if self.transform:
            X_pad = add_gaussian_noise(X_pad)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y, key

