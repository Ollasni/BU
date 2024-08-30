"""Implements dataloaders for AFFECT data."""
import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
from numpy.core.numeric import zeros_like
from torch.nn.functional import pad
from torch.nn import functional as F

sys.path.append(os.getcwd())

import torch
import torchtext as text
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# from text_robust import add_text_noise
# from timeseries_robust import add_timeseries_noise
import random
np.seterr(divide='ignore', invalid='ignore')



def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


def get_rawtext(path, data_kind, vids):
    """Get raw text, video data from hdf5 file."""
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


def _get_word2id(text_data, vids):
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['unk']
    data_processed = dict()
    for i, segment in enumerate(text_data):
        words = []
        _words = segment.split()
        for word in _words:
            words.append(word2id[word])
        words = np.asarray(words)
        data_processed[vids[i]] = words

    def _return_unk():
        return UNK

    word2id.default_factory = _return_unk
    return data_processed, word2id


def _get_word_embeddings(word2id, save=False):
    vec = text.vocab.GloVe(name='840B', dim=300)
    tokens = []
    for w, _ in word2id.items():
        tokens.append(w)

    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
    return ret


def _glove_embeddings(text_data, vids, paddings=50):
    data_prod, w2id = _get_word2id(text_data, vids)
    word_embeddings_looks_up = _get_word_embeddings(w2id)
    looks_up = word_embeddings_looks_up.numpy()
    embedd_data = []
    for vid in vids:
        d = data_prod[vid]
        tmp = []
        look_up = [looks_up[x] for x in d]
        # Padding with zeros at the front
        # TODO: fix some segs have more than 50 words (FIXed)
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings - len(d)):
                tmp.append(np.zeros(300, ))
            for x in d:
                tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:

        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


class Affectdataset(Dataset):
    """Implements Affect data as a torch dataset."""

    def __init__(self, config, data: Dict,
                 flatten_time_series: bool,
                 aligned: bool = True,
                 task: str = None,
                 max_pad=False,
                 max_pad_num=50,
                 data_type='mosi',
                 z_norm=False) -> None:
        """Instantiate AffectDataset

        Args:
            data (Dict): Data dictionary
            flatten_time_series (bool): Whether to flatten time series or not
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            task (str, optional): What task to load. Defaults to None.
            max_pad (bool, optional): Whether to pad data to max_pad_num or not. Defaults to False.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            data_type (str, optional): What data to load. Defaults to 'mosi'.
            z_norm (bool, optional): Whether to normalize data along the z-axis. Defaults to False.
        """
        self.config = config
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.data_type = data_type
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):
        """Get item from dataset."""
        # vision = torch.tensor(vision)
        # audio = torch.tensor(audio)
        # text = torch.tensor(text)


        if self.config.dataset.modalities.video.activate:
            vision = torch.tensor(self.dataset['vision'][ind])
        if self.config.dataset.modalities.audio.activate:
            audio = torch.tensor(self.dataset['audio'][ind])
        if self.config.dataset.modalities.text.activate:
            text = torch.tensor(self.dataset['text'][ind])

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
                # start = 0
            except:
                raise ValueError("No text data found, needed for alignment")

            if self.config.dataset.modalities.video.activate:
                vision = vision[start:].float()
            if self.config.dataset.modalities.audio.activate:
                audio = audio[start:].float()
            if self.config.dataset.modalities.text.activate:
                text = text[start:].float()
        else:
            if self.config.dataset.modalities.video.activate:
                zero_id = vision.nonzero()
                if len(zero_id) > 0:
                    vision = vision[zero_id[0][0]:].float()
                else:
                    vision = vision.float()
            if self.config.dataset.modalities.audio.activate:
                audio = audio[audio.nonzero()[0][0]:].float()
            if self.config.dataset.modalities.text.activate:
                text = text[text.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            if self.config.dataset.modalities.video.activate:
                vision = torch.nan_to_num(
                    (vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
            if self.config.dataset.modalities.audio.activate:
                audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
            if self.config.dataset.modalities.text.activate:
                text = torch.nan_to_num((text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        def _get_class(flag, data_type=self.data_type):
            if data_type in ['mosi', 'mosei', 'sarcasm', "MOSI"]:
                if flag > 0:
                    return [[1]]
                else:
                    return [[0]]
            else:
                return [flag]

        tmp_label = self.dataset['labels'][ind]
        if self.data_type == 'humor' or self.data_type == 'sarcasm':
            if (self.task == None) or (self.task == 'regression'):
                if self.dataset['labels'][ind] < 1:
                    tmp_label = [[-1]]
                else:
                    tmp_label = [[1]]
        else:
            tmp_label = self.dataset['labels'][ind]

        label = torch.tensor(_get_class(tmp_label)).long() if self.task == "classification" else torch.tensor(tmp_label).float()

        if self.flatten:
            data = {}
            if self.config.dataset.modalities.video.activate:
                data[1] = vision.flatten()
            if self.config.dataset.modalities.audio.activate:
                data[0] = audio.flatten()
            if self.config.dataset.modalities.text.activate:
                data[2] = text.flatten()

            return {"data":data, 'label': label}
            # return [vision.flatten(), audio.flatten(), text.flatten(), ind, \
            #         label]
        else:
            if self.max_pad:
                # tmp = [vision, audio, text, label]
                data = {}
                if self.config.dataset.modalities.video.activate:
                    data[1] = vision
                if self.config.dataset.modalities.audio.activate:
                    data[0] = audio
                if self.config.dataset.modalities.text.activate:
                    data[2] = text

                tmp = {"data":data, 'label': label}
                for i in tmp["data"]:
                    tmp["data"][i] = tmp["data"][i][:self.max_pad_num]
                    tmp["data"][i] = F.pad(tmp["data"][i], (0, 0, 0, self.max_pad_num - tmp["data"][i].shape[0]))
            else:
                data = {}
                if self.config.dataset.modalities.video.activate:
                    data[1] = vision
                if self.config.dataset.modalities.audio.activate:
                    data[0] = audio
                if self.config.dataset.modalities.text.activate:
                    data[2] = text

                tmp = {"data":data, 'label': label}
            return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]


def get_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, train_shuffle: bool = True,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, robust_test=False, data_type='mosi',
        raw_path='/home/van/backup/pack/mosi/mosi.hdf5', z_norm=False) -> DataLoader:
    """Get dataloaders for affect data.

    Args:
        filepath (str): Path to datafile
        batch_size (int, optional): Batch size. Defaults to 32.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 50.
        max_pad (bool, optional): Whether to pad data to max length or not. Defaults to False.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 2.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        task (str, optional): Which task to load in. Defaults to None.
        robust_test (bool, optional): Whether to apply robustness to data or not. Defaults to False.
        data_type (str, optional): What data to load in. Defaults to 'mosi'.
        raw_path (str, optional): Full path to data. Defaults to '/home/van/backup/pack/mosi/mosi.hdf5'.
        z_norm (bool, optional): Whether to normalize data along the z dimension or not. Defaults to False.

    Returns:
        DataLoader: tuple of train dataloader, validation dataloader, test dataloader
    """


    return train, valid, test

class CMU_MOSEI_Dataloader():

    def __init__(self, config):
        """
        :param config:
        """

        self.config = config


        process = eval("_process_2") if self.config.dataset.max_pad else eval("_process_1")


        dataset_train, dataset_val, dataset_test = self._get_datasets()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        # num_cores = multiprocessing.cpu_count()-2
        # num_cores = 16
        num_cores = 5
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=num_cores,
                                                        shuffle=True,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        generator=g,
                                                        collate_fn=process,
                                                        worker_init_fn=seed_worker)
        self.valid_loader = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_cores,
                                                        collate_fn=process,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=num_cores,
                                                       collate_fn=process,
                                                       pin_memory=self.config.training_params.pin_memory)

    def _get_datasets(self):

        # if file ends with hdf5, then we are using the raw data
        if self.config.dataset.data_path.endswith('hdf5'):
            #read the file
            with h5py.File(self.config.dataset.data_path, 'r') as f:
                #load from the file
                alldata = f['data']


        else:
            with open(self.config.dataset.data_path, "rb") as f:
                alldata = pickle.load(f)

        processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
        alldata['train'] = drop_entry(alldata['train'])
        alldata['valid'] = drop_entry(alldata['valid'])
        alldata['test'] = drop_entry(alldata['test'])

        for dataset in alldata:
            processed_dataset[dataset] = alldata[dataset]


        train_dataset = Affectdataset(self.config,
                                      processed_dataset['train'],
                                      self.config.dataset.flatten_time_series,
                                      aligned=self.config.dataset.aligned,
                                      task=self.config.dataset.task,
                                      max_pad=self.config.dataset.max_pad,
                                      max_pad_num=self.config.dataset.max_seq_len,
                                      data_type=self.config.dataset.data_type,
                                      z_norm=self.config.dataset.z_norm)
        valid_dataset = Affectdataset(self.config,
                                      processed_dataset['valid'],
                                      self.config.dataset.flatten_time_series,
                                      aligned=self.config.dataset.aligned,
                                      task=self.config.dataset.task,
                                      max_pad=self.config.dataset.max_pad,
                                      max_pad_num=self.config.dataset.max_seq_len,
                                      data_type=self.config.dataset.data_type,
                                      z_norm=self.config.dataset.z_norm)
        test_dataset = Affectdataset(self.config,
                                     processed_dataset['test'],
                                     self.config.dataset.flatten_time_series,
                                     aligned=self.config.dataset.aligned,
                                     task=self.config.dataset.task,
                                     max_pad=self.config.dataset.max_pad,
                                     max_pad_num=self.config.dataset.max_seq_len,
                                     data_type=self.config.dataset.data_type,
                                     z_norm=self.config.dataset.z_norm)

        return train_dataset, valid_dataset, test_dataset



def _process_1(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:

        inds.append(sample[-2])
        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
           torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)


def _process_2(inputs: List):

    processed_input = {0: [], 1: [], 2: []}
    # processed_input_lengths = {'vision': [], 'audio': [], 'text': []}
    for data_type in inputs[0]['data']:
        feature = []
        for sample in inputs:
            feature.append(sample['data'][data_type])
        # processed_input_lengths[data_type] = torch.as_tensor([v.size(0) for v in feature])
        processed_input[data_type] = torch.stack(feature)

    # for i in range(len(inputs[0]) - 1):
    #     feature = []
    #     for sample in inputs:
    #         feature.append(sample[i])
    #     processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
    #     # pad_seq = pad_sequence(feature, batch_first=True)
    #     processed_input.append(torch.stack(feature))

    labels = []
    for sample in inputs:
        if sample["label"].shape[1] > 1:
            labels.append(sample["label"].reshape(sample["label"].shape[1], sample["label"].shape[0])[0])
        else:
            labels.append(sample["label"].squeeze().unsqueeze(0))
    labels = torch.concatenate(labels, dim=0)


    return {"data": processed_input, "label": labels}


if __name__ == '__main__':



    dataloader = CMU_MOSEI_Dataloader()

    traindata, validdata, test_robust = \
        get_dataloader('/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CMU_MOSEI/mosei_senti_data.pkl',
                       # raw_path='/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CMU_MOSEI/mosei.hdf5',
                       robust_test=False,
                       max_pad=True,
                       task='classification',
                       data_type='mosei',
                       max_seq_len=40)

    # keys = list(test_robust.keys())

    # for batch in traindata:

    #     break
    print("Train: {}, Valid: {}, Test: {}, Total:{}".format(len(traindata.dataset), len(validdata.dataset), len(test_robust.dataset), len(traindata.dataset) + len(validdata.dataset)+ len(test_robust.dataset)))

    for batch in traindata:
        print(batch["data"]["video"].shape)
        print(batch["data"]["text"].shape)
        print(batch["data"]["audio"].shape)
        print(batch["label"].shape)
        print(batch["label"])
        break

    # test_robust[keys[0]][1]
    # for batch in test_robust:
    #     print(batch[-1].shape)
    #     break
        # for b in batch:

        # break