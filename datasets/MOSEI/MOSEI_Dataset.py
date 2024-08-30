import os
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets.MOSEI.tokenizer import *



class Mosei_Loader(object):
    def __init__(self, args):
        self.args = args
        self.train_dataset = Mosei_Dataset('train', self.args, None)
        self.valid_dataset = Mosei_Dataset('valid', self.args, self.train_dataset.token_to_ix)
        self.test_dataset = Mosei_Dataset('test', self.args, self.train_dataset.token_to_ix)

    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=32,
                          pin_memory=True)

    @property
    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)

    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)


class Mosei_Dataset(Dataset):
    def __init__(self, mode, config, token_to_ix=None) -> None:
        super(Mosei_Dataset, self).__init__()
        assert mode in ["train", "valid", "test", "private"]
        self.args = config
        self.private_set = mode == "private"
        self.dataroot = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CMU-MOSEI/MOSEI"

        word_file = os.path.join(self.dataroot, mode + "_sentences.p")
        audio_file = os.path.join(self.dataroot, mode + "_mels.p")
        video_file = os.path.join(self.dataroot, mode + "_mels.p")
        # video_file = os.path.join(self.dataroot,name + "r21d.p")

        y_s_file = os.path.join(self.dataroot, mode + "_sentiment.p")
        y_e_file = os.path.join(self.dataroot, mode + "_emotion.p")

        self.set = eval(mode.upper() + "_SET")

        self.key_to_word = pickle.load(open(word_file, "rb"))
        self.key_to_audio = pickle.load(open(audio_file, "rb"))
        self.key_to_video = pickle.load(open(video_file, "rb"))

        # If private test,labels dont exist.
        if not self.private_set:
            if self.args.dataset.task == "emotion":
                self.key_to_label = pickle.load(open(y_e_file, "rb"))
            if self.args.dataset.task == "sentiment":
                self.key_to_label = pickle.load(open(y_s_file, "rb"))

            for key in self.set:
                if not (key in self.key_to_word and
                        key in self.key_to_audio and
                        key in self.key_to_video and
                        key in self.key_to_label):
                    self.set.remove(key)
        for key in self.set:
            Y = self.key_to_label[key]

        # Creating embeddings and word indexes
        self.key_to_sentence = tokenize(self.key_to_word)
        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else:
            self.token_to_ix, self.pretrained_emb = create_dict(self.key_to_sentence, self.dataroot)

        self.vocab_size = len(self.token_to_ix)

        self.l_max_len = self.args.dataset.lang_seq_len
        self.a_max_len = self.args.dataset.audio_seq_len
        self.v_max_len = self.args.dataset.video_seq_len

    def __getitem__(self, index):
        key = self.set[index]
        L = sent_to_ix(self.key_to_sentence[key], self.token_to_ix, max_token=self.l_max_len)
        A = pad_feature(self.key_to_audio[key], self.a_max_len)
        V = pad_feature(self.key_to_video[key], self.v_max_len)

        y = np.array([])
        if not self.private_set:
            Y = self.key_to_label[key]
            # print(Y)
            if self.args.dataset.task == "sentiment" and self.args.dataset.task_binary:
                c = cmumosei_2(Y)
                y = np.array(c)
            elif self.args.dataset.task == "sentiment" and not self.args.dataset.task_binary:
                c = cmumosei_7(Y)
                y = np.array(c)
            elif self.args.dataset.task == "emotion":
                Y[Y > 0] = 1
                y = Y

        return {"data":{0:torch.from_numpy(L), 1:torch.from_numpy(A), 2:torch.from_numpy(V).float()},"label": torch.from_numpy(y), "key":key}

    def __len__(self):
        return len(self.set)

class MOSEI_Dataloader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test = self._get_datasets()

        print("Train {}, Val {}, Test {}".format(len(sleep_dataset_train), len(sleep_dataset_val), len(sleep_dataset_test)))

        self.train_loader = torch.utils.data.DataLoader(sleep_dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id))
        self.valid_loader = torch.utils.data.DataLoader(sleep_dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(sleep_dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)
    def _get_datasets(self):

        train_dataset = Mosei_Dataset(config=self.config, mode="train")
        valid_dataset = Mosei_Dataset(config=self.config, mode="valid")
        test_dataset = Mosei_Dataset(config=self.config, mode="test")

        return train_dataset, valid_dataset, test_dataset


# import json
# from types import SimpleNamespace
#
# config = '{"dataset":{"data_root":"/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CMU-MOSEI/MOSEI", "task":"emotion", "lang_seq_len":60, "audio_seq_len":60, "video_seq_len":60}}'
# config = json.loads(config, object_hook=lambda d: SimpleNamespace(**d))
# print(config)
#
# # dataset = Mosei_Dataset('train', config, None)
# dataloader = MOSEI_Dataloader(config)
# print(len(dataloader.train_loader))
# print(len(dataloader.valid_loader))
# print(len(dataloader.test_loader))