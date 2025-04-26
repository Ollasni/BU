import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchaudio
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def collate_fn_padd(batch):
    aggregated_batch = {"data": {}, "label": [], "idx": []}

    # Сбор спектрограмм (0)
    specs = [d["data"][0] for d in batch if d["data"][0] is not False]
    if specs:
        max_freq = max(sp.shape[0] for sp in specs)
        max_time = max(sp.shape[1] for sp in specs)
        padded_specs = torch.stack([
            torch.nn.functional.pad(sp, (0, max_time - sp.shape[1], 0, max_freq - sp.shape[0]))
            for sp in specs
        ])
        aggregated_batch["data"][0] = padded_specs
    else:
        aggregated_batch["data"][0] = False

    # Сбор видео (1)
    vids = [d["data"][1].unsqueeze(0) for d in batch if d["data"][1] is not False]
    if vids:
        aggregated_batch["data"][1] = torch.cat(vids, dim=0)
    else:
        aggregated_batch["data"][1] = False

    aggregated_batch["label"] = torch.tensor([d["label"] for d in batch])
    aggregated_batch["idx"] = torch.tensor([d["idx"] for d in batch])
    return aggregated_batch


class CremadDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.active_modalities = {"video": True, "spectrogram": True}
        self.data = {"video": [], "audio": []}
        self.label = []
        self.mode = mode
        self.fps = config.dataset.get("fps", 1)
        self.config = config
        self.num_frame = config.dataset.get("num_frame", 3)
        self.norm_type = config.dataset.get("norm_type", "per_sample")
        self.sampling_rate = config.dataset.sampling_rate

        self.visual_feature_path = config.dataset.data_roots
        self.audio_feature_path = os.path.join(config.dataset.data_roots, "AudioWAV")

        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}
        self.train_csv = './datasets/CREMAD/train.csv'
        self.test_csv = './datasets/CREMAD/test.csv'

        def load_data(csv_file):
            items, labels, videos, audios = [], [], [], []
            with open(csv_file, encoding='UTF-8-sig') as f:
                for row in f:
                    file_id, label = row.strip().split(',')
                    video_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', file_id)
                    audio_path = os.path.join(self.audio_feature_path, file_id + '.wav')
                    if os.path.exists(video_path) and os.path.exists(audio_path):
                        videos.append(video_path)
                        audios.append(audio_path)
                        items.append(file_id)
                        labels.append(class_dict[label])
            return items, labels, videos, audios

        all_items, all_labels, all_videos, all_audios = load_data(self.train_csv)
        X = np.array([all_videos, all_audios, all_items]).T
        y = np.array(all_labels)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config.dataset.get("val_split_rate", 0.1),
            stratify=y,
            random_state=config.training_params.seed
        )

        if mode == "train":
            X_final, y_final = X_train, y_train
        else:
            X_final, y_final = X_val, y_val

        self.data["video"] = X_final[:, 0].tolist()
        self.data["audio"] = X_final[:, 1].tolist()
        self.label = y_final.tolist()

    def __len__(self):
        return len(self.data["video"])

    def _get_images(self, idx):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, antialias=True) if self.mode == 'train' else transforms.Resize((224, 224), antialias=True),
            transforms.RandomHorizontalFlip() if self.mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_samples = sorted(os.listdir(self.data["video"][idx]))
        images = torch.zeros((self.num_frame, 3, 224, 224))
        for i, img_file in enumerate(image_samples[:self.num_frame]):
            img = Image.open(os.path.join(self.data["video"][idx], img_file)).convert('RGB')
            images[i] = transform(img)
        return images.permute(1, 0, 2, 3)  # (3, num_frame, 224, 224)

    def _get_audio(self, idx):
        audio, sr = torchaudio.load(self.data["audio"][idx])
        return torchaudio.functional.resample(audio[0], sr, self.sampling_rate)

    def _get_spectrogram(self, idx, audio):
        resamples = audio.numpy()
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        if self.norm_type == "per_sample":
            spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-9)
        return torch.from_numpy(spectrogram)

    def __getitem__(self, idx):
        audio_data = self._get_audio(idx)
        data = {
            0: self._get_spectrogram(idx, audio_data),
            1: self._get_images(idx)
        }
        return {"data": data, "label": self.label[idx], "idx": idx}
