import copy
import csv
import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import json
from sklearn.model_selection import train_test_split
import multiprocessing
import torchaudio
from tqdm import tqdm
import logging
import librosa
import sys

def collate_fn_padd(batch):
    aggregated_batch = {}
    for key in batch[0].keys():
        if key == "data":
            aggregated_batch[key] = {}
            for subkey in batch[0][key].keys():
                # Собираем все тензоры данной "подмодальности" (subkey)
                aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]

                if aggregated_list:
                    # ----- Случай 1: Спектрограмма -----
                    # subkey == 0 (в вашем коде означает spectrogram)
                    if subkey == 0:
                        # Ищем максимальные размеры freq и time
                        max_freq = max(sp.shape[0] for sp in aggregated_list)
                        max_time = max(sp.shape[1] for sp in aggregated_list)

                        padded_spectros = []
                        for sp in aggregated_list:
                            freq, time = sp.shape
                            # Создаём тензор нулей [max_freq, max_time], куда копируем данные
                            pad_sp = torch.zeros((max_freq, max_time), dtype=sp.dtype)
                            pad_sp[:freq, :time] = sp
                            padded_spectros.append(pad_sp.unsqueeze(0))  
                            # теперь shape одного элемента → (1, max_freq, max_time)

                        # После паддинга склеиваем по batch размерности
                        aggregated_batch[key][subkey] = torch.cat(padded_spectros, dim=0)
                    
                    # ----- Случай 2: Видео -----
                    elif subkey == 1:
                        # Для видео обычно все кадры одного размера, поэтому .cat() напрямую
                        aggregated_batch[key][subkey] = torch.cat(
                            [d.unsqueeze(0) for d in aggregated_list], dim=0
                        )

                    # ----- Случай 3: Остальные (Audio=2, Face=3, Face_image=4) -----
                    else:
                        lengths = [len(d) for d in aggregated_list]
                        # Для Face/Face_image обрезаем до 150; для Audio - полную длину
                        max_length = min(max(lengths), 150) if subkey in [3, 4] else max(lengths)
                        aggregated_list = [d[:max_length] for d in aggregated_list]
                        padded = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)
                        aggregated_batch[key][subkey] = padded

                        # Добавляем attention mask
                        if subkey in [2, 3, 4]:  # Audio=2, Face=3, Face_image=4
                            mask = torch.zeros((len(aggregated_list), max_length))
                            for i, dur in enumerate(lengths):
                                mask[i, :min(dur, max_length)] = 1
                            aggregated_batch[key][f"attention_mask_{subkey}"] = mask

                else:
                    # Ни у одного элемента batch нет этой модальности → False
                    aggregated_batch[key][subkey] = False

        else:
            # Ключ "label" или "idx"
            aggregated_batch[key] = torch.LongTensor([d[key] for d in batch])
    return aggregated_batch

class CremadDataset(Dataset):

    def __init__(self, config, visual_path="/kaggle/input/image-01-fps",
                 audio_path="/kaggle/input/crema-d-emotional-multimodal-dataset/content/CREMA-D/AudioWAV", fps=1, mode='train'):
        # Смотрим, какие модальности включены в конфиге
        self.active_modalities = config.dataset.get(
            "return_data",
            {"video": True, "spectrogram": True, "audio": False, "face": False, "face_image": False}
        )
        
        # Инициализируем self.data с учётом того, что spectrogram может тоже требовать audio.
        self.data = {}
        if self.active_modalities.get("video", False):
            self.data["video"] = []
        if self.active_modalities.get("face", False):
            self.data["face"] = []
        if self.active_modalities.get("face_image", False):
            self.data["face_image"] = []
        # Если хотя бы одна из audio или spectrogram активна – нужен ключ "audio"
        if self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False):
            self.data["audio"] = []

        self.label = []
        self.mode = mode
        self.fps = fps
        self.config = config
        self.num_frame = self.config.dataset.get("num_frame", 3)
        data_split = self.config.dataset.get("data_split", {"a": 0})
        fold = data_split.get("fold", 0)
        self.norm_type = self.config.dataset.get("norm_type", False)
        self.sampling_rate = self.config.dataset.sampling_rate

        # Выводим активные модальности
        print("Используемые модальности:")
        for modality, enabled in self.active_modalities.items():
            if enabled:
                print(f"- {modality}")

        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        # Пути к данным
        self.visual_feature_path = self.config.dataset.data_roots
        self.audio_feature_path = os.path.join(self.config.dataset.data_roots, "AudioWAV")
        self.face_feature_path = os.path.join(self.config.dataset.data_roots, "Face_features")
        self.face_image_path = os.path.join(self.config.dataset.data_roots, "Face_features_images")

        # Проверка базовых директорий
        required_paths = {}
        if self.active_modalities.get("video", False):
            required_paths["visual"] = self.visual_feature_path
        # Даже если audio=False, но spectrogram=True – нам всё равно нужна директория с аудио
        if self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False):
            required_paths["audio"] = self.audio_feature_path
        if self.active_modalities.get("face", False):
            required_paths["face_features"] = self.face_feature_path
        if self.active_modalities.get("face_image", False):
            required_paths["face_images"] = self.face_image_path

        missing_paths = [name for name, path in required_paths.items() if not os.path.exists(path)]
        if missing_paths:
            print(f"Ошибка: следующие директории не найдены: {missing_paths}")
            sys.exit(1)

        # Выбор метода разделения
        if data_split.get("method", "inclusive") == "inclusive":
            self.split_inclusive(mode, class_dict)
        elif data_split.get("method", "inclusive") == "non_inclusive":
            self.split_noninclusive(fold, mode, class_dict)
        else:
            raise ValueError(
                "config.dataset.data_split должно быть 'inclusive' или 'non_inclusive', "
                f"получили: {self.config.dataset.get('data_split', 'inclusive')}"
            )

        # Проверка, что данные загружены
        if not any(len(v) > 0 for v in self.data.values()):
            print(f"Ошибка: не найдены данные для активных модальностей в режиме {mode}")
            sys.exit(1)

        # Нормализация аудио (если audio или spectrogram включены)
        if (self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False)):
            if self.config.dataset.get("norm_wav_path", None) and os.path.exists(self.config.dataset.get("norm_wav_path", None)):
                self.wav_norm = pickle.loads(open(self.config.dataset.norm_wav_path, "rb").read())
                logging.info(f"Loaded wav norm from {self.config.dataset.norm_wav_path}")
                logging.info(f"Norm values are {self.wav_norm}")
            else:
                if mode == 'train':
                    self.get_wav_normalizer()
                    save_dir = self.config.dataset.get("norm_wav_path", None) or "./datasets/CREMAD/wav_norm.pkl"
                    logging.warning(f"Saving wav norm to {save_dir}")

        # Нормализация лиц
        if self.active_modalities.get("face", False) and self.config.dataset.get("norm", True):
            if self.config.dataset.get("norm_face_path", None) and os.path.exists(self.config.dataset.get("norm_face_path", None)):
                self.face_norm = pickle.loads(open(self.config.dataset.norm_face_path, "rb").read())
                logging.info(f"Loaded face norm from {self.config.dataset.norm_face_path}")
                logging.info(f"Norm values are {self.face_norm}")
            else:
                if mode == 'train':
                    self.get_face_normalizer()
                    save_dir = self.config.dataset.get("norm_face_path", None) or "./datasets/CREMAD/face_norm.pkl"
                    logging.warning(f"Saving face norm to {save_dir}")

    def split_inclusive(self, mode, class_dict):
        """
        Если выбран метод 'inclusive', мы читаем train.csv и test.csv,
        а затем делаем дополнительный train/val/test split (stratified_split).
        """
        self.norm_audio = {"total": {"mean": -7.1276217, "std": 5.116028}}
        self.train_data = {}
        self.test_data = {}
        # Инициализируем train_data/test_data теми же ключами, что и self.data, 
        # чтобы было куда складывать пути
        for k in self.data.keys():
            self.train_data[k] = []
            self.test_data[k] = []

        self.train_label, self.test_label, self.train_item, self.test_item = [], [], [], []

        self.train_csv = './datasets/CREMAD/train.csv'
        self.test_csv = './datasets/CREMAD/test.csv'

        if not os.path.exists(self.train_csv) or not os.path.exists(self.test_csv):
            print(f"Ошибка: CSV-файлы не найдены: train_csv={self.train_csv}, test_csv={self.test_csv}")
            sys.exit(1)

        def load_data(csv_file, data_dict, label_list, item_list):
            with open(csv_file, encoding='UTF-8-sig') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    file_id, emotion_label = row[0], row[1]
                    paths = {
                        "video": os.path.join(self.visual_feature_path, 'Image-01-FPS', file_id),
                        "audio": os.path.join(self.audio_feature_path, file_id + '.wav'),
                        "face": os.path.join(self.face_feature_path, file_id + '.npy'),
                        "face_image": os.path.join(self.face_image_path, file_id + '.npy')
                    }
                    
                    # Нужно учесть, что аудио нам нужно, если audio=True или spectrogram=True
                    need_audio = self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False)
                    
                    # Собираем список модальностей, которые нужно проверить на существование
                    check_mods = []
                    if self.active_modalities.get("video", False):
                        check_mods.append("video")
                    if need_audio:
                        check_mods.append("audio")
                    if self.active_modalities.get("face", False):
                        check_mods.append("face")
                    if self.active_modalities.get("face_image", False):
                        check_mods.append("face_image")

                    all_exist = all(os.path.exists(paths[m]) for m in check_mods)
                    if all_exist:
                        item_list.append(file_id)

                        # Если video включено, сохраняем путь
                        if self.active_modalities.get("video", False):
                            data_dict["video"].append(paths["video"])
                        # Если audio ИЛИ spectrogram включены, сохраняем путь в audio
                        if need_audio:
                            data_dict["audio"].append(paths["audio"])
                        if self.active_modalities.get("face", False):
                            data_dict["face"].append(paths["face"])
                        if self.active_modalities.get("face_image", False):
                            data_dict["face_image"].append(paths["face_image"])

                        label_list.append(class_dict[emotion_label])
                    else:
                        print(f"Пропущен {file_id}: отсутствуют файлы для активных модальностей")

        # Загружаем train.csv → в self.train_data
        load_data(self.train_csv, self.train_data, self.train_label, self.train_item)
        # Загружаем test.csv → в self.test_data
        load_data(self.test_csv, self.test_data, self.test_label, self.test_item)

        self.split_mode = "stratified_split"
        print(self.split_mode)

        # Теперь объединим train и test (по сути, это "train+test" до split'a)
        if self.split_mode == "stratified_split":
            total_data = {}
            for k in self.data.keys():
                total_data[k] = self.train_data[k] + self.test_data[k]
            total_item = self.train_item + self.test_item
            total_label = self.train_label + self.test_label

            X = []
            # Идём в порядке ["video", "audio", "face", "face_image"] — но только для тех, кто есть в self.data
            # (иначе будет несоответствие длины)
            mods_in_use = [m for m in ["video","audio","face","face_image"] if m in self.data]
            for m in mods_in_use:
                X.append(total_data[m])
            X.append(total_item)  # последний столбец – имя файла
            X = np.array(X).T  # shape: (samples, num_modalities+1)
            y = np.array(total_label)

            # Делим на train+val и test
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y,
                test_size=self.config.dataset.get("val_split_rate", 0.1),
                random_state=self.config.training_params.seed,
                stratify=y
            )
            # Делим X_trainval на train и val
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=self.config.dataset.get("val_split_rate", 0.1),
                random_state=self.config.training_params.seed,
                stratify=y_trainval
            )

            if mode == "test":
                X_final, y_final = X_test, y_test
            elif mode == "train":
                X_final, y_final = X_train, y_train
            elif mode == "val":
                X_final, y_final = X_val, y_val
            else:
                raise ValueError(f"Неизвестный режим: {mode}")

            # num_modalities – количество реальных ключей (video, audio, face, face_image)
            num_modalities = len(mods_in_use)
            # Заполняем self.data по колонкам
            for i, modality in enumerate(mods_in_use):
                self.data[modality] = X_final[:, i].tolist()
            # последний столбец – "item"
            self.data["item"] = X_final[:, num_modalities].tolist()
            self.label = y_final.tolist()

    def split_noninclusive(self, fold, mode, class_dict):
        """
        Если выбран метод 'non_inclusive', то данные берутся из data_splits_VALV.pkl.
        """
        split_file = './datasets/CREMAD/data_splits_VALV.pkl'
        if not os.path.exists(split_file):
            print(f"Ошибка: файл разбиения {split_file} не найден")
            sys.exit(1)

        with open('./datasets/CREMAD/normalization_audio.pkl', "r") as json_file:
            self.norm_audio = json.load(json_file)

        with open(split_file, "r") as json_file:
            train_val_test_splits = json.load(json_file)

        need_audio = self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False)

        for i in train_val_test_splits[str(fold + 1)][mode]:
            name = i.split("-")[0]
            label = class_dict[name.split("_")[2]]  # пример: '1025_IEO_NEU_XX' → [2] = 'NEU'
            paths = {
                "video": os.path.join(self.visual_feature_path, f'Image-{{:02d}}-FPS'.format(self.fps), name.split(".")[0]),
                "audio": os.path.join(self.audio_feature_path, name),
                "face": os.path.join(self.face_feature_path, name.replace(".wav", ".npy")),
                "face_image": os.path.join(self.face_image_path, name.replace(".wav", ".npy"))
            }

            # Собираем список, что проверять
            check_mods = []
            if self.active_modalities.get("video", False):
                check_mods.append("video")
            if need_audio:
                check_mods.append("audio")
            if self.active_modalities.get("face", False):
                check_mods.append("face")
            if self.active_modalities.get("face_image", False):
                check_mods.append("face_image")

            all_exist = all(os.path.exists(paths[m]) for m in check_mods)
            if all_exist:
                # Сохраняем пути
                if self.active_modalities.get("video", False):
                    self.data["video"].append(paths["video"])
                if need_audio:
                    self.data["audio"].append(paths["audio"])
                if self.active_modalities.get("face", False):
                    self.data["face"].append(paths["face"])
                if self.active_modalities.get("face_image", False):
                    self.data["face_image"].append(paths["face_image"])

                self.label.append(label)
            else:
                print(f"Пропущен {name}: отсутствуют файлы для активных модальностей")

    def get_wav_normalizer(self):
        # Если совсем нет self.data["audio"], выходим
        if not self.data.get("audio"):
            return
        count, wav_sum, wav_sqsum, max_duration = 0, 0, 0, 0
        for cur_wav in tqdm(self.data["audio"]):
            audio, fps = torchaudio.load(cur_wav)
            audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)
            audio = audio[0]
            if audio.shape[0] > max_duration * self.sampling_rate:
                max_duration = audio.shape[0] / self.sampling_rate
            wav_sum += torch.sum(audio)
            wav_sqsum += torch.sum(audio ** 2)
            count += len(audio)
        wav_mean = wav_sum / count
        wav_var = (wav_sqsum / count) - (wav_mean ** 2)
        wav_std = np.sqrt(wav_var)
        self.wav_norm = {"mean": wav_mean, "std": wav_std, "max_duration": max_duration}
        save_path = self.config.dataset.get("norm_wav_path", "./datasets/CREMAD/wav_norm.pkl")
        with open(save_path, "wb") as f:
            f.write(pickle.dumps(self.wav_norm))

    def get_face_normalizer(self):
        if not self.data.get("face", []):
            return
        count, vid_sum, vid_sqsum, max_faces = 0, 0, 0, 0
        for cur_vid in tqdm(self.data["face"]):
            feats = np.load(cur_vid)
            vid_sum += np.sum(feats, axis=0)
            vid_sqsum += np.sum(feats ** 2, axis=0)
            count += feats.shape[0]
            if feats.shape[0] > max_faces:
                max_faces = feats.shape[0]
        vid_mean = vid_sum / count
        vid_var = (vid_sqsum / count) - (vid_mean ** 2)
        vid_std = np.sqrt(vid_var)
        self.face_norm = {"mean": vid_mean, "std": vid_std, "max_faces": max_faces}
        save_path = self.config.dataset.get("norm_face_path", "./datasets/CREMAD/face_norm.pkl")
        with open(save_path, "wb") as f:
            f.write(pickle.dumps(self.face_norm))

    def __len__(self):
        # Возвращаем длину на основе первой "реальной" модальности (без spectrogram)
        # Порядок можно выбирать: например, отдаем длину "audio", если есть, иначе "video" и т.д.
        # В данном случае идём по keys() в том порядке, в котором мы их создали.
        for m in ["video", "audio", "face", "face_image"]:
            if m in self.data:
                return len(self.data[m])
        return 0

    def _get_images(self, idx):
        if "video" not in self.data:
            return False

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

        image_samples = sorted(os.listdir(self.data["video"][idx]))
        images = torch.zeros((self.num_frame, 3, 224, 224))
        for i, img_file in enumerate(image_samples[:self.num_frame]):
            img = Image.open(os.path.join(self.data["video"][idx], img_file)).convert('RGB')
            images[i] = transform(img)

        return torch.permute(images, (1, 0, 2, 3))

    def _get_face(self, idx):
        if "face" not in self.data:
            return False
        face_features = torch.from_numpy(np.load(self.data["face"][idx]))
        # Если у вас не инициализирован self.face_norm (например, mode='val'/'test' до train),
        # нужно предусмотреть проверку. Предположим, что она есть.
        if hasattr(self, "face_norm"):
            return (face_features - self.face_norm["mean"]) / self.face_norm["std"]
        else:
            return face_features

    def _get_face_image(self, idx):
        if "face_image" not in self.data:
            return False
        return torch.from_numpy(np.load(self.data["face_image"][idx]))

    def _get_audio(self, idx):
        # Если audio и/или spectrogram неактивны, возвращаем False
        if not (self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False)):
            return False

        audio, fps = torchaudio.load(self.data["audio"][idx])
        audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)[0]
        if hasattr(self, "wav_norm"):
            audio = (audio - self.wav_norm["mean"]) / self.wav_norm["std"]
        return audio

    def _get_spectrogram(self, idx, audio):
        if not self.active_modalities.get("spectrogram", False):
            return False

        # Если audio=False, загружаем через librosa
        if audio is False:
            samples, _ = librosa.load(self.data["audio"][idx], sr=self.sampling_rate)
            # Пример: вы берёте максимум 3 секунды
            resamples = np.tile(samples, 3)[:self.sampling_rate * 3]
            resamples = np.clip(resamples, -1, 1)
        else:
            resamples = audio.numpy()

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        if self.norm_type == "per_sample":
            spectrogram = (
                spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-9
            )
        elif self.norm_type == "total":
            spectrogram = (
                spectrogram - self.norm_audio["total"]["mean"]
            ) / (self.norm_audio["total"]["std"] + 1e-9)

        return torch.from_numpy(spectrogram)

    def __getitem__(self, idx):
        data = {}

        # 1. АУДИО: грузим, только если audio=True или spectrogram=True
        if self.active_modalities.get("audio", False) or self.active_modalities.get("spectrogram", False):
            audio_data = self._get_audio(idx)
        else:
            audio_data = False

        # 2. СПЕКТР
        if self.active_modalities.get("spectrogram", False):
            data[0] = self._get_spectrogram(idx, audio_data)
        else:
            data[0] = False

        # 3. ВИДЕО
        if self.active_modalities.get("video", False):
            data[1] = self._get_images(idx)
        else:
            data[1] = False

        # 4. АУДИО в data[2]
        # Если audio=False, тогда audio_data уже False, так что всё корректно
        data[2] = audio_data

        # 5. FACE
        if self.active_modalities.get("face", False):
            data[3] = self._get_face(idx)
        else:
            data[3] = False

        # 6. FACE_IMAGE
        if self.active_modalities.get("face_image", False):
            data[4] = self._get_face_image(idx)
        else:
            data[4] = False

        return {"data": data, "label": self.label[idx], "idx": idx}



class CramedD_Dataloader:
    def __init__(self, config):
        self.config = config
        train_dataset, valid_dataset, test_dataset = self._get_datasets()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        # При желании можно включить больше потоков
        num_cores = 0
        print(f"Available cores {len(os.sched_getaffinity(0))}")
        print(f"Using {num_cores} workers")

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training_params.batch_size,
            num_workers=num_cores,
            shuffle=True,
            pin_memory=self.config.training_params.pin_memory,
            generator=g,
            collate_fn=collate_fn_padd,
            worker_init_fn=seed_worker
        )

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            collate_fn=collate_fn_padd,
            pin_memory=self.config.training_params.pin_memory
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            collate_fn=collate_fn_padd,
            pin_memory=self.config.training_params.pin_memory
        )

    def _get_datasets(self):
        return (
            CremadDataset(config=self.config, mode="train"),
            CremadDataset(config=self.config, mode="val"),
            CremadDataset(config=self.config, mode="test")
        )
