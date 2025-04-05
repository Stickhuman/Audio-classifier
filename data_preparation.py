import librosa
from joblib import Memory
import numpy as np
import os
from tqdm import tqdm
import configparser

memory = Memory(
    location='./feature_cache', 
    verbose=1, 
    bytes_limit=10*1024**3  # 限制缓存大小
)
memory.clear(warn=False)

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.sr = int(config['Audio']['sample_rate'])
        self.duration = int(config['Audio']['duration'])
        self.target_samples = self.sr * self.duration
        
    def load_audio(self, file_path, augment=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件 {file_path} 不存在")
        try:
            y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            y = librosa.util.fix_length(y, self.target_samples)
            
            if augment and np.random.rand() > 0.5:
                speed_factor = np.random.uniform(0.85, 1.15)
                y = librosa.effects.time_stretch(y, rate=speed_factor)
            return y
        except librosa.util.exceptions.ParameterError:
            print(f"不支持的格式: {file_path}")
            return None
    
    def extract_features(self, y):
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=int(self.config['Audio']['n_mels']),
            fmin=int(self.config['Audio']['fmin']),
            fmax=int(self.sr//2 * float(self.config['Audio']['fmax_ratio'])),
            n_fft=int(self.config['Audio']['n_fft']),
            hop_length=int(self.config['Audio']['hop_length'])
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-8)
        return S_norm[..., np.newaxis]

def load_dataset(config_path, data_dir, augment=False):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    processor = AudioProcessor(config)
    features, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for file in tqdm(os.listdir(class_path), desc=class_name):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(class_path, file)
                y = processor.load_audio(file_path, augment=augment)
                if y is not None:
                    features.append(processor.extract_features(y))
                    labels.append(label_idx)
    return np.array(features), np.array(labels), class_names