import librosa
import numpy as np
import os
import csv
import json
from tqdm.auto import tqdm  # 修正后的导入

class BPMCalculator:
    def __init__(self, config_path='config.ini'):
        """
        初始化BPM计算器
        """
        self.config = self._load_config(config_path)
        self.sr = self.config.getint('Audio', 'sr', fallback=22050)

    def _load_config(self, path):
        """ 加载配置文件 """
        import configparser
        config = configparser.ConfigParser()
        config.read(path)
        return config

    def calculate_bpm(self, audio_path):
        try:
            # 加载音频（强制单声道）
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
            # 检查有效音频长度
            if len(y) < sr * 0.5:  # 短于0.5秒的视为无效
                raise ValueError("音频过短")
            
          # 计算节拍
            tempo = librosa.beat.tempo(
                y=y, 
                sr=sr,
                start_bpm=60,  # 合理预设范围
                std_bpm=20,    # 允许的BPM波动范围
                aggregate=np.mean
            )
        
            # 处理多维返回
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            
            return float(tempo)
    
        except Exception as e:
            print(f"[ERROR] 无法处理 {audio_path}: {str(e)}")
            return 0.0  # 或抛出异常终止程序

    def batch_process(self, file_list, output_path):
        """
        批量处理模式
        """
        results = []
        for file in tqdm(file_list, desc="Processing"):
            try:
                bpm = self.calculate_bpm(file)
                results.append({'file': file, 'bpm': bpm})
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

        # 保存结果
        ext = os.path.splitext(output_path)[1].lower()
        with open(output_path, 'w', encoding='utf-8') as f:
            if ext == '.json':
                json.dump(results, f, indent=2)
            else:  # 默认csv格式
                writer = csv.DictWriter(f, fieldnames=['file', 'bpm'])
                writer.writeheader()
                writer.writerows(results)

if __name__ == "__main__":
    # 测试用代码
    calc = BPMCalculator()
    print(f"Test BPM: {calc.calculate_bpm('test.mp3')}")
