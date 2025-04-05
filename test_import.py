import librosa
print("Librosa版本:", librosa.__version__)  # 应显示0.9.2
print("模块检查:", hasattr(librosa.util, 'cache'))  # 应输出True