from sklearn.model_selection import train_test_split
import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from data_preparation import load_dataset
from module_preparation import build_model
from training import train_model
from saving import save_model, convert_to_tflite
from bpm_measurement import BPMCalculator

def main():
    parser = argparse.ArgumentParser(description='音乐BPM分类器')
    parser.add_argument('--data_dir', type=str, required=False, help='数据集目录路径')
    parser.add_argument('--config', type=str, default='config.ini', help='配置文件路径')
    parser.add_argument('--measure_bpm', type=str, help='单文件BPM测量模式')
    parser.add_argument('--batch_bpm', type=str, help='批量测量模式（输入目录）')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录路径')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'], help='结果文件格式')
    
    args = parser.parse_args()

    if args.measure_bpm:
        calculator = BPMCalculator()
        print(f"估计BPM: {calculator.calculate_bpm(args.measure_bpm):.1f}")
    elif args.batch_bpm:
        files = glob.glob(f"{args.batch_bpm}/**/*.*", recursive=True)
        valid_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        
        calculator = BPMCalculator()
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'bpm_results.{args.format}')
        calculator.batch_process(valid_files, output_path)
    else:
        config = configparser.ConfigParser()
        config.read(args.config)
    
        X, y, classes = load_dataset(config_path=args.config, data_dir=args.data_dir)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2,
            stratify=y,
            random_state=config.getint('Training', 'seed', fallback=42)
        )

        model = build_model(args.config, X_train[0].shape, len(classes))
        history = train_model(model, X_train, y_train, X_val, y_val, args.config)
        
        save_model(model, path="saved_model")
        convert_to_tflite('saved_model', quantize=True)
        
        if config.getboolean('Visualization', 'enable_training_curve', fallback=True):
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['accuracy'], label='训练集准确率')
            plt.plot(history.history['val_accuracy'], label='验证集准确率')
            plt.title('模型训练指标')
            plt.ylabel('准确率')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig('training_curve.png', dpi=300)
            plt.close()

if __name__ == "__main__":
    main()