import tensorflow as tf
import os

def save_model(model, format='tf', path='saved_model'):
    if format == 'tf':
        model.save(path, save_format='tf')
    elif format == 'h5':
        model.save(f"{path}.h5")
        
def convert_to_tflite(model_path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)