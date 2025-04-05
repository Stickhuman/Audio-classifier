import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import configparser
from sklearn.utils.class_weight import compute_class_weight
import os

def get_callbacks(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    callbacks = [
        EarlyStopping(
            patience=int(config['Training']['early_stopping_patience']),
            monitor='val_loss',
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=int(config['Training']['reduce_lr_patience']),
            monitor='val_loss'
        ),
        TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            profile_batch=0
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    
    history = model.fit(
        X_train, y_train,
        batch_size=int(config['Training']['batch_size']),
        epochs=int(config['Training']['epochs']),
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(config_path),
        class_weight=class_weights
    )
    return history