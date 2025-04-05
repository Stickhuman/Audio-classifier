import tensorflow as tf
from keras import layers, models
import configparser

def build_model(config_path, input_shape, num_classes):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    conv_filters = [int(f) for f in config['Model']['conv_filters'].split(',')]
    dense_units = int(config['Model']['dense_units'])
    dropout_rate = float(config['Model']['dropout_rate'])
    learning_rate = float(config['Model']['learning_rate'])
    use_residual = config.getboolean('Model', 'use_residual', fallback=False)
    use_attention = config.getboolean('Model', 'use_attention', fallback=False)
    pool_type = config['Model'].get('pool_type', 'max')
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    for filters in conv_filters:
        residual = x  # 残差分支
        x = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        
        if use_residual:
            x = layers.add([x, residual])
        
        if pool_type == 'max':
            x = layers.MaxPooling2D((2,2))(x)
        elif pool_type == 'avg':
            x = layers.AveragePooling2D((2,2))(x)
    
    if use_attention:
        x = layers.Attention()([x, x])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model