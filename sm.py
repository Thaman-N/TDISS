import tensorflow as tf
import h5py
import numpy as np
import os

MODEL_PATH = 'D:\\Thaman\\College\\Capstone\\capstoneproj\\Violence-Detection-RWF-2000\\Models\\keras_model.h5'
OUTPUT_PATH = 'D:\\Thaman\\College\\Capstone\\capstoneproj\\cuenet\\converted_model.h5'

# Define the model architecture exactly as it was in the original repository
def create_c3d_model(input_shape=(64, 224, 224, 3)):
    """
    Create C3D model based on the GitHub repository's implementation
    This should match the structure of the keras_model.h5 file
    """
    inputs = tf.keras.layers.Input(input_shape)
    
    # First 3D Conv Block
    x = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_1')(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                             name='max_pooling3d_1')(x)
    
    # Second 3D Conv Block
    x = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                             name='max_pooling3d_2')(x)
    
    # Third 3D Conv Block
    x = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_3')(x)
    x = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_4')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                             name='max_pooling3d_3')(x)
    
    # Fourth 3D Conv Block
    x = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_5')(x)
    x = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_6')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                             name='max_pooling3d_4')(x)
    
    # Fifth 3D Conv Block
    x = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_7')(x)
    x = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', 
                          activation='relu', name='conv3d_8')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                             name='max_pooling3d_5')(x)
    
    # Fully Connected Layers
    x = tf.keras.layers.Flatten(name='flatten_1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='dense_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_2')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', name='dense_3')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

try:
    # Create new model with accurate architecture
    print("Creating new model with C3D architecture...")
    model = create_c3d_model()
    
    print("Model created. Compiling...")
    # Compile the model with similar settings to original
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled. Summary:")
    model.summary()
    
    # Create a new weights file that doesn't rely on the old serialization format
    print("\nExtracting weights from original h5 file...")
    
    with h5py.File(MODEL_PATH, 'r') as h5f:
        weight_group = h5f['model_weights']
        
        # Get layer names from both models
        original_layer_names = list(weight_group.keys())
        print(f"Original layers: {original_layer_names}")
        
        # Create a dictionary to store the weights temporarily
        weights_dict = {}
        
        # Extract weights from h5 file
        for layer_name in weight_group:
            if layer_name in ['conv3d_1', 'conv3d_2', 'conv3d_3', 'conv3d_4', 'conv3d_5', 
                             'conv3d_6', 'conv3d_7', 'conv3d_8', 'dense_1', 'dense_2', 'dense_3']:
                layer_weights = []
                
                # Get kernel and bias weights
                if 'kernel:0' in weight_group[layer_name]:
                    kernel = np.array(weight_group[layer_name]['kernel:0'])
                    layer_weights.append(kernel)
                
                if 'bias:0' in weight_group[layer_name]:
                    bias = np.array(weight_group[layer_name]['bias:0'])
                    layer_weights.append(bias)
                
                if layer_weights:
                    weights_dict[layer_name] = layer_weights
    
    # Apply weights to the new model
    print("\nApplying weights to new model...")
    for layer in model.layers:
        if layer.name in weights_dict:
            print(f"Setting weights for layer: {layer.name}")
            try:
                weights = weights_dict[layer.name]
                layer.set_weights(weights)
                print(f"  Successfully set weights for {layer.name}")
            except Exception as e:
                print(f"  Error setting weights for {layer.name}: {e}")
    
    # Test with dummy data to ensure the model works
    print("\nTesting model with dummy data...")
    dummy_input = np.random.rand(1, 64, 224, 224, 3).astype(np.float32)
    result = model.predict(dummy_input)
    print(f"Test prediction: {result}")
    
    # Save the model in TF 2.10 compatible format
    print("\nSaving model...")
    model.save(OUTPUT_PATH)
    print(f"Model saved to {OUTPUT_PATH}")
    
except Exception as e:
    print(f"Error in model creation or conversion: {e}")