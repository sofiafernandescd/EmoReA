import tensorflow as tf
from tensorflow.keras import layers, models

def create_raw_audio_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

target_length = int(2 * 16000)
num_emotions = 8 # Example
input_shape = (target_length, 1) # Single channel for raw audio

raw_audio_model = create_raw_audio_cnn_model(input_shape, num_emotions)
raw_audio_model.summary()

raw_audio_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# Assuming you have your training data (X_train, y_train) prepared
# X_train should have shape (num_samples, target_length) and needs a channel dimension
# X_train = np.expand_dims(X_train, axis=-1)
# raw_audio_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
