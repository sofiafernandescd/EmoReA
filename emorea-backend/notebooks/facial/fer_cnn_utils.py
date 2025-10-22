import numpy as np
import cv2

##### PREPROCESSING FUNCTIONS ######
def preprocess_frame(frame, target_size=(48, 48)):
    """
    Preprocess a single frame for CNN input.
    - Resize
    - Convert to grayscale (optional for FER-like tasks)
    - Normalize pixel values to [0, 1]
    - Expand dims for batch/channel compatibility
    """
    # Convert to grayscale if desired
    gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size)

    # Normalize and shape to (H, W, 1)
    img = resized.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def preprocess_frame(frame, target_size=(48, 48)):

    # ensure frame is in RGB mode (convert from RGBA, L, or P if needed)
    if not frame.mode == 'RGB':
        frame = frame.convert('RGB')
    
    np_frame = np.array(frame)
    gray = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size)

    # normalize and add channel dimension
    resized = resized.astype("float32") / 255.0
    resized = np.expand_dims(resized, axis=-1)

    return resized


####### BUILD MODEL #######
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization, Input
)

#### LIGHT CNN OUTPUT EMOTIONS ####
def build_light_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Lightweight CNN for emotion recognition from frames.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#### LIGHT CNN OUTPUT EMBEDDINGS ####

def build_light_cnn_embedder(input_shape=(48, 48, 1), embedding_dim=128):
    """
    Lightweight CNN for extracting embeddings from facial frames.
    Returns feature vectors (not class predictions).
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    embeddings = Dense(embedding_dim, activation='relu', name='embedding')(x)

    model = Model(inputs, embeddings)
    return model


#### CNN Frame Embedder ######
from sklearn.base import BaseEstimator, TransformerMixin

class CNNFrameEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape=(48,48,1), embedding_dim=128):
        self.model = build_light_cnn_embedder(input_shape, embedding_dim)
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim

    def fit(self, X, y=None):
        # CNN feature extractor is not trained here (could be fine-tuned later)
        return self

    def transform(self, frames_list):
        """
        frames_list: list of lists of frames (one list per video)
        returns: array of shape (n_videos, embedding_dim)
        """
        video_embeddings = []
        for frames in frames_list:
            X_frames = np.stack([preprocess_frame(f, self.input_shape[:2]) for f in frames])
            emb = self.model.predict(X_frames, verbose=0)
            video_embeddings.append(np.mean(emb, axis=0))
        return np.vstack(video_embeddings)


###### TRANSFER LEARNING #####

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def build_pretrained_cnn_embedder(input_shape=(64,64,3), embedding_dim=128):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # freeze for feature extraction
    x = GlobalAveragePooling2D()(base.output)
    embeddings = Dense(embedding_dim, activation='relu', name='embedding')(x)
    return Model(base.input, embeddings)


def preprocess_frame(frame, target_size=(48, 48)):

    # ensure frame is in RGB mode (convert from RGBA, L, or P if needed)
    if not frame.mode == 'RGB':
        frame = frame.convert('RGB')
    
    np_frame = np.array(frame)
    gray = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size)

    # normalize and add channel dimension
    resized = resized.astype("float32") / 255.0
    resized = np.expand_dims(resized, axis=-1)

    return resized

def build_light_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Lightweight CNN for emotion recognition from frames.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(256, (3,3), activation='relu', padding='same',),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.7),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Callbacks for better training control
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

callbacks = [
    # Save only the best model (based on validation accuracy)
    ModelCheckpoint(
    "best_cnn_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
    ),
    # Stop early if no improvement
    EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation stops improving
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]