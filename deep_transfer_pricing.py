import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_dataset(json_file_path):
    """load image file paths and PPSF values from a JSON file"""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    image_paths = [item['FilePath'] for item in data]
    prices = np.array([item['PPSF'] for item in data])
    return image_paths, prices

def preprocess_images(image_paths):
    """load and preprocess images for model input"""
    images = [preprocess_image(path) for path in image_paths]
    return np.array(images)

def preprocess_image(path):
    """load a single image and preprocess it for VGG19"""
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img[0]

def build_model():
    """build/compile VGG19"""
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # freeze base

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

def plot_history(history):
    """plot training/validation loss and MAE"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    image_paths, prices = load_dataset('data.json')
    images = preprocess_images(image_paths)

    X_train, X_val, y_train, y_val = train_test_split(images, prices, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    model = build_model()
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=15)
    plot_history(history)
    model.summary()

if __name__ == "__main__":
    main()
