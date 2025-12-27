import tensorflow as tf
from tensorflow.keras import layers, models

def create_dummy_model():
    # Architecture: CNN (Convolutional Neural Network)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # 0 = Fake, 1 = Real
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save('deepfake_model.h5')
    print("SUCCESS: deepfake_model.h5 generated.")

if __name__ == "__main__":
    create_dummy_model()