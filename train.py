from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from data.loader import load_data

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train():
    X_train, y_train = load_data()
    if X_train is None:
        print("Dataset not available, please add data.")
        return
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save('model/eye_status_cnn.h5')
    print("Model trained and saved.")

if __name__ == "__main__":
    train()
