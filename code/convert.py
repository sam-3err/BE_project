import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Rebuild model architecture manually
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.load_weights('fer.h5')
print('Model loaded!')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print('Converted!')

with open('fer.tflite', 'wb') as f:
    f.write(tflite_model)
print('Done! fer.tflite created!')