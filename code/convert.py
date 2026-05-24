import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_MODEL = os.path.join(BASE_DIR, "_mini_XCEPTION.102-0.66.hdf5")
OUTPUT_MODEL = os.path.join(BASE_DIR, "fer.tflite")


def main():
    # This checkpoint has better validation accuracy than the old fer.h5 model.
    # Run this file again whenever you need to regenerate the deployed TFLite model.
    print("Loading improved model from:", SOURCE_MODEL)
    model = tf.keras.models.load_model(SOURCE_MODEL, compile=False)
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(OUTPUT_MODEL, "wb") as f:
        f.write(tflite_model)

    print("Saved improved TFLite model to:", OUTPUT_MODEL)


if __name__ == "__main__":
    main()
