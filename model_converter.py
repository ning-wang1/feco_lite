import tensorflow as tf


def model_converter(model_path, save_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.

    with open(save_path, 'wb') as f:
      f.write(tflite_model)


if __name__ == '__main__':
    model_path = '/home/ning/extens/federated_contrastive/checkpoints/model1'
    save_path = 'model.tflite'
    model_converter(model_path, save_path)