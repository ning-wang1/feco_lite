import tensorflow as tf


def model_converter(model_path, save_path, quantization=None):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
    if quantization is None:
        tflite_model = converter.convert()
    elif 'int8' in quantization:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()
    elif 'float16' in quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    else:
        print('undefined quantization')

    # Save the model.
    with open(save_path, 'wb') as f:
      f.write(tflite_model)


if __name__ == '__main__':
    model_path = '/home/ning/extens/federated_contrastive/checkpoints/model1'
    # save_path = 'model.tflite'
    # model_converter(model_path, save_path)

    save_path = 'model_int8.tflite'
    model_converter(model_path, save_path, 'int8')

    save_path = 'model_float16.tflite'
    model_converter(model_path, save_path, 'float16')