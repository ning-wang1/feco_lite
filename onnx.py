# -- Ning Wang 02/28/2022

# How you can convert a pytorch model to tflite, there are three steps:

# 1. Save pytorch model to ONNX.
# You should do this in the pytorch env where you train your pytorch model
"""

dummy = torch.autograd.Variable(torch.randn(1, 115, device='cuda'))  # the input data

torch.onnx.export(model.module,  # model being run
                  dummy,  # model input (or a tuple for multiple inputs)
                  "checkpoints/model1.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

This will output a model with a suffix .onnx
"""

# 2. Convert from ONNX to TensorFlow freezGraph

"""
if you dont have the onnx-tf, then install this:
    pip install  onnx-tf
use the following command line:
    onnx-tf convert -i 'model1.onnx' -o 'model1'
    
you will get a folder that includes another two folders and also a file with '.pb' suffix
"""

# 3. Convert the general tfmodel to a tflite model
"""
you can use a function like what follows
def model_converter(model_path, save_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.

    with open(save_path, 'wb') as f:
      f.write(tflite_model)
      
An example of the save path is: model_path = 'model.tflite'
you can get a model with suffix '.tflite'
"""

# 4 some additional information
"""
How to use the tflite model?

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the image and decode to a tensor
x_train_attack, x_train_normal = data.x_train_attack, data.x_train_normal

# set the tensor to point to the input data to be inferred
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)

# Run the inference
interpreter.invoke()
output_details = interpreter.get_output_details()[0]
"""

























# import onnx
# from onnx2keras import onnx_to_keras
#
# # Load ONNX model
# onnx_model = onnx.load('/home/ning/extens/federated_contrastive/checkpoints/model1.onnx')
#
# # Call the converter (input - is the main model input name, can be different for your model)
# k_model = onnx_to_keras(onnx_model, ['input'])

# onnx-tf convert -i 'model1.onnx' -o 'model1'
