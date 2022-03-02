import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
from setup_BaIoT import BaIoT
import time
import os
import pandas as pd


DEVICE_NAMES = ['Danmini_Doorbell', 'Ecobee_Thermostat',
               'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
               'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
               'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
               'SimpleHome_XCS7_1003_WHT_Security_Camera']

model_path = 'model.tflite'


def get_ouput(interpreter):
    output_details = interpreter.get_output_details()[1]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    return output


# Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the image and decode to a tensor
device_name = DEVICE_NAMES[1]
data = BaIoT(device_name)
x_test, y_test = data.x_test, data.y_test

# load the normal template and the threshold for testing
file_dir = os.getcwd()
reference = np.load(os.path.join(file_dir, 'utils/vec1.npy'))
th = np.load(os.path.join(file_dir, 'utils/threshold1.npy'))

# Preprocess the image to required size and cast
input_shape = input_details[0]['shape']

# testing
data_num = x_test.shape[0]
idxes = list(range(data_num))
np.random.shuffle(idxes)

for idx in idxes[:100]:
    input_tensor = np.array(np.expand_dims(x_test[idx,:], 0), dtype=np.float32)

    # set the tensor to point to the input data to be inferred
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)

    # Run the inference
    time1 = time.time()
    interpreter.invoke()
    out = get_ouput(interpreter)
    score = np.inner(out, reference)
    time2 = time.time()

    print(f'the processing time is {time2-time1} seconds')
    print(f"the detection results is {score < th}")
    print(f"the true label is {y_test[idx]}")
