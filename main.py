# Rewrite by Shanghao Shi
# The purpose is to generate proper reference templates and thresholds
# Every threshold is set as the minimal score obtained by the normal traffic

import tensorflow as tf
import numpy as np
from setup_BaIoT import BaIoT
import time
from evaluation import split_evaluate

def get_output(interpreter):
    output_details = interpreter.get_output_details()[1]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    return output


def generate_template(x_val, y_val):
  val_num = x_val.shape[0]
  val_idxes = list(range(val_num))
  reference=np.zeros(64)
  count=0
  for val_idx in val_idxes:
      if y_val[val_idx]==0:
          count=count+1
          input_tensor = np.array(np.expand_dims(x_val[val_idx,:], 0), dtype=np.float32)
    
          # Set the tensor to point to the input data to be inferred
          input_index =interpreter.get_input_details()[0]["index"]
          interpreter.set_tensor(input_index, input_tensor)
          interpreter.invoke()
          out = get_output(interpreter)
          reference=reference+out
  print("Number of good samples", count)
  reference=reference/count
  return count, reference
    
def generate_threshold(count, x_val, y_val, reference):
    th=1
    val_num = x_val.shape[0]
    val_idxes = list(range(val_num))
    scores=np.zeros(count)
    idx=0
    for val_idx in val_idxes:
        if y_val[val_idx]==0:
            input_tensor = np.array(np.expand_dims(x_val[val_idx,:], 0), dtype=np.float32)
            input_index =interpreter.get_input_details()[0]["index"]
            #print(input_index)
            #print(input_tensor)
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            out = get_output(interpreter)
            score = np.inner(out, reference)
            scores[idx]=score
            idx=idx+1
    scores.sort()
    th=scores[int(0.005*count)]
    print("th is:", th)
    return th
    
def testing(x_test, y_test, reference, th):
    data_num = x_test.shape[0]
    idxes = list(range(data_num))
    np.random.shuffle(idxes)
    test_time=0
    test_accuracy=0
    trp=0
    trn=0
    fp=0
    fn=0

    scores = np.zeros(data_num)
    for idx in idxes: # [:10]
        input_tensor = np.array(np.expand_dims(x_test[idx,:], 0), dtype=np.float32)
    
        # Set the tensor to point to the input data to be inferred
        input_index =interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index, input_tensor)

        # Run the inference
        time1 = time.time()
        interpreter.invoke()
        out = get_output(interpreter)
        score = np.inner(out, reference)
        time2 = time.time()
        # Test time and accuracy evaluation 
        test_time=test_time+(time2-time1)/data_num
        scores[idx] = score
        
        if int(score<th)==y_test[idx]:
            test_accuracy=test_accuracy+1/data_num  
        if int(score<th)==0 and y_test[idx]==0:
            trn=trn+1
        elif int(score<th)==1 and y_test[idx]==1:
            trp=trp+1
        elif int(score<th)==0 and y_test[idx]==1:
            fn=fn+1
        elif int(score<th)==1 and y_test[idx]==0:
            fp=fp+1
    print(f"trp is {trp}")
    print(f"trn is {trn}") 
    print(f"fp is {fp}") 
    print(f"fn is {fn}")     
    print(f'The processing time is {test_time} seconds')
    print(f"The detection results is {test_accuracy}")
    print(f"The precision is {trp/(trp+fp)}")
    print(f"The recall is {trp/(trp+fn)}")
    print(f"The f1 score is {2*(trp/(trp+fp)*trp/(trp+fn))/(trp/(trp+fp)+trp/(trp+fn))}")
    print(f"The FPR is {fp/(fp+trn)}")

    split_evaluate(y_test, scores, plot=True, filename='feco', manual_th=th)


if __name__=="__main__":
    DEVICE_NAMES = ['Danmini_Doorbell', 'Ecobee_Thermostat',
               'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
               'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
               'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
               'SimpleHome_XCS7_1003_WHT_Security_Camera']
    tflite_model_path = 'model.tflite'
    # tflite_model_path = 'model_float16.tflite'
    tflite_model_path = 'model_int8.tflite'
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
        
    # get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Read the image and decode to a tensor
    device_name = DEVICE_NAMES[1]
    data = BaIoT(device_name)
    x_val, y_val = data.x_val, data.y_val
    x_test, y_test = data.x_test, data.y_test

    count, reference=generate_template(x_val, y_val)
    #print(reference)
    th=generate_threshold(count, x_val, y_val, reference)
    testing(x_test, y_test, reference, th)
    
    
    
    
