
# coding: utf-8

# In[1]:


#python
# coding: utf-8

### Required Imports ###
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import pandas as pd
import numpy as np
import os, time, glob

import tensorflow as tf
from sklearn.model_selection import train_test_split

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2, os
import itertools
import random
import time
from sklearn.model_selection import train_test_split
print("Imported Requirements")


# In[6]:


##### HYPERPARAMETERS #####
INPUT_LAYER_SHAPE_X = 150
INPUT_LAYER_SHAPE_Y = 65
KERNEL_SIZE = [5, 5]
POOLING_SIZE = [2, 2]
LAYER_1_FILTERS = 32
LAYER_2_FILTERS = 64

DENSE_LAYER_UNITS = 1024
LEARNING_RATE = 0.001
DROPOUT = 0.4
BATCH_SIZE = 100

### Specifies the number of steps the model will take. Can exceed the number of images to train ###
TRAIN_STEPS = 200

### Specifies the number of runs through the training data ###
### None implies that the model will train till the number of steps specified ###
NUM_EPOCHS = None

### Training plus test data size ###
DATA_SIZE = 100000

MODEL_DIR = "tmp/mnist_convnet_model"


# In[3]:


def subtract_images(image1,image2):
    return 255-np.array([[k-l if k > l else l-k for k,l in zip(i,j)] for i,j in zip(image1.tolist(),image2.tolist())])   

start = time.time()

## Import images
mypath='AND_dataset\\Dataset[Without-Features]\\AND_Images[WithoutFeatures]'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]),cv2.IMREAD_GRAYSCALE )
print("Imported Images")

## Crop images ##
images = [i[~np.all(i == 255, axis=1)][:,~np.all(i[~np.all(i == 255, axis = 1)] == 255, axis=0)] for i in images]
print("Cropped Images")

## Resize images ##
shapes = [i.shape for i in images]
height = [i[0] for i in shapes]
length = [i[1] for i in shapes]
avg_height = int(sum(height)/len(height))
avg_length = int(sum(length)/len(length))
print(avg_length, avg_height)
images = [cv2.resize(images[i],(avg_length, avg_height)) for i in range(len(images))]
print("Resized Images")

## Write Processed images ##
processed_directory = "Processed/"
subtracted_directory = "Subtracted/"
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)
if not os.path.exists(subtracted_directory):
    os.makedirs(subtracted_directory)
save = [cv2.imwrite(processed_directory+onlyfiles[i],images[i]) for i in range(len(images))]
print("Processed and Written Images")

## Selecting similar sample ##
authors = sorted(set([i.split("_")[0][:4] for i in onlyfiles]))
author_dict = {}

for i in authors:
    author_dict[i] = []
continue_index = 0
sorted_keys = sorted(author_dict.keys())
for i in sorted_keys:
    for j in onlyfiles[continue_index:]:
        if i in j:
            author_dict[i].append(j)
        else:
            continue_index = onlyfiles.index(j)
            break

permutated_dict = {}
for i in authors:
    permutated_dict[i] = []
for i in author_dict.keys():
    for r in itertools.product(author_dict[i], author_dict[i]):
        if (r[0] != r[1]) and ([r[1],r[0]] not in permutated_dict[i]):
            permutated_dict[i].append([r[0],r[1]])
            
similar = [j for i in permutated_dict.keys() for j in permutated_dict[i]]
similar_sample = random.sample(similar,int(DATA_SIZE/2) if int(DATA_SIZE/2) < len(similar) else len(similar))
print("Similar Sample generated")

## Selecting different sample ##
different_keys = [i.split("*_*") for i in set(["*_*".join(sorted([i,j])) for i,j in itertools.product(author_dict.keys(),author_dict.keys()) if i!=j])]
diff_keys_sample = random.sample(different_keys,int(DATA_SIZE/2) if int(DATA_SIZE/2) < len(different_keys) else len(different_keys))
different_sample = [[random.choice(author_dict[i[0]]),random.choice(author_dict[i[1]])] for i in diff_keys_sample]
print("Different Sample generated")

## data ##
similar_sample_data = [subtract_images(images[onlyfiles.index(i[0])],images[onlyfiles.index(i[1])]) for i in similar_sample]
different_sample_data = [subtract_images(images[onlyfiles.index(i[0])],images[onlyfiles.index(i[1])]) for i in different_sample]
print("Subtracted data generated")

## Save subtracted data ##
"""
subtracted_directory = "Subtracted/"
if not os.path.exists(subtracted_directory):
    os.makedirs(subtracted_directory)
save = [cv2.imwrite(subtracted_directory+str(i)+ "___" +similar_sample[i][0].split(".")[0]+"__"+similar_sample[i][1],similar_sample_data[i]) for i in range(len(similar_sample_data))]
save = [cv2.imwrite(subtracted_directory+str(50000+i)+ "___" +different_sample[i][0].split(".")[0]+"__"+different_sample[i][1],different_sample_data[i]) for i in range(len(different_sample_data))]
print("Subtracted data saved")
"""

## Train test split ##
data = {}
train_data = {}
test_data = {}
data["x"] = np.array(similar_sample_data + different_sample_data,dtype='float32')/255.0
data["y"] = np.array([1 for i in range(50000)] + [0 for i in range(50000)])
train_data["x"], test_data["x"], train_data["y"], test_data["y"] = train_test_split(data["x"], data["y"], test_size=0.2)
print("Split data into training and test")

## Time taken ##
print(time.time() - start, "seconds")


# In[9]:


### Reference: https://www.tensorflow.org/tutorials/layers ###

### Model function that trains and evaluates the model ###
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, INPUT_LAYER_SHAPE_X, INPUT_LAYER_SHAPE_Y, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=LAYER_1_FILTERS,
      kernel_size=KERNEL_SIZE,
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=POOLING_SIZE, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=LAYER_2_FILTERS,
      kernel_size=KERNEL_SIZE,
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=POOLING_SIZE, strides=2)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * LAYER_2_FILTERS])
    dense = tf.layers.dense(inputs=pool2_flat, units=DENSE_LAYER_UNITS, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=DROPOUT, training=mode == tf.estimator.ModeKeys.TRAIN)
 
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# In[ ]:


mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)
tensors_to_log = {"probabilities": "softmax_tensor"}

### Logging to save progress. Checkpointing to restart from whenever necessary ###
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

### Feeding data to the model for running ###
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data["x"]},
    y=train_data["y"],
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    shuffle=True)

### Training starts ###
start_time = time.time()
mnist_classifier.train(input_fn=train_input_fn,steps=TRAIN_STEPS,hooks=[logging_hook])
print("Time taken to train =", float((time.time() - start_time)/60.0), "minutes")


# In[ ]:


##### Training complete #####


# In[ ]:


### Testing the trained model ###
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data["x"]},
    y=test_data["y"],
    num_epochs=10,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(" ")

##Printing accuracy
print("Accuracy", eval_results["accuracy"])

## printing loss
print("Loss",eval_results["loss"])


# In[ ]:


"""
# import zipfile
# zip_ref = zipfile.ZipFile("AND_dataset.zip", 'r')
# zip_ref.extractall(os.getcwd())
# zip_ref.close()
print("Starting")
#Directory locations for the code
image_path = "Data\\img_align_celeba"

#Cropped Output
output_path = "Data\\cropped_faces"
if not os.path.exists(output_path):
        os.mkdir(output_path)
        
output_labels = "Data\\glasses.csv"

### place to save the model ###
MODEL_DIR = "tmp/mnist_convnet_model"

def import_data(start,end):
    #Eyeglasses label
    feature = pd.read_csv(output_labels)
    eye_glasses = feature[["Images","Eyeglasses"]]

    ### Changing -1 to 0 in the labels ###
    Y_labels = list(feature.Eyeglasses)
    Y_labels = np.array([i if i == 1 else 0 for i in Y_labels])[start:end]

    #Importing images as numpy objects
    SCALE = "L"
    X_REZ = 28
    Y_REZ = 28

    ### Import and resize image ###
    def resizer(path,scale="L",resize_x=28,resize_y=28):
        return np.array(Image.open(path).convert(scale).resize((resize_x,resize_y), Image.ANTIALIAS)).ravel().tolist()

    ### List of path to cropped images
    dirs_list = glob.glob(output_path+"\\*.jpg")[start:end]

    ### Import all images in a directory ###
    train_img = np.array([resizer(i,SCALE,X_REZ,Y_REZ) for i in dirs_list],dtype='float32')/255.0

    ### dictionary of images and their labels ###
    data = {}
    data["x"] = train_img
    data["y"] = Y_labels

    return data

#### Range of images to be imported to train ####
data_length = 2000

data_start_index = 0
data_end_index = data_start_index + data_length
### importing training data ###
data = import_data(data_start_index,data_end_index)
print(data)
train_data = {}
test_data = {}

### Splitting the data into training and testing ###
train_data["x"], test_data["x"], train_data["y"], test_data["y"] = train_test_split(data["x"], data["y"], test_size=0.2)
"""

