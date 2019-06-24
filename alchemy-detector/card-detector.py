import tensorflow as tf
import os
import numpy as np
import sys
from PIL import Image
import cv2



if len(sys.argv) < 2:
    cap = cv2.VideoCapture(0)
else:
    file = sys.argv[1]
    cap = cv2.VideoCapture(file)


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


if len(sys.argv) < 2:
    cap = cv2.VideoCapture(0)
else:
    file = sys.argv[1]
    cap = cv2.VideoCapture(file)

graph_def = tf.GraphDef()
labels = []

# These are set to the default names from exported models, update as needed.
filename = "data/model.pb"
labels_filename = "data/labels.txt"

# Import the TF graph
with tf.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())


while True:
    ret, image = cap.read()
    # If the image has either w or h greater than 1600 we resize it down respecting
    # aspect ratio such that the largest dimension is 1600
    image = resize_down_to_1600_max_dim(image)
    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)
    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)
    # Get the input size of the model
    with tf.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]

    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
    
    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    with tf.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            image_np_expanded = np.expand_dims(augmented_image, axis=0)
            
            predictions, = sess.run(prob_tensor, {input_node: image_np_expanded })
            print(predictions)

            # Print the highest probability label
            highest_probability_index = np.argmax(predictions)
            print('Classified as: ' + labels[highest_probability_index])
            print()

            cv2.rectangle(image,(95,35),(410,68),(0,0,199), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'Classified as: ' + labels[highest_probability_index],(100,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)
           
            # Or you can print out all of the results mapping labels to probabilities.
            label_index = 0
            for p in predictions:
                truncated_probablity = np.float64(np.round(p,8))
                print (labels[label_index], truncated_probablity)
                label_index += 1
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)


    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    cv2.imshow('object detection', cv2.resize(image, (800,600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break