import tensorflow as tf
import numpy as np
import pandas as pd


BATCH_SIZE = 16
IMAGE_SIZE = 28

def _parse_function(filename, label):
    path = "../data_location/data/" + filename + ".jpg"
    image_string = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
    return image_resized, label

df = pd.read_csv("../train.csv")
filenames = tf.constant(df["id"])
labels = tf.constant(df["landmark_id"])
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function).batch(BATCH_SIZE).repeat()

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

print(dataset)

