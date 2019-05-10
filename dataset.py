import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import csv

import random
from datetime import datetime

random.seed(datetime.now())

def read_data_set(data_set_info_file_path):
    image_names = []
    mask_names = []

    images_fns = glob.glob(os.path.join(data_set_info_file_path, 'sample*.png'))
    images_fns = [ x for x in images_fns if 'mask' not in x]

    for x in images_fns:
        image_names.append(x)
        mask_names.append(os.path.splitext(x)[0] + '_mask.png')
            
    return image_names, mask_names


def read_train_validation_big_sets(train_path, validation_path, image_size):
    class BigDataSets(object):
        pass
    
    train_images_paths, train_masks_paths = read_data_set(train_path)
    train_images_paths, train_masks_paths = shuffle(train_images_paths, train_masks_paths)
    
    validation_images_paths, validation_masks_paths = read_data_set(validation_path)
    validation_images_paths, validation_masks_paths = shuffle(validation_images_paths, validation_masks_paths)
    
    data_sets = BigDataSets()
    data_sets.train = BigDataSet(train_images_paths, train_masks_paths, image_size)
    data_sets.valid = BigDataSet(validation_images_paths, validation_masks_paths, image_size)
    
    return data_sets

def load_images(image_paths, image_size):
    images = []

    for fl in image_paths:
        image = cv2.imread(fl, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
    images = np.array(images)

    return images

def load_masks(image_paths, image_size):
    images = []

    for fl in image_paths:
        image = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
    images = np.array(images)

    return images

class BigDataSet(object):
  def __init__(self, image_paths, masks_paths, image_size):
    self._num_examples = len(image_paths)

    self._image_paths = image_paths
    self._masks_paths = masks_paths
    self._epochs_done = 0
    self._index_in_epoch = 0
    self._image_size = image_size

  @property
  def image_paths(self):
    return self._image_paths

  @property
  def masks_paths(self):
    return self._masks_paths

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    images = load_images(self._image_paths[start:end], self._image_size)
    masks = load_masks(self._masks_paths[start:end], self._image_size)

    return images, masks



