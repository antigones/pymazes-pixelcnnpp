# Build a small Pixel CNN++ model to train on maze dataset.

import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from image_utils import read_images, sample_to_image
import maze_utils
import numpy as np
from numpy.random.mtrand import seed
import random as rd

def binary_tree(size, label):
  n = 1
  p = 0.5
  grid = np.random.binomial(n, p, size=(size, size))
  grid = maze_utils.preprocess_grid(grid, size)
  output = maze_utils.carve_maze(grid, size)
  s = ""
  img_out = []
  for elm in output:
      s = "".join(elm)
      img_out.append(list(map(int, s.replace("#", "1").replace(" ", "0"))))
  arr = np.asarray(img_out)
  arr[arr == 0] = 0
  arr[arr == 1] = 255

  # add 2 rows, 2 columns to achieve 28*28 images
  size_new = len(output)
  z = np.zeros((size_new, 1))
  z[z == 0] = 255
  arr = np.append(arr, z, axis=1)
  w = np.zeros((1, size_new + 1 ))
  w[w == 0] = 255
  arr = np.append(arr, w, axis=0)
  return arr[:, :, np.newaxis], label

def ald(size: int, label:int):
  grid = np.zeros(shape=(size,size))
  output_grid = np.empty([size*3, size*3], dtype=str)
  output_grid[:] = '#'
  c = size * size  # number of cells to be visited
  i = rd.randrange(size)
  j = rd.randrange(size)
  while np.count_nonzero(grid) < c:
      # visit this cell
      grid[i, j] = 1
      w = i*3 + 1
      k = j*3 + 1
      output_grid[w, k] = ' '
      can_go = [1, 1, 1, 1]
      if i == 0:
          can_go[0] = 0
      if i == size-1:
          can_go[2] = 0
      if j == 0:
          can_go[3] = 0
      if j == size-1:
          can_go[1] = 0
      #  it makes sense to choose neighbour among available directions
      neighbour_idx = np.random.choice(np.nonzero(can_go)[0])  # n,e,s,w
      if neighbour_idx == 0:
          #  has been visited?
          if grid[i-1, j] == 0:
              #  goto n
              output_grid[w-1, k] = ' '
              output_grid[w-2, k] = ' '
          i -= 1

      if neighbour_idx == 1:
          if grid[i, j+1] == 0:
              #  goto e
              output_grid[w, k+1] = ' '
              output_grid[w, k+2] = ' '
          j += 1

      if neighbour_idx == 2:
          if grid[i+1, j] == 0:
              #  goto s
              output_grid[w+1, k] = ' '
              output_grid[w+2, k] = ' '
          i += 1

      if neighbour_idx == 3:
          if grid[i, j-1] == 0:
              #  goto w
              output_grid[w, k-1] = ' '
              output_grid[w, k-2] = ' '
          j -= 1
  
  output = output_grid
  s = ""
  img_out = []
  for elm in output:
      s = "".join(elm)
      img_out.append(list(map(int, s.replace("#", "1").replace(" ", "0"))))
  arr = np.asarray(img_out)
  arr[arr == 0] = 0
  arr[arr == 1] = 255

  # add 2 rows, 2 columns to achieve 28*28 images
  size_new = len(output)
  z = np.zeros((size_new, 1))
  z[z == 0] = 255
  arr = np.append(arr, z, axis=1)
  w = np.zeros((1, size_new + 1 ))
  w[w == 0] = 255
  arr = np.append(arr, w, axis=0)
  return arr[:, :, np.newaxis], label


def gen_labyrinth(size):
  n_label = 2
  i = 0
  while True:
    # np.random.seed(0)
    if (i % n_label == 0):
      yield binary_tree(size, 0)
    else:
      yield ald(size, 1)
    i = i+1

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

# train_data = read_images("\\imgs\\train\\")
# test_data = read_images("\\imgs\\test\\")
batch_size = 32
maze_size = 9
# train_data = tf.keras.utils.image_dataset_from_directory(".\imgs\\",image_size=(28, 28), shuffle=True, batch_size=batch_size)
train_data = tf.data.Dataset.from_generator(
  lambda: gen_labyrinth(maze_size), 
  output_signature=(
    tf.TensorSpec(shape=(maze_size*3+1, maze_size*3+1, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32))
  )

def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  return (x['image'],)  # (input, output) of the model


# train_it = train_data.map(image_preprocess).batch(batch_size).shuffle(1000)
train_it = train_data.batch(batch_size)

image_shape = (28, 28, 1)
# Define a Pixel CNN network
dist = tfd.PixelCNN(
    image_shape=image_shape,
    num_resnet=2,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=2, #B/W
    dropout_p=.3,
)

# Define the model input
image_input = tfkl.Input(shape=image_shape)

# Define the log likelihood for the loss fn
log_prob = dist.log_prob(image_input)

# Define the model
model = tfk.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))

# Compile and train the model
model.compile(
    optimizer=tfk.optimizers.Adam(),
    metrics=[])

model.fit(train_it, epochs=10, steps_per_epoch=100, verbose=True)

# sample five images from the trained model
samples = dist.sample(5)

for i, sample in enumerate(samples):
  tf.keras.preprocessing.image.save_img('sample'+str(i)+'.gif',sample)