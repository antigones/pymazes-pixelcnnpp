# Build a small Pixel CNN++ model to train on MNIST.

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from image_utils import read_images, sample_to_image

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

# Load MNIST from tensorflow_datasets
# data = tfds.load('mnist')
# train_data, test_data = data['train'], data['test']
# print(train_data)

train_data = read_images("\\imgs\\train\\")
test_data = read_images("\\imgs\\test\\")


def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  return (x['image'],)  # (input, output) of the model

batch_size = 16
#train_it = train_data.map(image_preprocess).batch(batch_size).shuffle(1000)
train_it = train_data

image_shape = (28, 28, 1)
# Define a Pixel CNN network
dist = tfd.PixelCNN(
    image_shape=image_shape,
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
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
    optimizer=tfk.optimizers.Adam(.001),
    metrics=[])

model.fit(train_it, epochs=50, verbose=True)

# sample five images from the trained model
samples = dist.sample(5)

for i, sample in enumerate(samples):
  tf.keras.preprocessing.image.save_img('sample'+str(i)+'.gif',sample)