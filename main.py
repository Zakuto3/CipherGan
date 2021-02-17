import tensorflow as tf
import PIL
import imageio
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2 as cv2
from models import CVAE

localpath = "C:/Users/Zak/Documents/Code/AIChan/imgGan/CipherGan"
shapeSizex = 512
shapeSizey = 512
datasetSize= 30
train_size = int(datasetSize/2)
test_size = int(datasetSize/2)
batch_size = 6
num_examples_to_generate = 6
epochs = 150
latent_dim = 2
list_ = []
train_images=[]
test_images=[]

def read_img(img_list, img):
    n = cv2.imread(img, 0)
    img_list.append(n)
    return img_list

path = glob.glob(localpath+"/TestDataset/*.jpg") #or jpg
cv_image = [read_img(list_, img) for img in path]


for img in range(len(cv_image)):
    if img < test_size:
        train_images.append(cv_image[0][img])
    else:
        test_images.append(cv_image[0][img])
        
train_images, test_images = np.array(train_images),np.array(test_images)
# plt.imshow(train_images[0])
#DATA region END

def preprocess_images(images):
  images = images.reshape((images.shape[0], shapeSizex, shapeSizey, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
.shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
.shuffle(test_size).batch(batch_size))

#print(train_dataset[0].shape)

#TODO: Change to Compilable model
optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# this will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim]) 
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)

  fig = plt.figure(figsize=(4, 4))
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')


  plt.axis('off')
  # plt.imshow(predictions[0,:,:,0])#Single img show

  plt.savefig(localpath+'/genimg/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

#get some samples from test to gen output
assert batch_size >= num_examples_to_generate
test_sample = 0
for test_batch in test_dataset.take(1): #Changed for full img
  # test_sample = test_batch[0:num_examples_to_generate, :, :, :]
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]


generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  # display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)
