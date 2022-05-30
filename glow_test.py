#%%All important imports and shortcut names
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from operator import mul
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tfglow
from nets import GlowDefaultNetwork, GlowDefaultExitNetwork, GlowDefaultExitNetwork, ResnetGlowNetwork

tfb = tfp.bijectors
tfd = tfp.distributions

#%%load MNIST data and do some preprocessing 
data, info = tfds.load('mnist', with_info=True)
train_data, test_data = data['train'], data['test']

preprocess = lambda x: tf.cast((tf.image.resize(x['image'], (32, 32))+tf.random.uniform((32, 32, 32, 1))-0.5)/255, tf.float32)
train_data = train_data.batch(32).map(preprocess)
test_data = test_data.batch(32).map(preprocess)

x = next(iter(train_data))
plt.imshow(x[4])

#%%define and test our bijector
glow = tfglow.Glow(output_shape=[32, 32, 1],
                coupling_bijector_fn=ResnetGlowNetwork,
                exit_bijector_fn=tfb.GlowDefaultExitNetwork,
                num_glow_blocks=3,
                num_steps_per_block=6)

z_shape = glow.inverse_event_shape([32, 32, 1])

pz = tfd.Sample(tfd.Normal(0., 1.), z_shape)

# Calling glow on distribution p(z) creates our glow distribution over images.
px = glow(pz)

# Take samples from the distribution to get images from your dataset
images = px.sample(2)

plt.imshow(images[0])
# Map images to positions in the distribution
z = glow.inverse(x)
print(tf.reduce_mean(tf.reduce_sum(tf.math.pow(glow.inverse(x), 2), axis=(1))/2))
print(glow.inverse_log_det_jacobian(x))
print(tf.reduce_mean(glow.inverse_log_det_jacobian(x)))