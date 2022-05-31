import functools

import tensorflow.compat.v2 as tf

tfk = tf.keras
tfkl = tfk.layers

class GlowDefaultNetwork(tfk.Sequential):
  """Default network for the glow bijector.

  This builds a 3 layer convolutional network, with relu activation functions
  and he_normal initializer. The first and third layers have default kernel
  shape of 3, and the second layer is a 1x1 convolution. This is the setup
  in the public version of Glow.

  The output of the convolutional network defines the components of an Affine
  transformation (i.e. y = m * x + b), where m, x, and b are all tensors of
  the same shape, and * indicates elementwise multiplication.
  """

  def __init__(self, input_shape, num_hidden=400, kernel_shape=3):
    """Default network for glow bijector."""
    # Default is scale and shift, so 2c outputs.
    this_nchan = input_shape[-1] * 2
    conv_last = functools.partial(
        tfkl.Conv2D,
        padding='same',
        kernel_initializer=tf.initializers.zeros(),
        bias_initializer=tf.initializers.zeros())
    super(GlowDefaultNetwork, self).__init__([
        tfkl.Input(shape=input_shape),
        tfkl.Conv2D(num_hidden, kernel_shape, padding='same',
                    kernel_initializer=tf.initializers.he_normal(),
                    activation='relu'),
        tfkl.Conv2D(num_hidden, 1, padding='same',
                    kernel_initializer=tf.initializers.he_normal(),
                    activation='relu'),
        conv_last(this_nchan, kernel_shape)
    ])


class GlowDefaultExitNetwork(tfk.Sequential):
  """Default network for the glow exit bijector.

  This is just a single convolutional layer.
  """

  def __init__(self, input_shape, output_chan, kernel_shape=3):
    """Default network for glow bijector."""
    # Default is scale and shift, so 2c outputs.
    this_nchan = output_chan * 2
    conv = functools.partial(
        tfkl.Conv2D,
        padding='same',
        kernel_initializer=tf.initializers.zeros(),
        bias_initializer=tf.initializers.zeros())

    super(GlowDefaultExitNetwork, self).__init__([
        tfkl.Input(input_shape),
        conv(this_nchan, kernel_shape)])
    
REG=1e-3
class ResnetGlowNetwork(tfk.Model):
    """IOIOIOIOIOIOIOIOIOIOIOIOIO
    Resnet architecture simmilar to one in the kmkolasinski implementation 
    """

    def __init__(self, input_shape, num_hidden=None, kernel_shape=3, resnet_blocks=3):
        """Default network for glow bijector."""
        # Default is scale and shift, so 2c outputs.
        this_nchan = input_shape[-1] * 2
        inputs = tfkl.Input(shape=input_shape)
        x=inputs
        #first two layers to scale to num_hidden
        
        if num_hidden is None:
            num_hidden=this_nchan
        
        x=tfkl.Conv2D(num_hidden, kernel_shape, padding='same',
                        kernel_initializer=tf.initializers.he_normal(),
                        activation='relu',
                        kernel_regularizer=tfk.regularizers.L2(REG),
                        bias_regularizer=tfk.regularizers.L2(REG))(x)
        x=tfkl.Conv2D(num_hidden, 1, padding='same',
                        kernel_initializer=tf.initializers.he_normal(),
                        activation='relu')(x)
        #blocks with skip connections
        for i in range(resnet_blocks-1):
            x_input=x
            x=tfkl.Conv2D(num_hidden, kernel_shape, padding='same',
                        kernel_initializer=tf.initializers.he_normal(),
                        activation='relu',
                        kernel_regularizer=tfk.regularizers.L2(REG),
                        bias_regularizer=tfk.regularizers.L2(REG))(x_input)
            x=tfkl.Conv2D(num_hidden, 1, padding='same',
                        kernel_initializer=tf.initializers.he_normal(),
                        activation='relu',
                        kernel_regularizer=tfk.regularizers.L2(REG),
                        bias_regularizer=tfk.regularizers.L2(REG))(x)
            x = tfkl.Add()([x, x_input])
        
        outputs=tfkl.Conv2D(this_nchan, kernel_shape, padding='same',
                    kernel_initializer=tfk.initializers.VarianceScaling(scale=0.0001),
                    bias_initializer=tfk.initializers.VarianceScaling(scale=0.0001),
                    kernel_regularizer=tfk.regularizers.L2(REG),
                    bias_regularizer=tfk.regularizers.L2(REG))(x)        
        
        super(ResnetGlowNetwork, self).__init__(inputs, outputs)
        
        
class Resnet2GlowNetwork(tfk.Model):
    """IOIOIOIOIOIOIOIOIOIOIOIOIO
    Resnet architecture simmilar to one in the kmkolasinski implementation 
    """

    def __init__(self, input_shape, output_nchan=None, num_filters=None, kernel_shape=3, resnet_blocks=3, units_factor=2, skip_connection=True):
        """Default network for glow bijector."""
        # Default is scale and shift, so 2c outputs.
        if output_nchan is None:
            output_nchan = input_shape[-1] * 2
        inputs = tfkl.Input(shape=input_shape)
        x=inputs
        
        if num_filters is None:
            num_filters=output_nchan*units_factor
        
        #blocks with skip connections
        for i in range(resnet_blocks):
            x_input=x
            
            x = tfkl.Conv2D(num_filters, kernel_shape, padding='same',
                        kernel_initializer=tf.initializers.he_normal(),
                        activation=None,
                        kernel_regularizer=tfk.regularizers.L2(REG),
                        bias_regularizer=tfk.regularizers.L2(REG))(x_input)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.Conv2D(num_filters, 1, padding='same',
                        kernel_initializer=tf.initializers.he_normal(),
                        activation=None,
                        kernel_regularizer=tfk.regularizers.L2(REG),
                        bias_regularizer=tfk.regularizers.L2(REG))(x)
            
            
            if skip_connection:
                if num_filters != input_shape[-1]:
                    x_input = tfkl.Conv2D(num_filters, kernel_size=1,
                        kernel_initializer=tf.initializers.he_normal(),
                        activation=None,
                        kernel_regularizer=tfk.regularizers.L2(REG),
                        bias_regularizer=tfk.regularizers.L2(REG))(x_input)
                    x_input = tfkl.PReLU()(x_input)
                x = tfkl.Add()([x, x_input])
            x = tfkl.LeakyReLU()(x)
        
        outputs=tfkl.Conv2D(output_nchan, kernel_shape, padding='same',
                    kernel_initializer=tfk.initializers.VarianceScaling(scale=0.0001),
                    bias_initializer=tfk.initializers.VarianceScaling(scale=0.0001),
                    kernel_regularizer=tfk.regularizers.L2(REG),
                    bias_regularizer=tfk.regularizers.L2(REG))(x)        
        
        super(Resnet2GlowNetwork, self).__init__(inputs, outputs)
        
    