import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers

MAX_SCALE=2.
TANH_LINEARITY_SCALE=0.2

class AffineCouplingTanh(tfb.Bijector):
    """IOIOIOIOIOIOIOIOIOIOIOIOIO
    My implementation of the AffineCoupling bijector. Code inspired by many different implementaions
    """
    def __init__(self, input_shape, coupling_fn=None, name='affine_coupling', bayesian=False,  **kwargs):
        super(AffineCouplingTanh, self).__init__(
            inverse_min_event_ndims=3,
            is_constant_jacobian=False,
            name=name,
            **kwargs)
        
        self.inp_chans = input_shape[-1]//2
        self.cha_chans = input_shape[-1]-self.inp_chans
        nn_input_shape=(input_shape[0], input_shape[1], self.inp_chans)
        print(nn_input_shape)
        self.model = coupling_fn(nn_input_shape)

    def _forward(self, x):
        x = tf.cast(x, tf.float32)
        x1 = x[..., :self.inp_chans]
        x2 = x[..., self.inp_chans:]

        y1 = x1
        trafo_params = self.model(x1)
        log_scale=trafo_params[..., :self.cha_chans]
        translation=trafo_params[..., self.cha_chans:]
        log_scale = MAX_SCALE * tf.tanh(TANH_LINEARITY_SCALE * log_scale)
        y2 = x2 * tf.exp(log_scale) + translation
        return tf.concat([y1, y2], axis=-1)

    def _inverse(self, y):
        y = tf.cast(y, tf.float32)
        y1 = y[..., :self.inp_chans]
        y2 = y[..., self.inp_chans:]

        x1 = y1
        trafo_params = self.model(y1)
        log_scale = trafo_params[..., :self.cha_chans]
        translation = trafo_params[..., self.cha_chans:]
        
        log_scale = MAX_SCALE * tf.tanh(TANH_LINEARITY_SCALE * log_scale)
        x2 = (y2 - translation) * tf.exp(-log_scale)
        return tf.concat([x1, x2], axis=-1)

    def _inverse_log_det_jacobian(self, y):
        trafo_params = self.model(y[..., :self.inp_chans])
        log_scale = trafo_params[..., :self.cha_chans]
        log_scale = MAX_SCALE * tf.tanh(TANH_LINEARITY_SCALE * log_scale)
        log_det = -tf.reduce_sum(log_scale, axis=(-1, -2, -3))
        #further reduction is done via reduce_jacobian_det_over_shape inhereted by all bijectors
        return log_det

    def _forward_log_det_jacobian(self, x):
        trafo_params = self.model(x[..., :self.inp_chans])
        log_scale = trafo_params[..., :self.cha_chans]
        log_scale = MAX_SCALE * tf.tanh(TANH_LINEARITY_SCALE * log_scale)
        log_det = tf.reduce_sum(log_scale, axis=(-1, -2, -3))
        #further reduction is done via reduce_jacobian_det_over_shape inhereted by all bijectors
        return log_det