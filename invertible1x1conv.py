import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Tuple, List, Callable
from tensorflow_probability.python.bijectors import scale_matvec_lu
import scipy.linalg as salg

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers

class Invertible1x1conv(tfb.Bijector):
    def __init__(self, input_shape, name = "1x1conv", use_lu_decomposition: bool = True, **kwargs):
        """IOIOIOIOIOIOIOIOIOIOIOIOIO
        Reimplementation of the Invertible 1D convolution see Glow paper. for TF2_probability.bijector api
        """
        self._use_lu_decomposition = use_lu_decomposition
        self._input_shape = input_shape
        # only trainable weights
        self._weights: List[tf.Variable] = []
        # only non-trainable weights
        self._non_trainable_weights: List[tf.Variable] = []
        self._kernel_t: tf.Tensor = None
        self._inv_kernel_t: tf.Tensor = None
        self._dlogdet_t: tf.Tensor = None
        self.build()
        super(Invertible1x1conv, self)(is_constant_jacobian=False,
              forward_min_event_ndims=3,
              name=name).__init__(name=name, **kwargs)

    def build(self):
        dtype = "float64"
        shape = self._input_shape
        num_channels = shape[3]
        w_shape = [num_channels, num_channels]
        kernel_shape = [1, 1, num_channels, num_channels]
        # Sample a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype("float32")

        if not self._use_lu_decomposition:

            w = tf.get_variable("kernel", dtype=tf.float32, initializer=w_init)
            dlogdet = (
                tf.cast(
                    tf.log(tf.abs(tf.matrix_determinant(tf.cast(w, dtype)))), "float32"
                )
                * shape[1]
                * shape[2]
            )

            self._weights = [w]
            self._dlogdet_t = dlogdet
            self._kernel_t = tf.reshape(w, kernel_shape)
            self._inv_kernel_t = tf.reshape(tf.matrix_inverse(w), kernel_shape)

        else:
            np_p, np_l, np_u = salg.lu(w_init, permute_l=False)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p_mat = tf.get_variable("P_mat", initializer=np_p, trainable=False)
            l_mat = tf.get_variable("L_mat", initializer=np_l)
            sign_s = tf.get_variable("sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            u_mat = tf.get_variable("U_mat", initializer=np_u)

            self._weights = [l_mat, log_s, u_mat]
            self._non_trainable_weights = [p_mat, sign_s]

            p_mat = tf.cast(p_mat, dtype)
            l_mat = tf.cast(l_mat, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u_mat = tf.cast(u_mat, dtype)

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l_mat = l_mat * l_mask + tf.eye(*w_shape, dtype=dtype)
            u_mat = u_mat * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p_mat, tf.matmul(l_mat, u_mat))

            # inverse w
            u_inv = tf.matrix_inverse(u_mat)
            l_inv = tf.matrix_inverse(l_mat)
            p_inv = tf.matrix_inverse(p_mat)
            w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)
            self._dlogdet_t = tf.reduce_sum(log_s) * shape[1] * shape[2]
            self._kernel_t = tf.reshape(w, kernel_shape)
            self._inv_kernel_t = tf.reshape(w_inv, kernel_shape)
    
    def _forward(self, x):
        y = tf.nn.conv2d(x, self._kernel_t, [1, 1, 1, 1], "SAME", data_format="NHWC")
        return y

    def _inverse(self, y):
        x = tf.nn.conv2d(y, self._inv_kernel_t, [1, 1, 1, 1], "SAME", data_format="NHWC")
        return y

    def _forward_log_det_jacobian(self, x):
        return self._dlogdet_t
    
    def _inverse_log_det_jacobian(self, y):
        return -self._dlogdet_t
    
#
class OneByOneConv(tfb.Bijector):
  """The 1x1 Conv bijector used in Glow.

  This class has a convenience function which initializes the parameters
  of the bijector.
  """

  def __init__(self, event_size, seed=None, dtype=tf.float32,
               name='OneByOneConv', **kwargs):
    parameters = dict(locals())
    with tf.name_scope(name) as bijector_name:
      lower_upper, permutation = self.trainable_lu_factorization(
          event_size, seed=seed, dtype=dtype)
      self._bijector = scale_matvec_lu.ScaleMatvecLU(
          lower_upper, permutation, **kwargs)
      super(OneByOneConv, self).__init__(
          dtype=self._bijector.lower_upper.dtype,
          is_constant_jacobian=True,
          forward_min_event_ndims=1,
          parameters=parameters,
          name=bijector_name)

  def forward(self, x):
    return self._bijector.forward(x)

  def inverse(self, y):
    return self._bijector.inverse(y)

  def inverse_log_det_jacobian(self, y, event_ndims=None):
    return self._bijector.inverse_log_det_jacobian(y, event_ndims)

  def forward_log_det_jacobian(self, x, event_ndims=None):
    return self._bijector.forward_log_det_jacobian(x, event_ndims)

  @staticmethod
  def trainable_lu_factorization(event_size,
                                 seed=None,
                                 dtype=tf.float32,
                                 name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
      event_size = tf.convert_to_tensor(
          event_size, dtype_hint=tf.int32, name='event_size')
      random_matrix = tf.random.uniform(
          shape=[event_size, event_size],
          dtype=dtype,
          seed=seed)
      random_orthonormal = tf.linalg.qr(random_matrix)[0]
      lower_upper, permutation = tf.linalg.lu(random_orthonormal)
      lower_upper = tf.Variable(
          initial_value=lower_upper, trainable=True, name='lower_upper')
      # Initialize a non-trainable variable for the permutation indices so
      # that its value isn't re-sampled from run-to-run.
      permutation = tf.Variable(
          initial_value=permutation, trainable=False, name='permutation')
      return lower_upper, permutation