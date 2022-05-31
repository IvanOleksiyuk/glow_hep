
class ActivationNormalization(bijector.Bijector):
  """Bijector to implement Activation Normalization (ActNorm)."""

  def __init__(self, nchan, dtype=tf.float32, validate_args=False, name=None):
    parameters = dict(locals())

    self._initialized = tf.Variable(False, trainable=False)
    self._m = tf.Variable(tf.zeros(nchan, dtype))
    self._s = TransformedVariable(tf.ones(nchan, dtype), exp.Exp())
    self._bijector = invert.Invert(
        chain.Chain([
            scale.Scale(self._s),
            shift.Shift(self._m),
        ]))
    super(ActivationNormalization, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        parameters=parameters,
        name=name or 'ActivationNormalization')

  def _inverse(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse(y, **kwargs)

  def _forward(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward(x, **kwargs)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse_log_det_jacobian(y, 1, **kwargs)

  def _forward_log_det_jacobian(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward_log_det_jacobian(x, 1, **kwargs)

  def _maybe_init(self, inputs, inverse):
    """Initialize if not already initialized."""

    def _init():
      """Build the data-dependent initialization."""
      axis = prefer_static.range(prefer_static.rank(inputs) - 1)
      m = tf.math.reduce_mean(inputs, axis=axis)
      s = (
          tf.math.reduce_std(inputs, axis=axis) +
          10. * np.finfo(dtype_util.as_numpy_dtype(inputs.dtype)).eps)
      if inverse:
        s = 1 / s
        m = -m
      else:
        m = m / s
      with tf.control_dependencies([self._m.assign(m), self._s.assign(s)]):
        return self._initialized.assign(True)

    return tf.cond(self._initialized, tf.no_op, _init)


class ActnormLayer(FlowLayer):
    def __init__(
        self,
        name: str = "",
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        """
        An implementation of the actnorm layer:
                y = w * x + bias

        Args:
            scale: a scale parameter of the variance of the initial value of the
                log_scale parameter when using data dependent initialization.
                See get_ddi_init_ops for the implementation.
        """
        super().__init__(name=name, **kwargs)
        self._bias_layer = ActnormBiasLayer(input_shape=input_shape)
        self._scale_layer = ActnormScaleLayer(input_shape=input_shape, scale=scale)
        self._chain = ChainLayer([self._bias_layer, self._scale_layer])

    def get_ddi_init_ops(self, num_init_iterations: int = 0) -> tf.Operation:
        bias_update_op = self._bias_layer.get_ddi_init_ops(num_init_iterations)
        with tf.control_dependencies([bias_update_op]):
            scale_update_op = self._scale_layer.get_ddi_init_ops(num_init_iterations)
            update_ops = tf.group([bias_update_op, scale_update_op])
        return update_ops

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        return self._chain((x, logdet, z), forward=True, is_training=is_training)

    def backward(self, y, logdet, z, is_training: bool = True)-> FlowData:
        return self._chain((y, logdet, z), forward=False, is_training=is_training)









class ActnormLayer(FlowLayer):
    def __init__(
        self,
        name: str = "",
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        """
        An implementation of the actnorm layer:
                y = w * x + bias

        Args:
            scale: a scale parameter of the variance of the initial value of the
                log_scale parameter when using data dependent initialization.
                See get_ddi_init_ops for the implementation.
        """
        super().__init__(name=name, **kwargs)
        self._bias_layer = ActnormBiasLayer(input_shape=input_shape)
        self._scale_layer = ActnormScaleLayer(input_shape=input_shape, scale=scale)
        self._chain = ChainLayer([self._bias_layer, self._scale_layer])

    def get_ddi_init_ops(self, num_init_iterations: int = 0) -> tf.Operation:
        bias_update_op = self._bias_layer.get_ddi_init_ops(num_init_iterations)
        with tf.control_dependencies([bias_update_op]):
            scale_update_op = self._scale_layer.get_ddi_init_ops(num_init_iterations)
            update_ops = tf.group([bias_update_op, scale_update_op])
        return update_ops

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        return self._chain((x, logdet, z), forward=True, is_training=is_training)

    def backward(self, y, logdet, z, is_training: bool = True)-> FlowData:
        return self._chain((y, logdet, z), forward=False, is_training=is_training)