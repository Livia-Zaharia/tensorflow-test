import tensorflow as tf
from wavetf import WaveTFFactory

class WaveletTransformLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
      super(WaveletTransformLayer, self).__init__(**kwargs)
      self.wavelet_transform = WaveTFFactory().build('db2', dim=2)

  def call(self, inputs):
      coeffs = self.wavelet_transform.call(inputs)
      coeffs = [tf.expand_dims(coeff, axis=-1) for coeff in coeffs]
      return coeffs

  def compute_output_shape(self, input_shape):
      # The output will be a list of tensors, one for each wavelet coefficient
      return [input_shape] * len(self.call(tf.zeros(input_shape)))