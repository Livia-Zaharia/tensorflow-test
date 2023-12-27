import tensorflow as tf
from tensorflow_wavelets.Layers.DWT import DWT


class WaveletTransformLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(WaveletTransformLayer, self).__init__(**kwargs)
    self.wavelet_transform = DWT()

  def call(self, inputs):
    # Reshape the input tensor to a 3D tensor
    inputs = tf.reshape(inputs, (1, 1,(tf.sqrt(float(tf.shape(inputs)[-1]))), (tf.sqrt(float(tf.shape(inputs)[-1])))))
    coeffs = self.wavelet_transform(inputs)
    return coeffs

  def compute_output_shape(self, input_shape):
    # The output will be a list of tensors, one for each wavelet coefficient
    return [input_shape] * len(self.call(tf.zeros(input_shape)))