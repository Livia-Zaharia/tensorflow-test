import tensorflow as tf

from tensorflow_wavelets.Layers.DWT import DWT


class WaveletTransformLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(WaveletTransformLayer, self).__init__(**kwargs)
    self.wavelet_transform = DWT()

  def call(self, inputs):
    
    # Calculate padding size
    padding_size = tf.cast(tf.math.pow(tf.math.ceil(tf.math.sqrt(float(tf.shape(inputs)[-1]))), 2) - tf.cast(tf.shape(inputs)[-1], tf.float32), tf.int32)
    # Add padding to inputs
    
    original_inputs=inputs

    inputs = tf.pad(inputs, [[0, 0], [0, padding_size]])
    inputs = tf.reshape(inputs, (1, 1,(tf.sqrt(float(tf.shape(inputs)[-1]))), (tf.sqrt(float(tf.shape(inputs)[-1])))))
    coeffs = self.wavelet_transform(inputs)
    
    #should be of shape [46,46] ideea e sa scot padding si sa le aplic pe cele doua ca filtre hopefully
    filter1=coeffs[0][0]
    filter2=coeffs[0][1]
    
    filter1=tf.reshape(filter1,(23,92))
    filter2=tf.reshape(filter2,(23,92))
    
    filter1=filter1[:,2:-1]
    filter2=filter2[:,2:-1]
    
    
    filter1 = tf.reshape(filter1, [-1])
    filter2 = tf.reshape(filter2, [-1])
    
    new_element = tf.constant([0.0]) # The new element you want to add

    # Add an extra dimension to the tensor
    tensor_extended = tf.expand_dims(filter1, axis=0)
    new_element = tf.expand_dims(new_element, axis=0)
    
    # Append the new element to the tensor
    tensor_extended = tf.concat([tensor_extended, new_element],axis=1)
    filter1=tensor_extended
    
    
    # Add an extra dimension to the tensor
    tensor_extended = tf.expand_dims(filter2, axis=0)

    # Append the new element to the tensor
    tensor_extended = tf.concat([tensor_extended, new_element], axis=1)
    filter2=tensor_extended
    
    #filter1 = filter1(original_inputs)
    #filter2 = filter2(original_inputs)
    
    return filter1,filter2

  def compute_output_shape(self, input_shape):
    # The output will be a list of tensors, one for each wavelet coefficient
    return [input_shape] * len(self.call(tf.zeros(input_shape)))