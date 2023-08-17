
import tensorflow as tf
import tensorflow_transform as tft

import traffic_constants

_DENSE_FLOAT_FEATURE_KEYS = traffic_constants.DENSE_FLOAT_FEATURE_KEYS
_RANGE_FEATURE_KEYS = traffic_constants.RANGE_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = traffic_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = traffic_constants.VOCAB_SIZE
_OOV_SIZE = traffic_constants.OOV_SIZE
_CATEGORICAL_FEATURE_KEYS = traffic_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = traffic_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = traffic_constants.FEATURE_BUCKET_COUNT
_VOLUME_KEY = traffic_constants.VOLUME_KEY
_transformed_name = traffic_constants.transformed_name

def preprocessing_fn(inputs):
    '''
    tf.transform's callback function for preprocessing inputs
    
        Parameters:
            inputs : map from feature keys to raw not-yet-transformed features
        Returns:
            outputs : Map from string feature key to transformed feature operations.
    '''
    
    outputs = {}
    
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
        
    for key in _RANGE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])
    
    for key in _VOCAB_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key], top_k=_VOCAB_SIZE, num_oov_buckets=_OOV_SIZE)
        
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(inputs[key], _FEATURE_BUCKET_COUNT[key])
        
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = inputs[key] 
    
   
    traffic_volume = tf.cast(inputs[_VOLUME_KEY], tf.float32)
    
    outputs[_transformed_name(_VOLUME_KEY)] = tf.cast(
        tf.greater(tft.mean(inputs[_VOLUME_KEY]), traffic_volume),
        tf.int64
    )
    
    return outputs
