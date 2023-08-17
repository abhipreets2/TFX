
DENSE_FLOAT_FEATURE_KEYS = ['temp', 'snow_1h']
BUCKET_FEATURE_KEYS = ['rain_1h']
FEATURE_BUCKET_COUNT = {'rain_1h': 3}
RANGE_FEATURE_KEYS = ['clouds_all']

VOCAB_SIZE = 1000
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = [
    'holiday',
    'weather_main',
    'weather_description'
]

CATEGORICAL_FEATURE_KEYS = [
    'hour', 'day', 'day_of_week', 'month'
]

VOLUME_KEY = 'traffic_volume'

def transformed_name(key):
    return key + '_xf'
