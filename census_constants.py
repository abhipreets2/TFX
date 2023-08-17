
CATEGORICAL_FEATURE_KEYS = [
    'education', 'marital-status', 'occupation', 'race', 'relationship', 'workclass', 'sex', 'native-country'
]

NUMERIC_FEATURE_KEYS = [
    'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]

BUCKET_FEATURE_KEYS = [
    'age'
]

FEATURE_BUCKET_COUNT = {
    'age' : 4
}

LABEL_KEY = 'label'

def transformed_name(key):
    return key + '_xf'
