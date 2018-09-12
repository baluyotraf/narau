import tensorflow as tf


# noinspection PyPep8Naming
def Int64Feature(value):
    lst = tf.train.Int64List(value=value)
    return tf.train.Feature(int64_list=lst)


# noinspection PyPep8Naming
def BytesFeature(value):
    lst = tf.train.BytesList(value=value)
    return tf.train.Feature(bytes_list=lst)


# noinspection PyPep8Naming
def FloatFeature(value):
    lst = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list=lst)


# noinspection PyPep8Naming
def Int64FeatureList(values):
    feature = [Int64Feature(v) for v in values]
    return tf.train.FeatureList(feature=feature)


# noinspection PyPep8Naming
def BytesFeatureList(values):
    feature = [BytesFeature(v) for v in values]
    return tf.train.FeatureList(feature=feature)


# noinspection PyPep8Naming
def FloatFeatureList(values):
    feature = [FloatFeature(v) for v in values]
    return tf.train.FeatureList(feature=feature)


# noinspection PyPep8Naming
def FeatureLists(feature_dict):
    return tf.train.FeatureLists(feature_list=feature_dict)


# noinspection PyPep8Naming
def Features(feature_dict):
    return tf.train.Features(features=feature_dict)


# noinspection PyPep8Naming
def Example(features):
    return tf.train.Example(features=features)


# noinspection PyPep8Naming
def SequenceExample(feature_lists, context=None):
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)
