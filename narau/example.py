import tensorflow as tf
import os
from collections import Mapping


_is_iterable_exceptions = [str, bytes, Mapping]


def _is_iterable(iterable):
    try:
        iter(iterable)
        for t in _is_iterable_exceptions:
            if isinstance(iterable, t):
                return False
        return True
    except TypeError:
        return False


def _as_iterable(value):
    yield value


def _maybe_as_iterable(value):
    if _is_iterable(value):
        return value
    else:
        return _as_iterable(value)


# noinspection PyPep8Naming
def Int64Feature(value):
    value = _maybe_as_iterable(value)
    lst = tf.train.Int64List(value=value)
    return tf.train.Feature(int64_list=lst)


# noinspection PyPep8Naming
def BytesFeature(value):
    value = _maybe_as_iterable(value)
    lst = tf.train.BytesList(value=value)
    return tf.train.Feature(bytes_list=lst)


# noinspection PyPep8Naming
def FloatFeature(value):
    value = _maybe_as_iterable(value)
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
    return tf.train.Features(feature=feature_dict)


# noinspection PyPep8Naming
def Example(features):
    return tf.train.Example(features=features)


# noinspection PyPep8Naming
def SequenceExample(feature_lists, context=None):
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)


def save_example(example, path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with tf.python_io.TFRecordWriter(path) as writer:
        writer.write(example.SerializeToString())
