import tensorflow as tf
import os
from collections import Mapping


_is_iterable_exceptions = [str, bytes, Mapping]


def _is_iterable(iterable):
    """
    Checks if an object is an iterable
    :param iterable: object to check
    :return: True if iterable, else False
    """
    try:
        iter(iterable)
        for t in _is_iterable_exceptions:
            if isinstance(iterable, t):
                return False
        return True
    except TypeError:
        return False


def _as_iterable(value):
    """
    Returns the value as iterable
    :param value: value to convert
    :return: generator containing the value
    """
    yield value


def _maybe_as_iterable(value):
    """
    Converts a non-iterable value to one if it is not
    :param value: object to ensure iterability
    :return: iterable equivalent of value
    """
    if _is_iterable(value):
        return value
    else:
        return _as_iterable(value)


# noinspection PyPep8Naming
def Int64Feature(value):
    """
    Creates a int64 feature protobuf object
    :param value: list of int64 values
    :return: Feature protobuf object
    """
    value = _maybe_as_iterable(value)
    lst = tf.train.Int64List(value=value)
    return tf.train.Feature(int64_list=lst)


# noinspection PyPep8Naming
def BytesFeature(value):
    """
    Creates a bytes feature protobuf object
    :param value: list of bytes values
    :return: Feature protobuf object
    """
    value = _maybe_as_iterable(value)
    lst = tf.train.BytesList(value=value)
    return tf.train.Feature(bytes_list=lst)


# noinspection PyPep8Naming
def FloatFeature(value):
    """
    Creates a float feature protobuf object
    :param value: list of float values
    :return: Feature protobuf object
    """
    value = _maybe_as_iterable(value)
    lst = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list=lst)


# noinspection PyPep8Naming
def Int64FeatureList(values):
    """
    Creates a int64 feature list protobuf object
    :param value: list of list of int64 values
    :return: FeatureList protobuf object
    """
    feature = [Int64Feature(v) for v in values]
    return tf.train.FeatureList(feature=feature)


# noinspection PyPep8Naming
def BytesFeatureList(values):
    """
    Creates a bytes feature list protobuf object
    :param value: list of list of bytes values
    :return: FeatureList protobuf object
    """
    feature = [BytesFeature(v) for v in values]
    return tf.train.FeatureList(feature=feature)


# noinspection PyPep8Naming
def FloatFeatureList(values):
    """
    Creates a float feature list protobuf object
    :param value: list of list of float values
    :return: FeatureList protobuf object
    """
    feature = [FloatFeature(v) for v in values]
    return tf.train.FeatureList(feature=feature)


# noinspection PyPep8Naming
def FeatureLists(feature_dict):
    """
    Creates a named collection of feature lists
    :param feature_dict: dictionary of name and FeatureList value pair
    :return: FeatureLists protobuf object
    """
    return tf.train.FeatureLists(feature_list=feature_dict)


# noinspection PyPep8Naming
def Features(feature_dict):
    """
    Creates a named collection of features
    :param feature_dict: dictionary of name and Feature value pair
    :return: Features protobuf object
    """
    return tf.train.Features(feature=feature_dict)


# noinspection PyPep8Naming
def Example(features):
    """
    Creates an example from features
    :param features: Features protobuf object
    :return: Example protobuf object
    """
    return tf.train.Example(features=features)


# noinspection PyPep8Naming
def SequenceExample(feature_lists, context=None):
    """
    Creates an example from feature lists with context
    :param feature_lists: FeatureLists protobuf object
    :param context: Features protobuf object
    :return: Sequence Example protobuf object
    """
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)


def save_example(example, path):
    """
    Saves an example to the given path
    :param example: Example or SequenceExample protobuf object
    :param path: path to save the file
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with tf.python_io.TFRecordWriter(path) as writer:
        writer.write(example.SerializeToString())
