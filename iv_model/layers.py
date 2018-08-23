import tensorflow as tf
from contextlib import contextmanager


class Layer:
    def __init__(self, name):
        with tf.variable_scope(name, self.__class__.__name__) as vs:
            self._vs = vs
            self._ns = vs.original_name_scope

    @contextmanager
    def _scope(self):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False):
            with tf.name_scope(self._ns):
                yield


class TokenEmbedding(Layer):

    def __init__(self, token_size, embedding_size, with_pad=True, with_unk=True, weights=None, dtype=tf.float32, name=None):
        super().__init__(name)
        self._dtype = dtype

        with self._scope():
            embedding_tensors = []
            if with_pad:
                pad = tf.fill([1, embedding_size], 0.0, name='pad')
                embedding_tensors.append(pad)
                token_size -= 1

            if with_unk:
                unk = tf.get_variable('unk', [1, embedding_size], dtype=tf.float32)
                embedding_tensors.append(unk)
                token_size -= 1

            embedding_words = self._create_weights(weights, token_size, embedding_size)
            embedding_tensors.append(embedding_words)
            embedding_map = tf.concat(embedding_tensors, axis=0)
            self._embedding = embedding_map

    def _create_weights(self, weights, token_size, embedding_size):
        if weights is None:
            return tf.get_variable('weights', [token_size, embedding_size],
                                   dtype=self._dtype)
        else:
            return tf.constant(weights, self._dtype,
                               [token_size, embedding_size], 'weights', True)

    def __call__(self, tokens):
        with self._scope():
            return tf.nn.embedding_lookup(self._embedding, tokens)


class EmbeddingTransform(Layer):

    def __init__(self, units, activation=None, name=None):
        super().__init__(name)

        with self._scope():
            self._layers = [tf.layers.Dense(u, activation) for u in units]
            self._output_dim = units[-1]

    def __call__(self, x):
        with self._scope():
            shape = tf.shape(x)
            out_dim = x.shape[2]

            x = tf.reshape(x, shape=(-1, out_dim))
            for layer in self._layers:
                x = layer(x)
            x = tf.reshape(x, tf.concat([shape[:2], [self._output_dim]], axis=0))
            return x


class NLPStackedCNN(Layer):

    def __init__(self, filter_sizes, filter_num, max_pool_size, activation=None,
                 residues=False, drop_out=None, name=None):
        super().__init__(name)

        self._max_pool_size = max_pool_size
        self._residues = residues
        self._dropout = drop_out

        if len(filter_sizes) < 1:
            raise ValueError()

        try:
            iter(filter_num)
        except TypeError:
            if isinstance(filter_num, int):
                filter_num = [filter_num] * len(filter_sizes)

        if len(filter_sizes) != len(filter_num):
            raise ValueError()

        with self._scope():
            self._filters = [tf.layers.Conv1D(num, size, padding='same', activation=activation)
                             for size, num in zip(filter_sizes, filter_num)]

    def __call__(self, x):
        with self._scope():
            residues = []
            for flt in self._filters[:-1]:
                x = flt(x)
                residues.append(x)
                x = tf.layers.max_pooling1d(x, self._max_pool_size, self._max_pool_size, padding='same')
            x = self._filters[-1](x)
            residues.append(x)

            if self._residues:
                x = [tf.reduce_max(r, axis=1) for r in residues]
                x = tf.concat(x, axis=1)
            else:
                x = tf.reduce_max(x, axis=1)

            if self._dropout:
                x = tf.layers.dropout(x, self._dropout)

            return x


class NLPMultiStackedCNN(Layer):

    def __init__(self, filter_base_sizes, filter_num, filter_height,
                 activation=None, residues=False, drop_out=None, name=None):
        super().__init__(name)

        if len(filter_base_sizes) < 1:
            raise ValueError()

        max_pool_size = 3
        filter_size = 3
        filter_top_sizes = [filter_size] * (filter_height - 1)

        with self._scope():
            self._cnns = []
            for fs in filter_base_sizes:
                filter_sizes = [fs] + filter_top_sizes
                self._cnns.append(
                    NLPStackedCNN(filter_sizes, filter_num, max_pool_size, activation, residues, drop_out))

    def __call__(self, x):
        with self._scope():
            x = [cnn(x) for cnn in self._cnns]
            x = tf.concat(x, axis=1)
            return x


class Projection(Layer):

    def __init__(self, units, activation=None, name=None):
        super().__init__(name)

        if len(units) < 1:
            raise ValueError()

        with self._scope():
            self._layers = [tf.layers.Dense(u, activation) for u in units]

    def __call__(self, x):
        with self._scope():
            for layer in self._layers[:-1]:
                x = layer(x)
            x = self._layers[-1](x)
            return x