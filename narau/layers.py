import tensorflow as tf
from contextlib import contextmanager


class Layer:
    def __init__(self, name=None):
        with tf.variable_scope(name, self.__class__.__name__) as vs:
            self._vs = vs
            self._ns = vs.original_name_scope

    @contextmanager
    def _scope(self):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False):
            with tf.name_scope(self._ns):
                yield


class TokenEmbedding(Layer):

    def __init__(self, token_size, dimensions,
                 dtype=tf.float32, special_token_size=None,
                 with_pad=None, weights=None, name=None):
        super().__init__(name)

        with self._scope():
            embeddings = self._create_s_tokens(special_token_size,
                                               dimensions, with_pad, dtype)
            weights_tensor = tf.get_variable('weights', [token_size, dimensions],
                                             dtype, trainable=(weights is None))
            embeddings.append(weights_tensor)

            self._embedding_map = tf.concat(embeddings, axis=0, name='embedding_map')
            self._weights_tensor = weights_tensor
            self._weights = weights

    # noinspection PyMethodMayBeStatic
    def _create_s_tokens(self, s_token_size, dimensions, with_pad, dtype):
        tensors = []
        if s_token_size is not None and s_token_size > 0:
            if with_pad:
                tensors.append(tf.fill([1, dimensions], 0.0, name='pad'))
                s_token_size -= 1
            s_tokens = tf.get_variable('special_tokens', [s_token_size, dimensions], dtype)
            tensors.append(s_tokens)
        return tensors

    def __call__(self, tokens):
        with self._scope():
            return tf.nn.embedding_lookup(self._embedding_map, tokens)

    def feed_dict_init(self, session):
        if self._weights is not None:
            w = self._weights_tensor
            session.run(w.initializer, {w.initial_value: self._weights})


class EmbeddingTransform(Layer):

    def __init__(self, units, activation=None, name=None):
        super().__init__(name)

        with self._scope():
            self._layers = [tf.keras.layers.TimeDistributed(tf.layers.Dense(u, activation))
                            for u in units]

    def __call__(self, x):
        with self._scope():
            for layer in self._layers:
                x = layer(x)
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


class BidirectionalLSTM(Layer):

    def __init__(self, units, return_sequence=False, drop_out=None, use_cuda=True, name=None):
        super().__init__(name)
        self._return_sequence = return_sequence

        with self._scope():
            fw_cells = [self._create_cell(u, drop_out, use_cuda) for u in units]
            bw_cells = [self._create_cell(u, drop_out, use_cuda) for u in units]
            self._fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells)
            self._bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells)

    # noinspection PyMethodMayBeStatic
    def _create_cell(self, unit, drop_out=None, use_cuda=True):
        if use_cuda:
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(unit)
        else:
            cell = tf.contrib.rnn.LSTMBlockFusedCell(unit)

        if drop_out is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1.0-drop_out))
        return cell

    def __call__(self, x, l):
        with self._scope():
            output, state = tf.nn.bidirectional_dynamic_rnn(self._fw_cell, self._bw_cell, x, l, dtype=x.dtype)
            if self._return_sequence:
                return output
            else:
                output = tf.concat(output, 2)
                output_rows = tf.range(tf.shape(output)[0])
                output_cols = l - 1
                output_last = tf.gather_nd(output, tf.stack([output_rows, output_cols], axis=1))
                return output_last


class Projection(Layer):

    def __init__(self, units, activation=None, name=None):
        super().__init__(name)

        if len(units) < 1:
            raise ValueError()

        with self._scope():
            self._layers = [tf.layers.Dense(u, activation) for u in units[:-1]]
            self._layers.append(tf.layers.Dense(units[-1]))

    def __call__(self, x):
        with self._scope():
            for layer in self._layers:
                x = layer(x)
            return x


class L2Normalization(Layer):

    def __call__(self, x):
        with self._scope():
            return tf.nn.l2_normalize(x, -1)
