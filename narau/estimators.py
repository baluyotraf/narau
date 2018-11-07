import tensorflow as tf
from . import layers


class SiameseBiLSTMEmbedding(tf.estimator.Estimator):
    """
    A Siamese Bidirectional LSTM network for sentence similarity

    :param embedding_size: number of embedding tokens
    :param embedding_dims: dimension of embedding dense representation
    :param embedding_special_tokens: number of special tokens
    :param embedding_with_pad: defines if embedding has a padding special token
    :param embedding_weights: initial weights of the embedding layer
    :param embedding_trainable: defines if embedding is trainable
    :param embedding_units: list of the embedding transformation units
    :param lstm_units: list of the LSTM cell units
    :param lstm_drop_out: LSTM cell output drop out probability
    :param projection_units: list units in the projection layer
    :param loss_margin: target distance of each embedding
    :param learning_rate: optimization learning rate
    :param model_dir: directory to save the checkpoints
    :param config: estimator configuration
    :param warm_start_from: estimator warm start configuration

    Training/Development Features:
        x1: Sentence 1
        len1: Length of sentence 1
        x2: Sentence 2
        len2: Length of sentence 2

    Prediction Features:
        x: Sentence for embedding
        len: Length of sentence to embed
    """

    def __init__(self, embedding_size, embedding_dims,
                 embedding_special_tokens, embedding_with_pad,
                 embedding_weights, embedding_trainable,
                 embedding_units,
                 lstm_units, lstm_drop_out,
                 projection_units, loss_margin=1.0,
                 learning_rate=0.1, model_dir=None,
                 config=None, warm_start_from=None):

        def model_fn(features, labels, mode):
            emb = layers.TokenEmbedding(embedding_size, embedding_dims, tf.float32,
                                        embedding_special_tokens, embedding_with_pad,
                                        embedding_trainable)
            embt = layers.EmbeddingTransform(embedding_units, tf.nn.relu)
            lstm = layers.BidirectionalLSTM(lstm_units,
                                            drop_out=(lstm_drop_out if mode == tf.estimator.ModeKeys.TRAIN
                                                      else None))
            proj = layers.Projection(projection_units, tf.nn.relu)
            norm = layers.L2Normalization()

            def network(x, l):
                x = emb(x)
                x = embt(x)
                x = lstm(x, l)
                x = proj(x)
                x = norm(x)
                return x

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                x1 = network(features['x1'], features['len1'])
                x2 = network(features['x2'], features['len2'])

                with tf.variable_scope('loss'):
                    loss = tf.contrib.losses.metric_learning.contrastive_loss(labels, x1, x2, loss_margin)

                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(mode, loss=loss)

                with tf.variable_scope('train'):
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    train_op = optimizer.minimize(loss, tf.train.get_global_step())

                def scaffold_init(scaffold, session):
                    emb.feed_dict_init(session, embedding_weights)
                scaffold = tf.train.Scaffold(init_fn=scaffold_init)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                  scaffold=scaffold)
            else:
                x_ = network(features['x'], features['len'])
                return tf.estimator.EstimatorSpec(mode, predictions={'embedding': x_})

        super().__init__(model_fn, model_dir, config, None, warm_start_from)


class BinarySentimentClassifier(tf.estimator.Estimator):
    """
    A sentence sentiment classification using binary labels

    :param embedding_size: number of embedding tokens
    :param embedding_dims: dimension of embedding dense representation
    :param embedding_special_tokens: number of special tokens
    :param embedding_with_pad: defines if embedding has a padding special token
    :param embedding_weights: initial weights of the embedding layer
    :param embedding_trainable: defines if embedding is trainable
    :param embedding_units: list of the embedding transformation units
    :param lstm_units: list of the LSTM cell units
    :param lstm_drop_out: LSTM cell output drop out probability
    :param projection_units: list units in the projection layer
    :param learning_rate: optimization learning rate
    :param model_dir: directory to save the checkpoints
    :param config: estimator configuration
    :param warm_start_from: estimator warm start configuration

    Training/Development/Prediction Features:
        x: Sentence to get the sentiment
        len: Length of sentence to get the sentiment
    """

    def __init__(self, embedding_size, embedding_dims,
                 embedding_special_tokens, embedding_with_pad,
                 embedding_weights, embedding_trainable,
                 embedding_units,
                 lstm_units, lstm_drop_out,
                 projection_units,
                 learning_rate=0.1, model_dir=None,
                 config=None, warm_start_from=None):
        def model_fn(features, labels, mode):
            emb = layers.TokenEmbedding(embedding_size, embedding_dims, tf.float32,
                                        embedding_special_tokens, embedding_with_pad,
                                        embedding_trainable)
            embt = layers.EmbeddingTransform(embedding_units, tf.nn.relu)
            lstm = layers.BidirectionalLSTM(lstm_units,
                                            drop_out=(lstm_drop_out if mode == tf.estimator.ModeKeys.TRAIN
                                                      else None))
            proj = layers.Projection(projection_units, tf.nn.relu)

            x, len_ = features['x'], features['len']
            x = emb(x)
            x = embt(x)
            x = lstm(x, len_)
            x = proj(x)

            if mode == tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope('predictions'):
                    logit = tf.squeeze(x, axis=1)
                    probability = tf.nn.sigmoid(logit)
                    label = tf.cast(probability + 0.5, tf.int32)

                predictions = {
                    'logit': logit,
                    'probability': probability,
                    'label': label,
                }

                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            with tf.variable_scope('loss'):
                loss = tf.losses.sigmoid_cross_entropy(labels, x)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss)

            with tf.variable_scope('train'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, tf.train.get_global_step())

            def scaffold_init(scaffold, session):
                emb.feed_dict_init(session, embedding_weights)
            scaffold = tf.train.Scaffold(init_fn=scaffold_init)

            return tf.estimator.EstimatorSpec(mode, loss=loss,
                                              train_op=train_op,
                                              scaffold=scaffold)

        super().__init__(model_fn, model_dir, config, None, warm_start_from)
