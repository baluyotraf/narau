import tensorflow as tf
from . import layers


class SiameseBiLSTMEmbedding(tf.estimator.Estimator):

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
            lstm = layers.BidirectionalLSTM(lstm_units, drop_out=lstm_drop_out)
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
