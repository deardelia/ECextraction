import tensorflow as tf
import sklearn as sk

import numpy as np

class ACNN(object):
  """
  A CNN for text classification.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  """
  def __init__(
      self, sequence_length, emotion_length, num_classes, vocab_size,
    embedding_size, filter_sizes, filter_emotion_sizes, num_filters, hidden_dim, num_layers, drop_keep_gru,learning_rate,window_size, num_features, l2_reg_lambda=0.0004):
      # Placeholders for input, output and dropout
      self.x = tf.placeholder(tf.int32, [None, sequence_length], name="x")
      self.emotion = tf.placeholder(tf.int32, [None, sequence_length], name="emotion")
      self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
      self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      self.length = sequence_length

      with tf.device('/cpu:0'), tf.name_scope("embedding"):
          self.W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
          self.input_x = tf.transpose(tf.nn.embedding_lookup(self.W, self.x),[0,2,1])
          self.input_emotion = tf.transpose(tf.nn.embedding_lookup(self.W, self.emotion),[0,2,1])
          #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
          #self.embedding_emotions_expanded = tf.expand_dims(self.embedded_emotions, -1)
      #print(self.input_x.get_shape().as_list() )#(?, 41, 128)
      #print(self.input_emotion.get_shape().as_list())#(?, 41, 128)


      def pad_for_wide_conv(x):
          return tf.pad(x, np.array([[0, 0], [0, 0], [window_size - 1, window_size - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

      def all_pool(variable_scope, x):
          with tf.variable_scope(variable_scope + "-all_pool"):
              if variable_scope.startswith("input"):
                  pool_width = sequence_length
                  d = embedding_size
              else:
                  pool_width = sequence_length + window_size - 1
                  d = 50

              all_ap = tf.layers.average_pooling2d(
                  inputs=x,
                  # (pool_height, pool_width)
                  pool_size=(1, pool_width),
                  strides=1,
                  padding="VALID",
                  name="all_ap"
              )
              # [batch, di, 1, 1]

              # [batch, di]
              all_ap_reshaped = tf.reshape(all_ap, [-1, d])
              # all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

              return all_ap_reshaped

      def w_pool(variable_scope, x, attention):
          # x: [batch, di, s+w-1, 1]
          # attention: [batch, s+w-1]
          with tf.variable_scope(variable_scope + "-w_pool"):
                pools = []
                # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])
                for i in range(sequence_length):
                    # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                    pools.append(tf.reduce_sum(x[:, :, i:i + window_size, :] * attention[:, :, i:i + window_size, :],
                                                 axis=2,
                                                 keep_dims=True))
                # [batch, di, s, 1]
                w_ap = tf.concat(pools, axis=2, name="w_ap")

                # [batch, di, s, 1]

                return w_ap

      def make_attention_mat(x1, x2):
          # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
          # x2 => [batch, height, 1, width]
          # [batch, width, wdith] = [batch, s, s]
          euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
          return 1 / (1 + euclidean)


      def attention_machanism(x1, x2):

          # ATTENTION STARTS HERE
          post_att = tf.expand_dims(tf.convert_to_tensor([(1 - i/self.length) for i in range(self.length)]),-1)
          post_matrix = tf.reshape(tf.reduce_sum(tf.tensordot(tf.transpose(x1,[1,0,2,3]), post_att, 0),-2), x2.get_shape())
          x = tf.multiply(post_matrix, x2)
          attention_mul = tf.multiply(tf.nn.softmax(x), x)
          return attention_mul

      def convolution(name_scope, x, d, reuse):
          with tf.name_scope(name_scope + "-conv"):
              with tf.variable_scope("conv") as scope:
                  conv = tf.contrib.layers.conv2d(
                      inputs=x,
                      num_outputs=50,
                      kernel_size=(d, window_size),
                      stride=1,
                      padding="VALID",
                      activation_fn=tf.nn.tanh,
                      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                      weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                      biases_initializer=tf.constant_initializer(1e-04),
                      reuse=reuse,
                      trainable=True,
                      scope=scope
                  )
                  # Weight: [filter_height, filter_width, in_channels, out_channels]
                  # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                  # [batch, di, s+w-1, 1]
                  conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                  return conv_trans
      def CNN_layer(variable_scope, x1, x2, d):
          # x1, x2 = [batch, d, s, 1]
          with tf.variable_scope(variable_scope):
              #print("cnn:")
              left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d,
                                      reuse=False)  # [batch, di, s+w-1, 1]
              right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d,
                                       reuse=True)  # [batch, di, s+w-1, 1]
              #print(left_conv.get_shape().as_list())#(? 50 44 1)
              #print(right_conv.get_shape().as_list())

              left_attention, right_attention = None, None


              # [batch, s+w-1, s+w-1]
              att_mat = make_attention_mat(left_conv, right_conv)
              # [batch, s+w-1], [batch, s+w-1]
              left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

              left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)  ## [batch, di, s, 1]
              left_ap = all_pool(variable_scope="left", x=left_conv)  # [batch, di]
              right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
              right_ap = all_pool(variable_scope="right", x=right_conv)

              return left_wp, left_ap, right_wp, right_ap

      def cos_sim(v1, v2):
          norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
          norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
          dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

          return dot_products / (norm1 * norm2)


      # BiGRU
      '''def lstm_cell():  # lstm核
          return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)'''
      def gru_cell():  # gru核
          return tf.contrib.rnn.GRUCell(hidden_dim)
      def dropout():  # 为每一个rnn核后面加一个dropout层
          '''if (self.config.rnn == 'lstm'):
              cell = lstm_cell()
          else:
              cell = gru_cell()'''
          cell = gru_cell()
          return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=drop_keep_gru)



      x1_expanded = tf.expand_dims(self.input_x, -1)#(?, 41, 128,1)
      x2_expanded = tf.expand_dims(self.input_emotion, -1)
      #print(x1_expanded.get_shape().as_list())
      #print(x2_expanded.get_shape().as_list())

      #L_init1 = attention_machanism(x1_expanded, x2_expanded)
      #L_init2 = x2_expanded


      LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)#(?, 128)
      RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)
      #print(LO_0.get_shape().as_list())
      #print(RO_0.get_shape().as_list())

      LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=embedding_size)
      sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)]

      _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", x1=LI_1, x2=RI_1, d=50)
      self.test = LO_2
      self.test2 = RO_2
      sims.append(cos_sim(LO_2, RO_2))
      """print("sims:\n")
      print(np.shape(sims))
      print(tf.stack(sims, axis=1).get_shape().as_list())
      with tf.variable_scope("Acnn-output-layer"):
          self.output_features = tf.concat([self.features, tf.stack(sims, axis=1)], axis=1, name="cnn-output_features")
          self.estimation = tf.contrib.layers.fully_connected(
              inputs=self.output_features,
              num_outputs=num_classes,
              activation_fn=None,
              weights_initializer=tf.contrib.layers.xavier_initializer(),
              weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
              biases_initializer=tf.constant_initializer(1e-04),
              scope="FC"
          )"""

      # Combine clause and emotion
      # x_combine = tf.concat([self.h_pool_flat_e, self.h_pool_flat], 1)
      x_combine = tf.concat([LO_2, RO_2], 1)
      x_combine = tf.expand_dims(x_combine, -1)
      '''print(tf.shape(self.h_pool_flat_e))
      print(tf.shape(self.h_pool_flat))
      print("x_combine: ")
      print(np.shape(x_combine))'''

      with tf.name_scope("BiGRU"):
          cells = [dropout() for _ in range(num_layers)]
          # print("cells: ", tf.shape(cells))
          rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
          _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x_combine, dtype=tf.float32)
          last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

      with tf.name_scope("score"):
          # 全连接层，后面接dropout以及relu激活
          fc = tf.layers.dense(last, hidden_dim, name='fc1')
          fc = tf.contrib.layers.dropout(fc, drop_keep_gru)
          fc = tf.nn.relu(fc)

          # 分类器
          self.logits = tf.layers.dense(fc, num_classes, name='fc2')
          self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

      with tf.name_scope("optimize"):
          # 损失函数，交叉熵
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
          self.loss = tf.reduce_mean(cross_entropy)
          # 优化器
          self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

      with tf.name_scope("accuracy"):
          # 准确率
          correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
          self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

      with tf.name_scope("precision"):
          self.y_true = tf.argmax(self.input_y, 1)

