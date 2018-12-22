#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from ACNN import ACNN
from tensorflow.contrib import learn
from sklearn import metrics

import jieba
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/chinese/pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/chinese/neg.txt", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_emotion_sizes", "1,2,3", "Comma-separated filter emotion sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("hidden_dim",128, "Hidden dim in gru (default: 128)")
tf.flags.DEFINE_integer("num_layers", 2, "Number of layers in gru (default: 2)")
tf.flags.DEFINE_float("drop_keep_gru", 0.8, "Dropout keep probability in gru (default: 0.8)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate in gru (default: 1e-3)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("window_size", 4, "The size of filter window (default: 4)")
tf.flags.DEFINE_integer("num_features", 64, "The size of num_features (default: 64)")

# Training parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, emotions, y= data_helpers.load_data_and_labels()
# Build vocabulary
max_document_length = max([len(x) for x in x_text])
#max_emotion_length = max([len(e) for e in emotions])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#vocab_emotion_processor = learn.preprocessing.VocabularyProcessor(max_emotion_length)
text_list=[]
emotion_list=[]
for text in x_text:
  text_list.append(' '.join(text))
for e in emotions:
  emotion_list.append(' '.join(e))
x = np.array(list(vocab_processor.fit_transform(text_list)))#变为矩阵句子数*每句词汇数
emotions = np.array(list(vocab_processor.fit_transform(emotion_list)))
# Randomly shuffle data
'''np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-300], x_shuffled[-300:]
y_train, y_dev = y_shuffled[:-300], y_shuffled[-300:]'''
x_train = x[:11138]
x_dev = x[11138:]
x_emotions = emotions[:11138]
x_emotions_dev = emotions[11138:]
y_train = y[:11138]
y_dev = y[11138:]

sequence_length = x_train.shape[1]
emotion_length = x_emotions.shape[1]
#print("Vocabulary Size: {:d}".format(len(vocabulary)))
#print("Train split: {:d}".format(len(y_train)))
print("Sequnence Length: {:d}".format(sequence_length))
print("Emotion Length: {:d}".format(emotion_length))


# Training
# ==================================================

with tf.Graph().as_default():
  session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
  sess = tf.Session(config=session_conf)
  with sess.as_default():
    cnn = ACNN(
      sequence_length=sequence_length,
      emotion_length=emotion_length,
      num_classes=y_train.shape[1],
      vocab_size=len(vocab_processor.vocabulary_),
      embedding_size=FLAGS.embedding_dim,
      filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
      filter_emotion_sizes=list(map(int, FLAGS.filter_emotion_sizes.split(","))),
      num_filters=FLAGS.num_filters,
      hidden_dim=FLAGS.hidden_dim,
      num_layers=FLAGS.num_layers,
      drop_keep_gru=FLAGS.drop_keep_gru,
      learning_rate=FLAGS.learning_rate,
      window_size=FLAGS.window_size,
      num_features=len(vocab_processor.vocabulary_),
      l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    """grad_summaries = []
    for g, v in grads_and_vars:
      if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)"""

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())


    def train_step(x_batch, x_emotion_batch, y_batch):
      """
      A single training step
      """
      feed_dict = {
        cnn.x: x_batch,
        cnn.emotion: x_emotion_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
      }
      _, step, summaries, loss, accuracy, prediction, y_true = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.y_pred_cls, cnn.y_true],
        feed_dict)
      #_, step, summaries, loss, accuracy = sess.run(
       # [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        #feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)
      return [prediction, y_true]

    def dev_step(x_batch, y_batch, writer=None):
      """
      Evaluates model on a dev set
      """
      feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.input_emotion: x_emotion_batch,
        cnn.dropout_keep_prob: 1.0}

      step, summaries, loss, accuracy= sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
      time_str = datetime.datetime.now().isoformat()
      #
      #print("okkkkkk")
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      if writer:
        writer.add_summary(summaries, step)

    def caculate_matricx(y,pre):
      precision = metrics.precision_score(y, pre)
      recall = metrics.recall_score(y, pre)
      f1 = metrics.f1_score(y, pre)
      return [precision, recall, f1]
    # Generate batches


    # Generate batches
    batch_size = data_helpers.get_batch_size()
    batches = data_helpers.batch_iter(
      list(zip(x_train, x_emotions, y_train)), batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    count = 0
    prediction_totall = []
    y_true_total = []
    for batch in batches:
      x_batch, x_emotion_batch, y_batch = zip(*batch)
      prediction, y_true = train_step(x_batch, x_emotion_batch, y_batch)
      prediction_totall += prediction.tolist()
      y_true_total += y_true.tolist()
      count = count + 1
      if (count % 10 == 0):
        precision, recall, f1 = caculate_matricx(np.array(y_true_total), np.array(prediction_totall))
        print("################################# Matricx ##########################\n")

        print("precision {:g}, recall {:g}, f1 {:g}".format(precision, recall, f1))
        count = 0
        prediction_totall = []
        y_true_total = []
      current_step = tf.train.global_step(sess, global_step)
      if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        # dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
      if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
