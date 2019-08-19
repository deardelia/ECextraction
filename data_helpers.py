import numpy as np
import re
import itertools
import codecs
from collections import Counter
import jieba

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def load_data_and_labels():
  """
  Loads MR polarity data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  """
  # Load data from files
  examples = list(codecs.open("./data/data_included.txt", "r", "utf-8").readlines())
  examples = [s.strip() for s in examples]
  examples = [s.split(',')[2] for s in examples]
  examples = [s.replace(' ','') for s in examples]
  x_text = [[item for item in jieba.cut(s, cut_all=False)] for s in examples]

  # Split by words
  #x_text = examples
  # x_text = [clean_str(sent) for sent in x_text]
  #x_text = [list(s) for s in examples]

  # Load emotion from files
  emotion_examples = list(codecs.open("./data/train_data.txt","r","utf-8").readlines())
  emotion_examples = [e.split('\t')[1] for e in emotion_examples]
  emotion_text = [[item for item in jieba.cut(s, cut_all=False)] for s in emotion_examples]

  # Generate labels
  sentences = list(codecs.open("./data/train_data.txt", "r", "utf-8").readlines())
  sentences = [s.split(',')for s in sentences]
  y=[]
  for l in sentences:
    if l[0][0] == '1':
      y.append([0,1])
    elif l[0][0] == '0':
      y.append([1,0])
  #negative_labels = [[1, 0] for _ in negative_examples]
  #y = np.concatenate([labels, negative_labels], 0)
  y=np.array(y)
  return [x_text, emotion_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
  """
  Pads all sentences to the same length. The length is defined by the longest sentence.
  Returns padded sentences.
  """
  sequence_length = max(len(x) for x in sentences)
  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
  return padded_sentences


def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]


def build_input_data(sentences, emotion_sentences, labels, vocabulary):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  """
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  emotions = np.array([[vocabulary[emotion] for emotion in emotion_sentence] for emotion_sentence in emotion_sentences])
  y = np.array(labels)
  return [x, emotions, y]


def load_data():
  """
  Loads and preprocessed data for the MR dataset.
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """
  # Load and preprocess data
  sentences, emotion_sentences, labels = load_data_and_labels()
  sentences_padded = pad_sentences(sentences)
  emotion_sentences_padded = pad_sentences(emotion_sentences)
  vocabulary, vocabulary_inv = build_vocab(sentences_padded)
  #vocabulary_emotion, vocabulary_inv_emotion = build_vocab(emotion_sentences_padded)
  x, emotions, y = build_input_data(sentences_padded, emotion_sentences_padded, labels, vocabulary)
  return [x, emotions, y, vocabulary, vocabulary_inv]


def get_batch_size():
  j = 0
  with open('./data/train_data.txt', 'r', encoding="utf8") as fin:
    for line in fin:
      subsentence = line.split('\t')
      j = j + 1
  train_session_list = {};
  train_dialog = []
  l = 0
  with open('./data/train_data.list', 'r', encoding="utf8") as fin:
    for line in fin:
      index = line.split('\t')
      pos = index[1].find('Q') + 1
      qid = int(index[1][pos:len(index[1])])

      if (index[0] not in train_session_list):
        train_session_list[index[0]] = []
        train_session_list[index[0]].append(l)
        a_vec = []
        a_vec.append(l)
        train_dialog.append(a_vec)
      else:
        train_session_list[index[0]].append(l)
        train_dialog[qid - 1].append(l)
      l = l + 1
  train_session = train_dialog
  return train_session



def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  k=0
  #num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(191):
      if batch_num == 0:
        start_index = 0
        end_index = start_index + len(batch_size[batch_num])
      if batch_num > 0:
        start_index = start_index + len(batch_size[batch_num-1])
        end_index = start_index + len(batch_size[batch_num])
      yield shuffled_data[start_index:end_index]
