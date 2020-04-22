# Emotion Cause Extraction
## Overview
[paper](<https://ieeexplore.ieee.org/document/8598785>)

Emotion cause extraction is one of the most important applications in natural language processing tasks. It is a difﬁcult 
challenge due to the complex semantic information between emotion 
description and the whole document.
In this research we proposed a Hierarchical Network Based Clause Selection (HCS) framework in which the similarity is calculated by considering document features from word’s position, different semantic level (word and phrase) and interaction among clauses, respectively. 
## Experiment Detials
### Dataset
Experimental study on a Chinese emotion-cause corpus has shown the proposed framework’s effectiveness and potential of integrating different level’s information. The overview of dataset is like this:
![]( image.png) 

### World Level Attention
- `Position`
```sh
post_att = tf.expand_dims(tf.convert_to_tensor([(1 - i/self.length) for i in range(self.length)]),-1)
post_matrix = tf.reshape(tf.reduce_sum(tf.tensordot(tf.transpose(x1,[1,0,2,3]), post_att, 0),-2), x2.get_shape())
```
   
- `Content`
```
 x = tf.multiply(post_matrix, x2)
 attention_mul = tf.multiply(tf.nn.softmax(x), x)
```
         

### Phrase Network
```buildoutcfg
      LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=L_init1, x2=L_init2, d=embedding_size)
      ...
```

### Clause Network
```buildoutcfg
      with tf.name_scope("BiGRU"):
          cells = [dropout() for _ in range(num_layers)]
          # print("cells: ", tf.shape(cells))
          rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
          _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x_combine, dtype=tf.float32)
          last = _outputs[:, -1, :]
```

## Please Attention
```
A PyTorch verson is already under work.
If you have some questions, feel freely to contact xinyiyu0424@gmail.com
```

