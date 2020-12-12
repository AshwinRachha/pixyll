---
layout:     post
title:      Detecting Insincere Questions with LSTMs.
date:       2020-11-11 11:21:29
summary:    In this project we see how text classification is done using Word Embeddings and LSTMs.
categories: jekyll pixyll
---




The internet has become the pre-eminent source of sharing information in the world today. People use content based websites such as Quora, Reddit and StackOverflow, posting genuine questions pertaining to various domains. Although majority of questions are genuine, it may so happen that some eccentric person posts a statement which is not a genuine query. In order to enure the plain sailing dissemination of useful information online and avoid spreading toxic insincere queries, we need to screen them first so that they can be detected beforehand and prevented from subsisting online. In this project we see how this can be done using word embeddings and long short term memory networks.



We will import all necessary libraries.




```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
```

Next lets read our training data. The dataframe consists of the question id, the corresponding question and the target label. A target label of '0' corresponds to a sincere question and '1' corresponds to an insincere question.




```python
train_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>qid</th>
      <th>question_text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00002165364db923c7e6</td>
      <td>How did Quebec nationalists see their province...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000032939017120e6e44</td>
      <td>Do you have an adopted dog, how would you enco...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000412ca6e4628ce2cf</td>
      <td>Why does velocity affect time? Does velocity a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000042bf85aa498cd78e</td>
      <td>How did Otto von Guericke used the Magdeburg h...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0000455dfa3e01eae3af</td>
      <td>Can I convert montra helicon D to a mountain b...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Next up we split the dataframe into train and test splits. 


```python
train_df, val_df = train_test_split(train_df, test_size = 0.2)
```



Now we load the pretrained glove embeddings into an embeddings dictionary. 






```python
embeddings_index = {}
f = open('/kaggle/working/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print(f'Found {len(embeddings_index)} word vectors')
```

    2196017it [05:02, 7268.92it/s]
    

    Found 2196016 word vectors
    


```python
def text_to_array(text):
    empty_emb =  np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empty_emb) for x in text]
    embeds+=[empty_emb] * (30 - len(embeds))
    return np.array(embeds)

val_vects = np.array([text_to_array(xtext) for xtext in tqdm(val_df['question_text'][:3000])])
val_y = np.array(val_df['target'][:3000])
```

    100%|██████████| 3000/3000 [00:00<00:00, 12486.90it/s]
    


```python
batch_size = 128

def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True:
        train_df = train_df.sample(frac = 1.)
        
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size,1]
            text_arr = np.array([text_to_array(text) for text in texts])
            
            yield text_arr, np.array(train_df['target'][i*batch_size:(i+1)*batch_size])
            
```

1. RNNs



Recurrent Neural Networks are a class of artificial neural

networks that were first introduced to manipulate sequential

data. RNNs exhibit temporal dynamic behavior which allow

them to transfer information more easily from one time step

to another. The information from the previous states is termed

as a hidden state.

RNNs showed good results in a majority of applications

ranging from time series prediction, text generation, biological modeling, speech recognition etc. However vanilla RNNs

suffer from a major problem of vanishing gradients. As many

other machine learning algorithms, RNNs are optimized using

Back Propagation and due to their sequential nature the error

decays severely as it propagates back through layers. The

gradient which is very small, thus effectively prevents the

weight from changing its value. In the worst case the neural

network may even stop training further.



2. LSTMs



In order to remedy the vanishing gradients problem, Long

Short Term Memory Networks were introduced, the architecture of which consists of different gates viz update, relevance,

forget and output. This introduction of a tweaking in the

architecture of RNNs enabled LSTMs to remember relevant

context from long range dependencies and introduced the

flexibility to use long sentences. 

To boost the performance of existing architectures Bidirectional LSTMs were introduced that can process sequences

from forward as well as backward directions intended for the

network to learn better understandings from both sequence

directions






```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
```


```python
    inp = Input(shape=(max_length,))
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model
```


```python
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```


```python
mg = batch_gen(train_df)
model.fit_generator(mg, epochs = 20, steps_per_epoch = 1000, validation_data = (val_vects, val_y), verbose = True)

```

    Epoch 1/20
    1000/1000 [==============================] - 30s 30ms/step - loss: 0.1322 - accuracy: 0.9486 - val_loss: 0.1255 - val_accuracy: 0.9530
    Epoch 2/20
    1000/1000 [==============================] - 29s 29ms/step - loss: 0.1178 - accuracy: 0.9538 - val_loss: 0.1158 - val_accuracy: 0.9547
    Epoch 3/20
    1000/1000 [==============================] - 27s 27ms/step - loss: 0.1132 - accuracy: 0.9551 - val_loss: 0.1125 - val_accuracy: 0.9547
    Epoch 4/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.1124 - accuracy: 0.9556 - val_loss: 0.1109 - val_accuracy: 0.9587
    Epoch 5/20
    1000/1000 [==============================] - 27s 27ms/step - loss: 0.1088 - accuracy: 0.9576 - val_loss: 0.1126 - val_accuracy: 0.9557
    Epoch 6/20
    1000/1000 [==============================] - 29s 29ms/step - loss: 0.1085 - accuracy: 0.9572 - val_loss: 0.1070 - val_accuracy: 0.9547
    Epoch 7/20
    1000/1000 [==============================] - 27s 27ms/step - loss: 0.1065 - accuracy: 0.9582 - val_loss: 0.1044 - val_accuracy: 0.9580
    Epoch 8/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.1063 - accuracy: 0.9576 - val_loss: 0.1046 - val_accuracy: 0.9573
    Epoch 9/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.0991 - accuracy: 0.9607 - val_loss: 0.1029 - val_accuracy: 0.9567
    Epoch 10/20
    1000/1000 [==============================] - 29s 29ms/step - loss: 0.0989 - accuracy: 0.9606 - val_loss: 0.1039 - val_accuracy: 0.9563
    Epoch 11/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.0977 - accuracy: 0.9613 - val_loss: 0.1048 - val_accuracy: 0.9557
    Epoch 12/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.0991 - accuracy: 0.9608 - val_loss: 0.1033 - val_accuracy: 0.9557
    Epoch 13/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.1001 - accuracy: 0.9599 - val_loss: 0.1040 - val_accuracy: 0.9553
    Epoch 14/20
    1000/1000 [==============================] - 27s 27ms/step - loss: 0.0988 - accuracy: 0.9611 - val_loss: 0.1063 - val_accuracy: 0.9583
    Epoch 15/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.0972 - accuracy: 0.9608 - val_loss: 0.1044 - val_accuracy: 0.9590
    Epoch 16/20
    1000/1000 [==============================] - 27s 27ms/step - loss: 0.0997 - accuracy: 0.9598 - val_loss: 0.1011 - val_accuracy: 0.9560
    Epoch 17/20
    1000/1000 [==============================] - 29s 29ms/step - loss: 0.0901 - accuracy: 0.9638 - val_loss: 0.1080 - val_accuracy: 0.9573
    Epoch 18/20
    1000/1000 [==============================] - 27s 27ms/step - loss: 0.0873 - accuracy: 0.9650 - val_loss: 0.1079 - val_accuracy: 0.9557
    Epoch 19/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.0885 - accuracy: 0.9640 - val_loss: 0.1047 - val_accuracy: 0.9557
    Epoch 20/20
    1000/1000 [==============================] - 28s 28ms/step - loss: 0.0885 - accuracy: 0.9644 - val_loss: 0.1064 - val_accuracy: 0.9557
    




    <tensorflow.python.keras.callbacks.History at 0x7fddf669c2d0>




```python
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())
```

    165it [00:18, 10.07it/s]


```python
all_preds
```
.
