---
layout:     post
title:      Contradictory, My Dear Watson.
date:       2014-06-11 15:31:19
summary:    Detecting contradiction from a given premise and hypothesis using transformers. 
categories: jekyll pixyll
---


Let's add the libraries where they are really needed, not all of them at the first line


```python
import pandas as pd
```

### our data frames


```python
train_df = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test_df  = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
sample_df = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
```

### Initiative knowledge about our data


```python
train_df
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
      <th>id</th>
      <th>premise</th>
      <th>hypothesis</th>
      <th>lang_abv</th>
      <th>language</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5130fd2cb5</td>
      <td>and these comments were considered in formulat...</td>
      <td>The rules developed in the interim were put to...</td>
      <td>en</td>
      <td>English</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5b72532a0b</td>
      <td>These are issues that we wrestle with in pract...</td>
      <td>Practice groups are not permitted to work on t...</td>
      <td>en</td>
      <td>English</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3931fbe82a</td>
      <td>Des petites choses comme celles-là font une di...</td>
      <td>J'essayais d'accomplir quelque chose.</td>
      <td>fr</td>
      <td>French</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5622f0c60b</td>
      <td>you know they can't really defend themselves l...</td>
      <td>They can't defend themselves because of their ...</td>
      <td>en</td>
      <td>English</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86aaa48b45</td>
      <td>ในการเล่นบทบาทสมมุติก็เช่นกัน โอกาสที่จะได้แสด...</td>
      <td>เด็กสามารถเห็นได้ว่าชาติพันธุ์แตกต่างกันอย่างไร</td>
      <td>th</td>
      <td>Thai</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12115</th>
      <td>2b78e2a914</td>
      <td>The results of even the most well designed epi...</td>
      <td>All studies have the same amount of uncertaint...</td>
      <td>en</td>
      <td>English</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12116</th>
      <td>7e9943d152</td>
      <td>But there are two kinds of  the pleasure of do...</td>
      <td>But there are two kinds of the pleasure of doi...</td>
      <td>en</td>
      <td>English</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12117</th>
      <td>5085923e6c</td>
      <td>The important thing is to realize that it's wa...</td>
      <td>It cannot be moved, now or ever.</td>
      <td>en</td>
      <td>English</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12118</th>
      <td>fc8e2fd1fe</td>
      <td>At the west end is a detailed model of the who...</td>
      <td>The model temple complex is at the east end.</td>
      <td>en</td>
      <td>English</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>44301dfb14</td>
      <td>For himself he chose Atat??rk, or Father of th...</td>
      <td>Ataturk was the father of the Turkish nation.</td>
      <td>en</td>
      <td>English</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12120 rows × 6 columns</p>
</div>




```python
train_df.isna().sum()
```




    id            0
    premise       0
    hypothesis    0
    lang_abv      0
    language      0
    label         0
    dtype: int64




```python
train_df['language'].value_counts()
```




    English       6870
    Chinese        411
    Arabic         401
    French         390
    Swahili        385
    Urdu           381
    Vietnamese     379
    Russian        376
    Hindi          374
    Greek          372
    Thai           371
    Spanish        366
    Turkish        351
    German         351
    Bulgarian      342
    Name: language, dtype: int64




```python
train_df['label'].value_counts()
```




    0    4176
    2    4064
    1    3880
    Name: label, dtype: int64




```python
test_df
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
      <th>id</th>
      <th>premise</th>
      <th>hypothesis</th>
      <th>lang_abv</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c6d58c3f69</td>
      <td>بکس، کیسی، راہیل، یسعیاہ، کیلی، کیلی، اور کولم...</td>
      <td>کیسی کے لئے کوئی یادگار نہیں ہوگا, کولمین ہائی...</td>
      <td>ur</td>
      <td>Urdu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cefcc82292</td>
      <td>هذا هو ما تم نصحنا به.</td>
      <td>عندما يتم إخبارهم بما يجب عليهم فعله ، فشلت ال...</td>
      <td>ar</td>
      <td>Arabic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e98005252c</td>
      <td>et cela est en grande partie dû au fait que le...</td>
      <td>Les mères se droguent.</td>
      <td>fr</td>
      <td>French</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58518c10ba</td>
      <td>与城市及其他公民及社区组织代表就IMA的艺术发展进行对话&amp;amp</td>
      <td>IMA与其他组织合作，因为它们都依靠共享资金。</td>
      <td>zh</td>
      <td>Chinese</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c32b0d16df</td>
      <td>Она все еще была там.</td>
      <td>Мы думали, что она ушла, однако, она осталась.</td>
      <td>ru</td>
      <td>Russian</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5190</th>
      <td>5f90dd59b0</td>
      <td>نیند نے وعدہ کیا کہ موٹل نے سوال میں تحقیق کی.</td>
      <td>نیمیتھ کو موٹل کی تفتیش کے لئے معاوضہ دیا جارہ...</td>
      <td>ur</td>
      <td>Urdu</td>
    </tr>
    <tr>
      <th>5191</th>
      <td>f357a04e86</td>
      <td>The  rock  has a soft texture and can be bough...</td>
      <td>The rock is harder than most types of rock.</td>
      <td>en</td>
      <td>English</td>
    </tr>
    <tr>
      <th>5192</th>
      <td>1f0ea92118</td>
      <td>她目前的存在，并考虑到他与沃佛斯顿争执的本质，那是尴尬的。</td>
      <td>她在与Wolverstone的打斗结束后才在场的事实被看作是很尴尬的。</td>
      <td>zh</td>
      <td>Chinese</td>
    </tr>
    <tr>
      <th>5193</th>
      <td>0407b48afb</td>
      <td>isn't it i can remember i've only been here ei...</td>
      <td>I could see downtown Dallas from where I lived...</td>
      <td>en</td>
      <td>English</td>
    </tr>
    <tr>
      <th>5194</th>
      <td>16c2f2ab89</td>
      <td>In Hong Kong you can have a plate, or even a w...</td>
      <td>It's impossible to have a plate hand-painted t...</td>
      <td>en</td>
      <td>English</td>
    </tr>
  </tbody>
</table>
<p>5195 rows × 5 columns</p>
</div>




```python
test_df['language'].value_counts()
```




    English       2945
    Spanish        175
    Swahili        172
    Russian        172
    Urdu           168
    Greek          168
    Turkish        167
    Thai           164
    Arabic         159
    French         157
    German         152
    Chinese        151
    Hindi          150
    Bulgarian      150
    Vietnamese     145
    Name: language, dtype: int64




```python
sample_df
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
      <th>id</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c6d58c3f69</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cefcc82292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e98005252c</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58518c10ba</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c32b0d16df</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5190</th>
      <td>5f90dd59b0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5191</th>
      <td>f357a04e86</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5192</th>
      <td>1f0ea92118</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5193</th>
      <td>0407b48afb</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5194</th>
      <td>16c2f2ab89</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5195 rows × 2 columns</p>
</div>



### Modeling


```python
import tensorflow as tf
```


```python
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() 
```


```python
from transformers import TFAutoModel, AutoTokenizer
```

    [34m[1mwandb[0m: [33mWARNING[0m W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
    


```python
from tensorflow.keras.layers import Dense, Input
```


```python
from tensorflow.keras.models import Model
```


```python
from tensorflow.keras.optimizers import Adam
```


```python
def model_watson(strategy,transformer):
    with strategy.scope():
        transformer_encoder = TFAutoModel.from_pretrained(transformer)
        
        input_layer = Input(shape=(100,), dtype=tf.int32, name="input_layer")
        sequence_output = transformer_encoder(input_layer)[0]
        
        cls_token = sequence_output[:, 0, :]
        
        output_layer = Dense(3, activation='softmax')(cls_token)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
```


```python
model = model_watson(strategy,"distilbert-base-multilingual-cased")
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=466.0, style=ProgressStyle(description_…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=910749124.0, style=ProgressStyle(descri…


    
    


```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
```


```python
train_data = train_df[['premise', 'hypothesis']].values.tolist()
test_data = test_df[['premise', 'hypothesis']].values.tolist()
```


```python
train_encoded=tokenizer.batch_encode_plus(train_data,pad_to_max_length=True,max_length=100)
test_encoded=tokenizer.batch_encode_plus(test_data,pad_to_max_length=True,max_length=100)
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_valid, y_train, y_valid = train_test_split(train_encoded['input_ids'], train_df.label.values, test_size=0.2)
x_test = test_encoded['input_ids']
```


```python
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(2048).batch(20 * strategy.num_replicas_in_sync).prefetch(tf.data.experimental.AUTOTUNE))
valid_dataset = (tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(20 * strategy.num_replicas_in_sync).cache().prefetch(tf.data.experimental.AUTOTUNE))
test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(20 * strategy.num_replicas_in_sync))
```


```python
model.summary()
```


```python
history = model.fit(train_dataset,steps_per_epoch=len(train_df) // 20 * strategy.num_replicas_in_sync,validation_data=valid_dataset,epochs= 5)
```

### Our prediction output 


```python
predictions = model.predict(test_dataset, verbose=1)
sample_df['prediction'] = predictions.argmax(axis=1)
```


```python
import os
os.chdir(r'/kaggle/working')
```


```python
sample_df.to_csv(r'submission.csv',index= False)
```


```python
sample_df.head(10)
```
