---
layout:     post
title:      Covid 19 Xray Detection
date:       2020-11-11 11:21:29
summary:    We learn how to classify covid, pneumonia and no findings images using efficientnets.
categories: jekyll pixyll
---


The coronavirus outbreak has caused a devastating effect on people all around the world and has infected millions.

The exponential escalation of the spread of the disease makes it emergent for appropriate screening methods to detect the

disease and take steps in mitigating it. The conventional testing technique involves the use of Reverse-Transcriptase

Polymerase Chain Reaction (RT-PCR). Due to limited sensitivity it is more prone to providing high false negative rates. Also

due to a high turnaround time (6-9 hours) and a high cost, an alternative approach for screening is called for. Chest

radiographs are the most frequently used imaging procedures in radiology. They are cheaper compared to CT scans and are

more readily available and accessible to the public. Application of advanced artificial intelligence (AI) techniques coupled with

radiological imaging can be helpful for the accurate detection of this disease. In this projecct we will study how state of the art model -  EfficientNetB7 is applied to the problem of classification.



To check out my research paper on this work please refer to the following url:



<a href = 'https://www.irjet.net/archives/V7/i11/IRJET-V7I1144.pdf' > Here </a>




{% highlight ruby %}

os.listdir('/content/gdrive/My Drive/Final XRAY')

{% endhighlight %}




    ['models', 'TEST', 'TRAIN']




{% highlight ruby %}

%reload_ext autoreload
%autoreload 2
%matplotlib inline

{% endhighlight %}

*Next up we will import all the necessary libraries. We will use the fastai library which consists of various state of the art pretrained models for image classification and a lot of utilitarian tools which make deep learning easier with less lines of code!*


{% highlight ruby %}
from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from fastai.callbacks import *
{% endhighlight %}

# *EfficientNets*



With the rise of transfer learning, the essentiality of scaling has been deeply realised for enhancing the performance as well as efficieny of models. Traditionaly scaling can be done in three dimensions viz. depth, width and resolution in terms of convolutional neural networks. Depth scaling pertains to increasing the number of layers in the model, making it more deeper; width scaling makes the model wider (one possible way is to increase the number of channels in a layer) and resolution scaling means using high resolution images so that features are more fine-grained. Each method applied individually has some drawbacks such as in depth scaling we have the problem of vanishing gradients and in width scaling the accuracy saturates after a point and there is a limit to increasing resolution of images and a slight increase doesnt result in significant improvement of performance. Hence Efficientnets are proposed to deal with balancing all dimensions of a network during CNN scaling for getting improved accuracy and efficieny. The authors proposed a simple yet very effective scaling technique which uses a compound coefficientto uniformly scale network width, depth, and resolution in a principled way. We used the pytorch wrapper for efficientnets. To install run the following command:


{% highlight ruby %}
pip install efficientnet-pytorch

import warnings
warnings.filterwarnings('ignore')
{% endhighlight %}


Now we will define an ImageDataBunch which gets our image data into DataLoaders over which our models can be fit. We use the from_folder function to get the images from our folder which is subdivided into Train and Test folders. For image transformations we use the get_transforms function which performs various augmentations on our images viz rotation, , horizontal flip, vertical flip, zooming, warping, affine transformation etc. 



The input target size of the images is defined as 224 since most models are compatible with this size including efficientnets. We maintain a Batch Size of 32 in order to ensure efficient usage of memory. 


{% highlight ruby %}
path = '/content/gdrive/My Drive/Final XRAY'
np.random.seed(44)
data = ImageDataBunch.from_folder(path, train="TRAIN", valid ="TEST",
        ds_tfms=get_transforms(), size=(224,224), bs=32, num_workers=4).normalize()
{% endhighlight %}


{% highlight ruby %}
data.classes, data.c
{% endhighlight %}




    (['COVID', 'NON-COVID', 'PNEUMONIA'], 3)




{% highlight ruby %}
data.train_ds
{% endhighlight %}




    LabelList (1482 items)
    x: ImageList
    Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
    y: CategoryList
    COVID,COVID,COVID,COVID,COVID
    Path: /content/gdrive/My Drive/Final XRAY




{% highlight ruby %}
data.valid_ds
{% endhighlight %}




    LabelList (217 items)
    x: ImageList
    Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
    y: CategoryList
    COVID,COVID,COVID,COVID,COVID
    Path: /content/gdrive/My Drive/Final XRAY



We can visualize our Training data in the following cell. 

The train and validation split of the respective images is as follows.









Split COVID-19 NO-FINDINGS PNEUMONIA





---





Training Set 454 610 418





---





Testing Set 41 94 82







---












{% highlight ruby %}
data.show_batch(rows=3, figsize=(10,10))
{% endhighlight %}


![png](/blogpost/images/EfficientNet_B7_blog_17_0.png)



{% highlight ruby %}
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
{% endhighlight %}

We will use EfficientNetB7 for training. 


{% highlight ruby %}
model = EfficientNet.from_pretrained('efficientnet-b7')
{% endhighlight %}

    Loaded pretrained weights for efficientnet-b7
    

To better capture the essence of the performance of the model, along with traditional metrics we use the top2 accuracy which is predicted true for each image if the actual label of the image falls in the top 2 softmax probabilities of the model. 


{% highlight ruby %}
top_5 = partial(top_k_accuracy, k=2)

learn = Learner(data, model, metrics=[accuracy, top_5, error_rate], loss_func=LabelSmoothingCrossEntropy(), callback_fns=[ShowGraph, ReduceLROnPlateauCallback]).to_fp16()
{% endhighlight %}


{% highlight ruby %}
learn.fit_one_cycle(4)
{% endhighlight %}


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>top_k_accuracy</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.795816</td>
      <td>11.563503</td>
      <td>0.382488</td>
      <td>0.820276</td>
      <td>0.617512</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.872262</td>
      <td>4.012103</td>
      <td>0.483871</td>
      <td>0.852535</td>
      <td>0.516129</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.540921</td>
      <td>1.864752</td>
      <td>0.705069</td>
      <td>0.935484</td>
      <td>0.294931</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.356434</td>
      <td>1.389227</td>
      <td>0.834101</td>
      <td>1.000000</td>
      <td>0.165899</td>
      <td>01:02</td>
    </tr>
  </tbody>
</table>



![png](/blogpost/images/EfficientNet_B7_blog_23_1.png)



{% highlight ruby %}
learn.recorder.plot_losses()
{% endhighlight %}


![png](/blogpost/images/EfficientNet_B7_blog_24_0.png)


{% highlight ruby %}
learn.recorder.plot_metrics()
{% endhighlight %}


![png](/blogpost/images/EfficientNet_B7_blog_25_0.png)


Fastai comes with a very important utility of finding an appropriate learning rate and then fine tuning our models later with the set learning rate. This boosts the performance of the models significantly. 


{% highlight ruby %}
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion = True)
{% endhighlight %}



  
    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>top_k_accuracy</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.218536</td>
      <td>#na#</td>
      <td>00:55</td>
    </tr>
  </tbody>
</table><p>


    


    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    Min numerical gradient: 6.92E-06
    Min loss divided by 10: 6.31E-08
    


![png](/blogpost/images/EfficientNet_B7_blog_27_2.png)



{% highlight ruby %}
learn.fit_one_cycle(100, max_lr=slice(6.82e-6))
{% endhighlight %}


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>top_k_accuracy</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.211556</td>
      <td>1.233234</td>
      <td>0.921659</td>
      <td>1.000000</td>
      <td>0.078341</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.222816</td>
      <td>1.172435</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.244095</td>
      <td>1.147769</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.220592</td>
      <td>1.137416</td>
      <td>0.949309</td>
      <td>1.000000</td>
      <td>0.050691</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.212929</td>
      <td>1.136014</td>
      <td>0.949309</td>
      <td>1.000000</td>
      <td>0.050691</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.207222</td>
      <td>1.140133</td>
      <td>0.949309</td>
      <td>1.000000</td>
      <td>0.050691</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.212577</td>
      <td>1.142727</td>
      <td>0.949309</td>
      <td>1.000000</td>
      <td>0.050691</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.209110</td>
      <td>1.148496</td>
      <td>0.949309</td>
      <td>1.000000</td>
      <td>0.050691</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.213428</td>
      <td>1.153768</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.212053</td>
      <td>1.158563</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.207630</td>
      <td>1.160250</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>11</td>
      <td>1.210703</td>
      <td>1.161393</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>12</td>
      <td>1.210255</td>
      <td>1.163532</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>13</td>
      <td>1.210563</td>
      <td>1.162763</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>14</td>
      <td>1.199785</td>
      <td>1.161451</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>15</td>
      <td>1.204878</td>
      <td>1.165974</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1.200078</td>
      <td>1.163577</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1.196592</td>
      <td>1.163785</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>18</td>
      <td>1.191627</td>
      <td>1.167737</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1.196668</td>
      <td>1.164468</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>20</td>
      <td>1.193052</td>
      <td>1.161262</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>21</td>
      <td>1.192981</td>
      <td>1.164138</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>22</td>
      <td>1.184990</td>
      <td>1.163553</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>23</td>
      <td>1.190937</td>
      <td>1.170372</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>24</td>
      <td>1.186672</td>
      <td>1.174670</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>25</td>
      <td>1.170076</td>
      <td>1.174799</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>26</td>
      <td>1.174368</td>
      <td>1.171321</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>27</td>
      <td>1.166383</td>
      <td>1.174107</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>28</td>
      <td>1.174226</td>
      <td>1.177173</td>
      <td>0.940092</td>
      <td>0.995392</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>29</td>
      <td>1.170711</td>
      <td>1.176359</td>
      <td>0.940092</td>
      <td>0.995392</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>30</td>
      <td>1.162668</td>
      <td>1.175813</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>31</td>
      <td>1.162512</td>
      <td>1.178215</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>32</td>
      <td>1.168516</td>
      <td>1.180676</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>33</td>
      <td>1.146605</td>
      <td>1.181284</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>34</td>
      <td>1.154101</td>
      <td>1.174066</td>
      <td>0.949309</td>
      <td>0.995392</td>
      <td>0.050691</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>35</td>
      <td>1.159191</td>
      <td>1.185242</td>
      <td>0.944700</td>
      <td>0.995392</td>
      <td>0.055300</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>36</td>
      <td>1.149488</td>
      <td>1.187768</td>
      <td>0.944700</td>
      <td>0.995392</td>
      <td>0.055300</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>37</td>
      <td>1.145655</td>
      <td>1.181144</td>
      <td>0.944700</td>
      <td>0.995392</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>38</td>
      <td>1.152014</td>
      <td>1.185391</td>
      <td>0.944700</td>
      <td>0.995392</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>39</td>
      <td>1.153827</td>
      <td>1.186132</td>
      <td>0.944700</td>
      <td>0.995392</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>40</td>
      <td>1.146247</td>
      <td>1.193885</td>
      <td>0.944700</td>
      <td>0.995392</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>41</td>
      <td>1.143121</td>
      <td>1.197211</td>
      <td>0.940092</td>
      <td>0.995392</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>42</td>
      <td>1.135645</td>
      <td>1.209570</td>
      <td>0.930876</td>
      <td>0.995392</td>
      <td>0.069124</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>43</td>
      <td>1.135074</td>
      <td>1.196061</td>
      <td>0.935484</td>
      <td>0.995392</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>44</td>
      <td>1.151849</td>
      <td>1.193242</td>
      <td>0.935484</td>
      <td>0.995392</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>45</td>
      <td>1.134802</td>
      <td>1.188756</td>
      <td>0.940092</td>
      <td>0.995392</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>46</td>
      <td>1.138158</td>
      <td>1.192837</td>
      <td>0.935484</td>
      <td>0.995392</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>47</td>
      <td>1.129135</td>
      <td>1.186649</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>48</td>
      <td>1.133389</td>
      <td>1.193150</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>49</td>
      <td>1.125778</td>
      <td>1.191461</td>
      <td>0.944700</td>
      <td>1.000000</td>
      <td>0.055300</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>50</td>
      <td>1.131475</td>
      <td>1.198874</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>51</td>
      <td>1.141228</td>
      <td>1.205654</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>52</td>
      <td>1.129196</td>
      <td>1.201473</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>53</td>
      <td>1.134563</td>
      <td>1.200921</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>54</td>
      <td>1.127136</td>
      <td>1.202468</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>55</td>
      <td>1.125550</td>
      <td>1.201220</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>56</td>
      <td>1.117874</td>
      <td>1.212310</td>
      <td>0.930876</td>
      <td>1.000000</td>
      <td>0.069124</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>57</td>
      <td>1.118835</td>
      <td>1.209787</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>58</td>
      <td>1.140433</td>
      <td>1.212276</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>59</td>
      <td>1.125858</td>
      <td>1.208665</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>60</td>
      <td>1.112389</td>
      <td>1.208547</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>61</td>
      <td>1.117704</td>
      <td>1.207139</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>62</td>
      <td>1.115862</td>
      <td>1.201950</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>63</td>
      <td>1.114401</td>
      <td>1.203300</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>64</td>
      <td>1.104069</td>
      <td>1.210942</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>65</td>
      <td>1.111491</td>
      <td>1.204915</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>66</td>
      <td>1.111911</td>
      <td>1.203857</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>67</td>
      <td>1.106273</td>
      <td>1.206619</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>68</td>
      <td>1.111764</td>
      <td>1.205026</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>69</td>
      <td>1.104027</td>
      <td>1.209384</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>70</td>
      <td>1.096211</td>
      <td>1.209802</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>71</td>
      <td>1.104607</td>
      <td>1.211365</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>72</td>
      <td>1.101600</td>
      <td>1.207773</td>
      <td>0.940092</td>
      <td>1.000000</td>
      <td>0.059908</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>73</td>
      <td>1.105587</td>
      <td>1.210494</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>74</td>
      <td>1.099172</td>
      <td>1.213641</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>75</td>
      <td>1.101368</td>
      <td>1.213647</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>76</td>
      <td>1.101365</td>
      <td>1.209697</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>77</td>
      <td>1.111905</td>
      <td>1.212039</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>78</td>
      <td>1.097285</td>
      <td>1.214438</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>79</td>
      <td>1.098992</td>
      <td>1.217427</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:04</td>
    </tr>
    <tr>
      <td>80</td>
      <td>1.110196</td>
      <td>1.219014</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>81</td>
      <td>1.110621</td>
      <td>1.218604</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>82</td>
      <td>1.103908</td>
      <td>1.217055</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:04</td>
    </tr>
    <tr>
      <td>83</td>
      <td>1.103807</td>
      <td>1.217934</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>84</td>
      <td>1.106353</td>
      <td>1.220562</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>85</td>
      <td>1.102028</td>
      <td>1.218156</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>86</td>
      <td>1.098312</td>
      <td>1.219447</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>87</td>
      <td>1.100400</td>
      <td>1.220807</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>88</td>
      <td>1.099732</td>
      <td>1.220042</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>89</td>
      <td>1.098915</td>
      <td>1.219582</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>90</td>
      <td>1.093679</td>
      <td>1.217780</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>91</td>
      <td>1.096850</td>
      <td>1.215652</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>92</td>
      <td>1.098493</td>
      <td>1.219773</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>93</td>
      <td>1.098292</td>
      <td>1.217808</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>94</td>
      <td>1.098705</td>
      <td>1.219644</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>95</td>
      <td>1.091730</td>
      <td>1.222525</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>96</td>
      <td>1.093058</td>
      <td>1.221992</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>97</td>
      <td>1.100906</td>
      <td>1.220918</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:04</td>
    </tr>
    <tr>
      <td>98</td>
      <td>1.100290</td>
      <td>1.222682</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>99</td>
      <td>1.108818</td>
      <td>1.222131</td>
      <td>0.935484</td>
      <td>1.000000</td>
      <td>0.064516</td>
      <td>01:03</td>
    </tr>
  </tbody>
</table>



![png](/blogpost/images/EfficientNet_B7_blog_28_1.png)




{% highlight ruby %}
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')
{% endhighlight %}







![png](/blogpost/images/EfficientNet_B7_blog_29_1.png)


{% highlight ruby %}
probs,targets = learn.get_preds(ds_type=DatasetType.Valid) # Predicting without TTA


probs = np.argmax(probs, axis=1)
correct = 0
for idx, pred in enumerate(probs):
    if pred == targets[idx]:
        correct += 1
accuracy = correct / len(probs)
print(len(probs), correct, accuracy)

from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.inf) # shows whole confusion matrix
cm1 = confusion_matrix(targets, probs)
print(cm1)

from sklearn.metrics import classification_report
y_true1 = targets
y_pred1 = probs
target_names = ['Covid-19', 'No_findings', 'Pneumonia']
print(classification_report(y_true1, y_pred1, target_names=target_names))
{% endhighlight %}

217 203 0.9354838709677419
 [[41  0  0]
 [ 0 94  0]
  [ 2 12 68]]
               precision    recall  f1-score   support
    
  Covid-19       0.95      1.00      0.98        41
  No_findings    0.89      1.00      0.94        94
  Pneumonia      1.00      0.83      0.91        82
    
   accuracy                           0.94       217
   macro avg       0.95      0.94      0.94       217
   weighted avg    0.94      0.94      0.93       217
   



```
learn.save('/content/gdrive/My Drive/Models'+ 'EfficientNetB7')
```

```
preds,y, loss = learn.get_preds(with_loss=True)

```


```

from sklearn.metrics import roc_curve, auc

probs = np.exp(preds[:,1])

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

roc_auc = auc(fpr, tpr)
print('ROC area is {0}'.format(roc_auc))
```
ROC area is 0.9987891368275386
    

{% highlight ruby %}
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (DenseNet 169) ')
plt.legend(loc="lower right")
{% endhighlight %}



![png](/blogpost/images/EfficientNet_B7_blog_36_1.png)



