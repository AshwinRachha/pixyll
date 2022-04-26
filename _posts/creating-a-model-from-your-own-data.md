
In this Post we will create an Annotated Image Dataset of our own for the task of Image Classification. A lot of tutorials out there cover how the use those complex Computer Vision Models on the most basic Datasets. The MNIST and CIFAR10 datasets must have been dragged way to much than they should have and now its time we move away from them and work on some real world data. 

The basic steps for data collections we are going to take is:

1. Use DuckDuckGo to search for images of 'batman'
2. Use DuckDuckGo to search for images of 'superman'
3. Fine-tune a pretrained neural network to recognize these two groups.
4. Try running this model on a picture of a bird and see if it works. 


```python
from fastcore.all import *
import time

def search_images(term, max_images=200):
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    while len(urls)<max_images and 'next' in data:
        data = urljson(requestUrl,data=params)
        urls.update(L(data['results']).itemgot('image'))
        requestUrl = url + data['next']
        time.sleep(0.2)
    return L(urls)[:max_images]
```


```python
urls = search_images('batman photos', max_images=1)
urls[0]
```


```python
from fastdownload import download_url
dest = 'batman.jpg'
download_url(urls[0], dest, show_progress = False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256, 256)
```


```python
download_url(search_images('superman photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```


```python
searches = 'batman', 'superman'
path = Path('batman_v_superman') # Haha, Pun Intended
for hero in searches:
    dest = path / hero
    #print(dest)
    dest.mkdir(exist_ok = True, parents = True)
    download_images(dest, urls = search_images(f'{hero} photo'))
    resize_images(path/hero, max_size = 400, dest = path / hero)
```


```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```


```python
dls = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    get_y = parent_label,
    item_tfms = [Resize(192, method = 'squish')]
).dataloaders(path)

dls.show_batch(max_n = 6)
```


```python
learn = learn = cnn_learner(dls, resnet50, metrics=[accuracy])
learn.fine_tune(2)
```


```python
learn.lr_find()
```


```python
learn.fine_tune(15, base_lr = 10e-5)
```


```python
interp = Interpretation.from_learner(learn)
```


```python
interp.plot_top_losses(k = 5)
```


```python
interp.show_results([1,5,10,15])
```


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```


```python
pred = learn.predict('./batman.jpg')
```


```python
from PIL import Image
import matplotlib.pyplot as plt

def predict(learn, image_path):
    plt.figure(figsize = (12, 5))
    plt.imshow(Image.open(image_path))
    pred = learn.predict(image_path)
    print(f'Your Superhero is {pred[0]}')
    plt.show()
```


```python
predict(learn, './batman.jpg')
```


```python

```
