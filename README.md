# AODNet-Based-Image-Haze-Removal
Single Image Haze Removal Using AODNet in Pytorch
* Implementation of Boyi Li's Paper [**An All-in-One Network for Dehazing and Beyond**](https://arxiv.org/pdf/1707.06543.pdf) on ICCV 2017.<br>

---
#### Contents

1. [Dependency](#Dependency)
1. [Usage](#Usage)
1. [Results](#Results)
1. [References](#References)
---

#### Dependency
###### &emsp;Python&ensp;3.6 or newer<br>
###### &emsp;torch == 1.7.1<br>
###### &emsp;pillow == 5.1.0<br>
###### &emsp;numpy == 1.14.3<br>
###### &emsp;matplotlib == 2.2.2<br>

#### Usage
* How to Use : download the whole project and run **inference.py**
* folder ./saved_models : where the trained models are saved, files are in .pth format.
* folder ./data : the training data.
* folder ./test_images : some testing images that appear in the original paper.
* data.py : function that loads the training data.
* train.py : train a new AODNet from scratch using training data saved in folder ./data/.
* model.py : definition of AODNet.
* utils.py : some auxiliary functions.
* inference.py : single image dehazing using the trained AODNet.

#### Results
* [**The Leaning Tower of Pisa  5.5°**](https://github.com/ForeverPs/content-aware-rotation/blob/master/image/image7.jpg)<br>
<img src= https://github.com/ForeverPs/content-aware-rotation/blob/master/eq/pisa_tower.jpg /><br><br>

* [**Palace Tower  -6.1°**](https://github.com/ForeverPs/content-aware-rotation/blob/master/image/image2.png)<br>
<img src= https://github.com/ForeverPs/content-aware-rotation/blob/master/eq/palace_tower.jpg /><br><br>

* [**House Building  -5.8°**](https://github.com/ForeverPs/content-aware-rotation/blob/master/image/image1.png)<br>
<img src= https://github.com/ForeverPs/content-aware-rotation/blob/master/eq/house.jpg /><br><br>

* [**The Oriental Pearl Tower  1.8°**](https://github.com/ForeverPs/content-aware-rotation/blob/master/image/image8.jpg)<br>
<img src= https://github.com/ForeverPs/content-aware-rotation/blob/master/eq/shanghai.jpg /><br><br>


#### References
##### Author&ensp;:&ensp;Kaiming He, Huiwen Chang, Jian Sun<br>
* ###### &ensp;[Content-Aware Rotation----ICCV 2013](http://kaiminghe.com/publications/iccv13car.pdf)<br>
##### Matlab Version<br>
* ###### &ensp;[iRotate](https://github.com/yuchien302/iRotate)<br>