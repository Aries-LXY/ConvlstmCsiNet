# ConvlstmCsiNet
### Code for "Spatio-Temporal Representation with Deep Recurrent Network in MIMO CSI Feedback"

## Introduction
This repo contains code accompaning the paper, "Spatio-Temporal Representation with Deep Recurrent Network in MIMO CSI Feedback", IEEE Wireless Communications Letters, 2020. [Online]. Available: 
https://ieeexplore.ieee.org/document/8951228  

## Dependencies
This code requires the following:
* Python 3.5 (or 3.6)
* TensorFlow v1.4+
* Keras v2.1.1+
* Numpy

## Data
The data is available online:  
https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing.   
After you got the data, create a folder named "data" and put the data files in it. Code files are outside the folder "data". Then run "data_transform.py" to generate the time-varying CSI according to the expression(3) in paper:  
![](http://spro.so.com/searchthrow/api/midpage/throw?ls=s112c46189d&lm_extend=ctype:3&ctype=3&q=github%E5%88%9B%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9&rurl=http%3A%2F%2Fwww.bubuko.com%2Finfodetail-3050701.html&img=http%3A%2F%2Fimage.bubuko.com%2Finfo%2F201905%2F20190508114839932093.png)
![expression](https://github.com/Aries-LXY/ConvlstmCsiNet/blob/master/result/1.png)
## Model code
The four model code files stand for the four models mentioned in paper and named after with the corresponding model name, respectively. The code runs for the model training and is set to check the performance index(NMSE and cosine similarity) every several epochs to find the best model.   

A few preparations should be taken before the training: Create a folder named "result" to save the models and create the subfolders corresponding with each model in each situation:  

CSI/  
>.py  
>result/  
>> ConvlstmCsiNet_A_indoor/  
>> ConvlstmCsiNet_B_outdoor/    
>>>dim_256/  
>>>dim_512/  
>>>>.h5  
>>>>.json   
>>>>.png  
