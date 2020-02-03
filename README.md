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
