import tensorflow as tf

import scipy.io as sio
import numpy as np
import math
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
tf.reset_default_graph()

arfa= 0.99
sigma = 0.00001


mat_path_F_out1 = 'CsiNet/data/DATA_HtestFout_all_T1.mat'
mat_path_F_out2 = 'CsiNet/data/DATA_HtestFout_all_T2.mat'
matt_F_out = sio.loadmat('CsiNet/data/DATA_HtestFout_all.mat')
mat_path_F_in1 = 'CsiNet/data/DATA_HtestFin_all_T1.mat'
mat_path_F_in2 = 'CsiNet/data/DATA_HtestFin_all_T2.mat'
matt_F_in = sio.loadmat('CsiNet/data/DATA_HtestFin_all.mat')
X_testout = matt_F_out['HF_all']
X_testout = np.reshape(X_testout, (len(X_testout), img_height, 125))
X_testin = matt_F_in['HF_all']
X_testin = np.reshape(X_testin, (len(X_testin), img_height, 125))
n1 = np.random.randn(len(X_testout),2048)
n2 = np.random.randn(len(X_testout),2048)
n3 = np.random.randn(len(X_testout),2048)

n1_C = n1[:, :1024] + 1j*n1[:, 1024:2048]
n1_F = np.reshape(n1_C, (len(n1_C), img_height, img_width))
n2_C = n2[:, :1024] + 1j*n2[:, 1024:2048]
n2_F = np.reshape(n2_C, (len(n2_C), img_height, img_width))
n3_C = n3[:, :1024] + 1j*n3[:, 1024:2048]
n3_F = np.reshape(n3_C, (len(n3_C), img_height, img_width))

nn1 = np.fft.fft(np.concatenate((n1_F, np.zeros((len(n1_C), img_height, 257-img_width))), axis=2), axis=2)
nn2 = np.fft.fft(np.concatenate((n1_F, np.zeros((len(n2_C), img_height, 257-img_width))), axis=2), axis=2)
nn3 = np.fft.fft(np.concatenate((n1_F, np.zeros((len(n3_C), img_height, 257-img_width))), axis=2), axis=2)
nn1 = nn1[:, :, 0:125]
nn2 = nn2[:, :, 0:125]
nn3 = nn3[:, :, 0:125]

X_testout0 = X_testout
X_testout1 = arfa*X_testout0+sigma*nn1
X_testout2 = arfa*X_testout1+sigma*nn2
X_testout3 = arfa*X_testout2+sigma*nn3
X_testoutFT1 = np.stack([X_testout0,X_testout1], axis=1)
X_testoutFT2 = np.stack([X_testout2,X_testout3], axis=1)
X_testin0 = X_testin
X_testin1 = arfa*X_testin0+sigma*nn1
X_testin2 = arfa*X_testin1+sigma*nn2
X_testin3 = arfa*X_testin2+sigma*nn3
X_testinFT1 = np.stack([X_testin0,X_testin1], axis=1)
X_testinFT2 = np.stack([X_testin2,X_testin3], axis=1)
print(np.shape(X_testinFT1))
sio.savemat(mat_path_F_in1, {'HT_all': X_testinFT1})
sio.savemat(mat_path_F_in2, {'HT_all': X_testinFT2})
sio.savemat(mat_path_F_out1, {'HT_all': X_testoutFT1})
sio.savemat(mat_path_F_out2, {'HT_all': X_testoutFT2})


mat_path_test_in = 'CsiNet/data/DATA_Htestin_T.mat'
matt_test_in = sio.loadmat('CsiNet/data/DATA_Htestin.mat')
mat_path_test_out = 'CsiNet/data/DATA_Htestout_T.mat'
matt_test_out = sio.loadmat('CsiNet/data/DATA_Htestout.mat')
x_testin = matt_test_in['HT']
x_testin = x_testin.astype('float32')
x_testout = matt_test_out['HT']
x_testout = x_testout.astype('float32')
x_testin = np.reshape(x_testin, (len(x_testin), img_channels, img_height, img_width))
x_testin = np.transpose(x_testin, (0, 2, 3, 1))
x_testin = np.reshape(x_testin, (len(x_testin), 2048))
x_testout = np.reshape(x_testout, (len(x_testout), img_channels, img_height, img_width))
x_testout = np.transpose(x_testout, (0, 2, 3, 1))
x_testout = np.reshape(x_testout, (len(x_testout), 2048))

x_testinT0 = x_testin
x_testinT1 = (arfa*x_testinT0+ sigma*(np.random.randn(len(x_testin), 2048)).astype('float32')).astype('float32')
x_testinT2 = (arfa*x_testinT1+ sigma*(np.random.randn(len(x_testin), 2048)).astype('float32')).astype('float32')
x_testinT3 = (arfa*x_testinT2+ sigma*(np.random.randn(len(x_testin), 2048)).astype('float32')).astype('float32')
x_testinT = np.stack([x_testinT0,x_testinT1,x_testinT2,x_testinT3], axis=1)
x_testinT = np.reshape(x_testinT, (len(x_testinT), 4, img_height, img_width, img_channels))

x_testoutT0 = x_testout
x_testoutT1 = (arfa*x_testoutT0+ sigma*(np.random.randn(len(x_testout), 2048)).astype('float32')).astype('float32')
x_testoutT2 = (arfa*x_testoutT1+ sigma*(np.random.randn(len(x_testout), 2048)).astype('float32')).astype('float32')
x_testoutT3 = (arfa*x_testoutT2+ sigma*(np.random.randn(len(x_testout), 2048)).astype('float32')).astype('float32')
x_testoutT = np.stack([x_testoutT0,x_testoutT1,x_testoutT2,x_testoutT3], axis=1)
x_testoutT = np.reshape(x_testoutT, (len(x_testoutT), 4, img_height, img_width, img_channels))

sio.savemat(mat_path_test_out, {'HT': x_testoutT})
sio.savemat(mat_path_test_in, {'HT': x_testinT})

mat_path_train_in = 'CsiNet/data/DATA_Htrainin_T.mat'
matt_train_in = sio.loadmat('CsiNet/data/DATA_Htrainin.mat')
mat_path_train_out = 'CsiNet/data/DATA_Htrainout_T.mat'
matt_train_out = sio.loadmat('CsiNet/data/DATA_Htrainout.mat')
x_trainin = matt_train_in['HT']
x_trainin = x_trainin.astype('float32')
x_trainout = matt_train_out['HT']
x_trainout = x_trainout.astype('float32')
x_trainin = np.reshape(x_trainin, (len(x_trainin), img_channels, img_height, img_width))
x_trainin = np.transpose(x_trainin, (0, 2, 3, 1))
x_trainin = np.reshape(x_trainin, (len(x_trainin), 2048))
x_trainout = np.reshape(x_trainout, (len(x_trainout), img_channels, img_height, img_width))
x_trainout = np.transpose(x_trainout, (0, 2, 3, 1))
x_trainout = np.reshape(x_trainout, (len(x_trainout), 2048))
x_traininT0 = x_trainin
x_traininT1 = (arfa*x_traininT0+ sigma*(np.random.randn(len(x_trainin), 2048)).astype('float32')).astype('float32')
x_traininT2 = (arfa*x_traininT1+ sigma*(np.random.randn(len(x_trainin), 2048)).astype('float32')).astype('float32')
x_traininT3 = (arfa*x_traininT2+ sigma*(np.random.randn(len(x_trainin), 2048)).astype('float32')).astype('float32')
x_traininT = np.stack([x_traininT0,x_traininT1,x_traininT2,x_traininT3], axis=1)
x_traininT = np.reshape(x_traininT, (len(x_traininT), 4, img_height, img_width, img_channels))
x_trainoutT0 = x_trainout
x_trainoutT1 = (arfa*x_trainoutT0+ sigma*(np.random.randn(len(x_trainout), 2048)).astype('float32')).astype('float32')
x_trainoutT2 = (arfa*x_trainoutT1+ sigma*(np.random.randn(len(x_trainout), 2048)).astype('float32')).astype('float32')
x_trainoutT3 = (arfa*x_trainoutT2+ sigma*(np.random.randn(len(x_trainout), 2048)).astype('float32')).astype('float32')
x_trainoutT = np.stack([x_trainoutT0,x_trainoutT1,x_trainoutT2,x_trainoutT3], axis=1)
x_trainoutT = np.reshape(x_trainoutT, (len(x_trainoutT), 4, img_height, img_width, img_channels))
sio.savemat(mat_path_train_out, {'HT': x_trainoutT})
sio.savemat(mat_path_train_in, {'HT': x_traininT})

mat_path_val_in = 'CsiNet/data/DATA_Hvalin_T.mat'
matt_val_in = sio.loadmat('CsiNet/data/DATA_Hvalin.mat')
mat_path_val_out = 'CsiNet/data/DATA_Hvalout_T.mat'
matt_val_out = sio.loadmat('CsiNet/data/DATA_Hvalout.mat')
x_valin = matt_val_in['HT']
x_valin = x_valin.astype('float32')
x_valout = matt_val_out['HT']
x_valout = x_valout.astype('float32')
x_valin = np.reshape(x_valin, (len(x_valin), img_channels, img_height, img_width))
x_valin = np.transpose(x_valin, (0, 2, 3, 1))
x_valin = np.reshape(x_valin, (len(x_valin), 2048))
x_valout = np.reshape(x_valout, (len(x_valout), img_channels, img_height, img_width))
x_valout = np.transpose(x_valout, (0, 2, 3, 1))
x_valout = np.reshape(x_valout, (len(x_valout), 2048))
x_valinT0 = x_valin
x_valinT1 = (arfa*x_valinT0+ sigma*(np.random.randn(len(x_valin), 2048)).astype('float32')).astype('float32')
x_valinT2 = (arfa*x_valinT1+ sigma*(np.random.randn(len(x_valin), 2048)).astype('float32')).astype('float32')
x_valinT3 = (arfa*x_valinT2+ sigma*(np.random.randn(len(x_valin), 2048)).astype('float32')).astype('float32')
x_valinT = np.stack([x_valinT0,x_valinT1,x_valinT2,x_valinT3], axis=1)
x_valinT = np.reshape(x_valinT, (len(x_valinT), 4, img_height, img_width, img_channels))
x_valoutT0 = x_valout
x_valoutT1 = (arfa*x_valoutT0+ sigma*(np.random.randn(len(x_valout), 2048)).astype('float32')).astype('float32')
x_valoutT2 = (arfa*x_valoutT1+ sigma*(np.random.randn(len(x_valout), 2048)).astype('float32')).astype('float32')
x_valoutT3 = (arfa*x_valoutT2+ sigma*(np.random.randn(len(x_valout), 2048)).astype('float32')).astype('float32')
x_valoutT = np.stack([x_valoutT0,x_valoutT1,x_valoutT2,x_valoutT3], axis=1)
x_valoutT = np.reshape(x_valoutT, (len(x_valoutT), 4, img_height, img_width, img_channels))
sio.savemat(mat_path_val_out, {'HT': x_valoutT})
sio.savemat(mat_path_val_in, {'HT': x_valinT})