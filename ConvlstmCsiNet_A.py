import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv3D, LSTM, ConvLSTM2D, add, LeakyReLU, \
    Concatenate, Lambda, Activation, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, TensorBoard
import scipy.io as sio
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

envir = 'indoor'
reduction_time = 4
encoded_dim = 128

T = 4
test_peroid = 10


def slice(x, index):
    return x[:, :, :, :, index]


def expand_dim(x, axis=-1):
    return K.expand_dims(x, axis)


def separable_conv3d(x, input_dim, output_dim, feature_sride=1):
    Z = []
    for i in range(input_dim):
        z = Lambda(slice, arguments={'index': i})(x)
        z = Lambda(expand_dim, arguments={'axis': -1})(z)
        z = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, feature_sride, feature_sride),
                   input_shape=(T, img_height, img_width, 1), padding='same', data_format='channels_last')(z)
        Z.append(z)
    m = Concatenate(axis=-1)(Z)
    m = Conv3D(filters=output_dim, kernel_size=(1, 1, 1), padding='same', data_format='channels_last')(m)
    return m


def residual_network():
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y

        y = separable_conv3d(y, 2, 8, 1)
        y = add_common_layers(y)

        y = separable_conv3d(y, 8, 16, 1)
        y = add_common_layers(y)

        y = separable_conv3d(y, 16, 2, 1)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    def LSTM_Dense_block(z, output_dim):
        shortcut = Dense(output_dim, activation='linear')(z)
        shortcut = Dropout(0.3)(shortcut)
        x = LSTM(units=output_dim, return_sequences='True')(z)

        x = add([shortcut, x])
        return x

    # encoder
    inp = Input(shape=(T, img_height, img_width, img_channels))

    x = ConvLSTM2D(filters=2, kernel_size=(3, 3), padding="same", data_format='channels_last',
                   return_sequences='True')(inp)
    x = add_common_layers(x)
    x = Conv3D(2, (3, 3, 3), padding="same", data_format='channels_last')(x)
    x = Reshape((T, img_total))(x)
    encoded = LSTM_Dense_block(x, encoded_dim)

    # decoder
    x = LSTM_Dense_block(encoded, img_total)
    x = Reshape((T, img_height, img_width, img_channels))(x)
    x = residual_block_decoded(x)
    x = residual_block_decoded(x)
    x = separable_conv3d(x, 2, 2, 1)
    out = Activation('sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model


def mseT(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(-4, -1))


autoencoder = residual_network()
autoencoder.compile(optimizer='adam', loss=mseT)
print(autoencoder.summary())

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('CsiNet/data/DATA_Htrainin_T.mat')
    x_train = mat['HT']  # array
    mat = sio.loadmat('CsiNet/data/DATA_Hvalin_T.mat')
    x_val = mat['HT']  # array
    mat = sio.loadmat('CsiNet/data/DATA_Htestin_T.mat')
    x_test = mat['HT']  # array

    mat1 = sio.loadmat('CsiNet/data/DATA_HtestFin_all_T1.mat')
    mat2 = sio.loadmat('CsiNet/data/DATA_HtestFin_all_T2.mat')
    X_test1 = mat1['HT_all']  # array
    X_test2 = mat2['HT_all']
    X_test1 = np.reshape(X_test1, (len(X_test1), 2, img_height, 125))
    X_test2 = np.reshape(X_test2, (len(X_test2), 2, img_height, 125))
elif envir == 'outdoor':
    mat = sio.loadmat('CsiNet/data/DATA_Htrainout_T.mat')
    x_train = mat['HT']  # array
    mat = sio.loadmat('CsiNet/data/DATA_Hvalout_T.mat')
    x_val = mat['HT']  # array
    mat = sio.loadmat('CsiNet/data/DATA_Htestout_T.mat')
    x_test = mat['HT']  # array

    mat1 = sio.loadmat('CsiNet/data/DATA_HtestFout_all_T1.mat')
    mat2 = sio.loadmat('CsiNet/data/DATA_HtestFout_all_T2.mat')
    X_test1 = mat1['HT_all']
    X_test2 = mat2['HT_all']
    X_test1 = np.reshape(X_test1, (len(X_test1), 2, img_height, 125))
    X_test2 = np.reshape(X_test2, (len(X_test2), 2, img_height, 125))
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = np.reshape(x_train, (len(x_train), T, img_height, img_width, img_channels))
x_val = np.reshape(x_val, (len(x_val), T, img_height, img_width, img_channels))
x_test = np.reshape(x_test, (len(x_test), T, img_height, img_width, img_channels))
X_test = np.concatenate([X_test1, X_test2], axis=1)
X_test = np.reshape(X_test, (len(X_test), T, img_height, 125))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = {'batch': [], 'epoch': []}
        self.losses_val = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses_train['batch'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val['epoch'].append(logs.get('val_loss'))
        self.losses_train['epoch'].append(logs.get('loss'))


train_Loss = []
val_Loss = []
nmse_all = []
rho_all = []
names = locals()
# test the model every test_peroid epochs
for period in range(int(1500 / test_peroid)):
    print('Period is', period)
    # set learning rate
    if period < 100:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    elif period < 120:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0005)
    else:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    names['history' + str(period)] = LossHistory()
    file = '_epoch' + str(period * test_peroid) + time.strftime('_%m_%d')
    path = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/TensorBoard_' + file + '.csv'
    autoencoder.fit(x_train, x_train,
                    epochs=test_peroid,
                    batch_size=250,

                    shuffle=True,
                    validation_data=(x_val, x_val),
                    callbacks=[names.get('history' + str(period)),
                               reduce_lr,
                               TensorBoard(log_dir=path)])

    # save
    # save and print loss
    filename = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/trainLoss_' + file + '.csv'
    trainloss_history = np.array(names.get('history' + str(period)).losses_train['epoch'])
    train_Loss = np.append(train_Loss, trainloss_history)
    train_Loss = np.reshape(train_Loss, (-1,))
    np.savetxt(filename, train_Loss, delimiter=",")

    filename = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/valLoss_' + file + '.csv'
    valloss_history = np.array(names.get('history' + str(period)).losses_val['epoch'])
    val_Loss = np.append(val_Loss, valloss_history)
    val_Loss = np.reshape(val_Loss, (-1,))
    np.savetxt(filename, val_Loss, delimiter=",")

    iters_train = range(len(train_Loss))
    plt.figure()
    plt.plot(iters_train, train_Loss, 'g', label='ConvlstmCsiNet_Conv_trainloss')
    plt.plot(iters_train, val_Loss, 'k', label='ConvlstmCsiNet_Conv_valloss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    picfile_loss = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/Loss_' + file + '.png'
    plt.savefig(picfile_loss)

    # serialize model to JSON
    model_json = autoencoder.to_json()
    outfile = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/model_' + file + '.json'
    with open(outfile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    outfile = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/model_' + file + '.h5'
    autoencoder.save_weights(outfile)

    # Testing data
    tStart = time.time()
    x_hat = autoencoder.predict(x_test)
    tEnd = time.time()
    print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

    # Calculate and print the NMSE and rho
    X_test = np.reshape(X_test, (len(X_test), T, img_height, 125))
    x_test_real = np.reshape(x_test[:, :, :, :, 0], (len(x_test), T, -1))
    x_test_imag = np.reshape(x_test[:, :, :, :, 1], (len(x_test), T, -1))
    x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
    x_test_F = np.reshape(x_test_C, (len(x_test_C), T, img_height, img_width))

    x_hat_real = np.reshape(x_hat[:, :, :, :, 0], (len(x_hat), T, -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, :, 1], (len(x_hat), T, -1))
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), T, img_height, img_width))
    X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), T, img_height, 257 - img_width))), axis=3),
                       axis=3)
    X_hat = X_hat[:, :, :, 0:125]

    n1 = np.sqrt(abs(np.sum(np.conj(X_test) * X_test, axis=2)))
    n1 = n1.astype('float64')
    n2 = np.sqrt(abs(np.sum(np.conj(X_hat) * X_hat, axis=2)))
    n2 = n2.astype('float64')
    aa = abs(np.sum(np.conj(X_test) * X_hat, axis=-2))
    rho = np.mean(aa / (n1 * n2), axis=(0, 1, 2))

    X_hat = np.reshape(X_hat, (len(X_hat), T, -1))
    X_test = np.reshape(X_test, (len(X_test), T, -1))
    power = np.sum(abs(x_test_C) ** 2, axis=2)
    x_hat_C = np.reshape(x_hat_C, (len(x_hat_C), T, -1))
    mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=2)
    nmse = np.mean(mse / power, (0, 1))
    minus_nmse = 10 * math.log10(nmse)
    print("In " + envir + " environment")
    print("When dimension is", encoded_dim)
    print('When the epoch is', test_peroid * (period + 1))
    print("NMSE is ", minus_nmse)
    print("Correlation is", rho)

    # save rho
    rho_all = np.append(rho_all, rho)
    rho_all = np.reshape(rho_all, (-1,))
    filename = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/rho' + file + '.csv'
    np.savetxt(filename, rho_all, delimiter=",")

    iters = range(0, len(rho_all) * test_peroid, test_peroid)
    plt.figure()
    plt.plot(iters, rho_all, 'g', label='ρ_Conv')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('ρ')
    plt.legend(loc="upper right")
    picfile_rho = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/rho_' + file + '.png'
    plt.savefig(picfile_rho)

    # print nmse
    nmse_all = np.append(nmse_all, minus_nmse)
    nmse_all = np.reshape(nmse_all, (-1,))
    filename = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/nmse_' + file + '.csv'
    np.savetxt(filename, nmse_all, delimiter=",")

    iters = range(0, len(nmse_all) * test_peroid, test_peroid)
    plt.figure()
    plt.plot(iters, nmse_all, 'g', label='10*log10(nmse)_Conv')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('nmse')
    plt.legend(loc="upper right")
    picfile_nmse = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/nmse_' + file + '.png'
    plt.savefig(picfile_nmse)

    # print CSI
    x_test_avrT = np.mean(x_test, axis=-4)
    x_hat_avrT = np.mean(x_hat, axis=-4)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        x_testplo = abs(x_test_avrT[i, :, :, 0] - 0.5 + 1j * (x_test_avrT[i, :, :, 1] - 0.5))
        plt.imshow(np.max(np.max(x_testplo)) - x_testplo.T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        decoded_imgsplo = abs(x_hat_avrT[i, :, :, 0] - 0.5
                              + 1j * (x_hat_avrT[i, :, :, 1] - 0.5))
        plt.imshow(np.max(np.max(decoded_imgsplo)) - decoded_imgsplo.T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()
    picfile = 'CsiNet' + '/Conv_' + envir + '/dim_' + str(encoded_dim) + '/pic_' + file + '.png'
    plt.savefig(picfile)
