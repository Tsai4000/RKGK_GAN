
import numpy as np
import cv2
from IPython.core.debugger import Tracer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose, ZeroPadding2D, Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop

import keras.backend as K

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY

class Gan:
    def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        # self.optimizer = SGD(lr=0.0002, decay=8e-8)# , beta_1=0.5
        self.optimizer = RMSprop(lr=0.00005)

        self.G = self.__generator()
        self.G.compile(loss=self.wasserstein_loss, optimizer=self.optimizer, metrics=['accuracy'])

        self.D = self.__discriminator()
        self.D.compile(loss=self.wasserstein_loss,
                       optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(
            loss=self.wasserstein_loss, optimizer=self.optimizer, metrics=['accuracy'])
        #WGAN
        self.clip_value = 0.001

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def __generator(self):
        """ Declare generator """
        model = Sequential()

        model.add(Dense(512 * 7 * 7, activation="relu",     
                    input_dim=100))
        model.add(Reshape((7, 7, 512)))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same", use_bias=False))
        model.add(Activation("relu"))

        # model.add(Dense(self.width * self.height * self.channels))
        # model.add(Reshape((self.width, self.height, self.channels)))

        model.summary()




        # model = Sequential()
        # model.add(Dense(256, input_shape=(100,)))
        # model.add(LeakyReLU(alpha=0.2))  # 使用 LeakyReLU 激活函數
        # model.add(BatchNormalization(momentum=0.8))  # 使用 BatchNormalization 優化
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(self.width * self.height *
        #                 self.channels, activation='tanh'))
        # model.add(Reshape((self.width, self.height, self.channels)))
        # model.summary()

        return model

    def __discriminator(self):
        """ Declare discriminator """
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2,     
                     input_shape=self.shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.summary()









        # model = Sequential()
        # model.add(Flatten(input_shape=self.shape))
        # model.add(Dense((self.width * self.height * self.channels),
        #                 input_shape=self.shape))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(int((self.width * self.height * self.channels)/2)))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=10000, batch=32, save_interval=100):
        # print(X_train)
        d_loss = 0
        g_loss = 0
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        print(X_train.shape)
        # # Adversarial ground truths
        # valid = -np.ones((batch_size, 1))
        # fake = np.ones((batch_size, 1))
        for cnt in range(epochs):
            for _ in range(5):    
                # train discriminator
                # random_index = np.random.randint(0, len(X_train) - batch/2)
                # legit_images = X_train[random_index: random_index + int(batch/2)].reshape(
                #     int(batch/2), self.width, self.height, self.channels)

                idx = np.random.randint(0, X_train.shape[0], batch)
                legit_images = X_train[idx]

                gen_noise = np.random.normal(0, 1, (batch, 100))
                syntetic_images = self.G.predict(gen_noise)

                # x_combined_batch = np.concatenate((legit_images, syntetic_images))
                # y_combined_batch = np.concatenate(
                #     (-np.ones((int(batch/2), 1)), np.ones((int(batch/2), 1))))

                d_loss_real = self.D.train_on_batch(legit_images, -np.ones((batch, 1)))
                d_loss_fake = self.D.train_on_batch(syntetic_images, np.ones((batch, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, 
                            self.clip_value) for w in weights]
                    l.set_weights(weights)
            # train generator

            noise = np.random.normal(0, 1, (batch, 100))  # 添加高斯噪聲
            y_mislabled = -np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(
                noise, y_mislabled)

            print('epoch: %d, [Discriminator :: d_loss: %f r: %f f: %f], [ Generator :: loss: %f]' % (
                cnt, 1-d_loss[0], d_loss_real[0], d_loss_fake[0], 1-g_loss[0]))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)
            if cnt % 1000 == 0:
                noise = np.random.normal(0, 1, (1, 100))
                image = self.G.predict(noise)
                cv2.imwrite('./output/%s.png' % cnt, image[0])
                
        self.G.save('./Gmodel/G%d.h5' % g_loss)
        self.D.save('./Dmodel/D%d.h5' % d_loss)


    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)
        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


# a = Gan()
# a.train(mnist.load_data(path="mnist.npz")[0][0])
