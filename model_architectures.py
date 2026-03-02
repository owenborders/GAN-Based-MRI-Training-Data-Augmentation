import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, Activation, GaussianNoise, MaxPooling2D, UpSampling2D, concatenate,Dropout

import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D,Conv3D, BatchNormalization, Activation, GaussianNoise,MaxPooling3D,UpSampling3D,Add, SpatialDropout2D, Conv2DTranspose,Multiply
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.regularizers import l2

from tensorflow.keras.losses import binary_crossentropy
import numpy as np
class Architectures():


    def deep_atrous_attn_unet(self,input_shape=(256, 256, 1), num_classes=1, activation='sigmoid'):
        def attention_block(F_g, F_l, F_int):
            g = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
            g = BatchNormalization()(g)
            x = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
            x = BatchNormalization()(x)
            psi = Add()([g, x])
            psi = Activation('elu')(psi)
            psi = SpatialDropout2D(0.2)(psi)
            psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
            psi = Activation('sigmoid')(psi)

            return Multiply()([F_l, psi])
            
        def convolution_block(x, filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.2)(x)
            shortcut = x
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=1,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.2)(x)
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.2)(x)
            x = Add()([x, shortcut])

            return x

        def atrous_spatial_pyramid_pooling(inputs, filters):
            dims = K.int_shape(inputs)
            pool_size = dims[1:3]

            conv_1x1 = convolution_block(inputs, filters, 1)

            conv_3x3_r6 = convolution_block(inputs, filters, 3, dilation_rate=2)
            conv_3x3_r12 = convolution_block(inputs, filters, 3, dilation_rate=4)
            conv_3x3_r18 = convolution_block(inputs, filters, 3, dilation_rate=6)

            image_pooling = layers.GlobalAveragePooling2D()(inputs)
            image_pooling = layers.Reshape((1, 1, dims[3]))(image_pooling)
            image_pooling = convolution_block(image_pooling, filters, 1)
            image_pooling = layers.UpSampling2D(size=pool_size, interpolation='bilinear')(image_pooling)

            x = layers.Concatenate()([conv_1x1, conv_3x3_r6, conv_3x3_r12, conv_3x3_r18, image_pooling])
            x = convolution_block(x, filters, 1)
            
            return x

        inputs = layers.Input(shape=input_shape)
        
        c1 = convolution_block(inputs, 32,3)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = convolution_block(p1, 64, 3)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = convolution_block(p2, 128, 3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = convolution_block(p3, 256, 3)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        b = atrous_spatial_pyramid_pooling(p4, 256)

        u4 = layers.UpSampling2D((2, 2))(b)
        att4 = attention_block(u4, c4, 256)
        u4 = layers.Concatenate()([u4, att4])
        u4 = convolution_block(u4, 256, 3)
        
        u3 = layers.UpSampling2D((2, 2))(u4)
        att3 = attention_block(u3, c3, 128)
        u3 = layers.Concatenate()([u3, att3])
        u3 = convolution_block(u3, 128, 3)

        u2 = layers.UpSampling2D((2, 2))(u3)
        att2 = attention_block(u2, c2, 64)
        u2 = layers.Concatenate()([u2, att2])
        u2 = convolution_block(u2, 64, 3)

        u1 = layers.UpSampling2D((2, 2))(u2)
        att1 = attention_block(u1, c1, 32)
        u1 = layers.Concatenate()([u1, att1])
        
        u1 = convolution_block(u1, 32, 3)
        u1 = convolution_block(u1, 32, 3)

        outputs = layers.Conv2D(num_classes, (1, 1), activation=activation)(u1)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model



    def atrous_attn_unet(self,input_shape=(256, 256, 1), num_classes=1, activation='sigmoid'):
        def attention_block(F_g, F_l, F_int):
            g = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
            g = BatchNormalization()(g)
            x = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
            x = BatchNormalization()(x)
            psi = Add()([g, x])
            psi = Activation('elu')(psi)
            psi = SpatialDropout2D(0.1)(psi)
            psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
            psi = Activation('sigmoid')(psi)

            return Multiply()([F_l, psi])
            
        def convolution_block(x, filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.1)(x)
            shortcut = x
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=1,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.1)(x)
  
            x = Add()([x, shortcut])

            return x

        def atrous_spatial_pyramid_pooling(inputs, filters):
            dims = K.int_shape(inputs)
            pool_size = dims[1:3]

            conv_1x1 = convolution_block(inputs, filters, 1)

            conv_3x3_r6 = convolution_block(inputs, filters, 3, dilation_rate=2)
            conv_3x3_r12 = convolution_block(inputs, filters, 3, dilation_rate=4)
            conv_3x3_r18 = convolution_block(inputs, filters, 3, dilation_rate=6)

            image_pooling = layers.GlobalAveragePooling2D()(inputs)
            image_pooling = layers.Reshape((1, 1, dims[3]))(image_pooling)
            image_pooling = convolution_block(image_pooling, filters, 1)
            image_pooling = layers.UpSampling2D(size=pool_size, interpolation='bilinear')(image_pooling)

            x = layers.Concatenate()([conv_1x1, conv_3x3_r6, conv_3x3_r12, conv_3x3_r18, image_pooling])
            x = convolution_block(x, filters, 1)
            
            return x

        inputs = layers.Input(shape=input_shape)
        
        c1 = convolution_block(inputs, 32,3)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = convolution_block(p1, 64, 3)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = convolution_block(p2, 128, 3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = convolution_block(p3, 256, 3)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        b = atrous_spatial_pyramid_pooling(p4, 256)

        u4 = layers.UpSampling2D((2, 2))(b)
        att4 = attention_block(u4, c4, 256)
        u4 = layers.Concatenate()([u4, att4])
        u4 = convolution_block(u4, 256, 3)
        
        u3 = layers.UpSampling2D((2, 2))(u4)
        att3 = attention_block(u3, c3, 128)
        u3 = layers.Concatenate()([u3, att3])
        u3 = convolution_block(u3, 128, 3)

        u2 = layers.UpSampling2D((2, 2))(u3)
        att2 = attention_block(u2, c2, 64)
        u2 = layers.Concatenate()([u2, att2])
        u2 = convolution_block(u2, 64, 3)

        u1 = layers.UpSampling2D((2, 2))(u2)
        att1 = attention_block(u1, c1, 32)
        u1 = layers.Concatenate()([u1, att1])
        
        u1 = convolution_block(u1, 32, 3)
        u1 = convolution_block(u1, 32, 3)

        outputs = layers.Conv2D(num_classes, (1, 1), activation=activation)(u1)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    

    def atrous_attn_unet_discriminator(self,input_shape=(256, 256, 1), num_classes=1, activation='sigmoid'):
        def attention_block(F_g, F_l, F_int):
            g = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
            g = BatchNormalization()(g)
            x = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
            x = BatchNormalization()(x)
            psi = Add()([g, x])
            psi = Activation('elu')(psi)
            psi = SpatialDropout2D(0.1)(psi)
            psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
            psi = Activation('sigmoid')(psi)

            return Multiply()([F_l, psi])
            
        def convolution_block(x, filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.1)(x)
            shortcut = x
            x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate,kernel_initializer='he_normal', padding='same', use_bias=use_bias)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('elu')(x)
            x = SpatialDropout2D(0.1)(x)
            x = Add()([x, shortcut])

            return x

        def atrous_spatial_pyramid_pooling(inputs, filters):
            dims = K.int_shape(inputs)
            pool_size = dims[1:3]

            conv_1x1 = convolution_block(inputs, filters, 1)

            conv_3x3_r6 = convolution_block(inputs, filters, 3, dilation_rate=2)
            conv_3x3_r12 = convolution_block(inputs, filters, 3, dilation_rate=4)
            conv_3x3_r18 = convolution_block(inputs, filters, 3, dilation_rate=6)

            image_pooling = layers.GlobalAveragePooling2D()(inputs)
            image_pooling = layers.Reshape((1, 1, dims[3]))(image_pooling)
            image_pooling = convolution_block(image_pooling, filters, 1)
            image_pooling = layers.UpSampling2D(size=pool_size, interpolation='bilinear')(image_pooling)

            x = layers.Concatenate()([conv_1x1, conv_3x3_r6, conv_3x3_r12, conv_3x3_r18, image_pooling])
            x = convolution_block(x, filters, 1)
            
            return x

        inputs = layers.Input(shape=input_shape)
        
        c1 = convolution_block(inputs, 32,3)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = convolution_block(p1, 64, 3)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = convolution_block(p2, 128, 3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = convolution_block(p3, 256, 3)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        b = atrous_spatial_pyramid_pooling(p4, 256)

        u4 = layers.UpSampling2D((2, 2))(b)
        att4 = attention_block(u4, c4, 256)
        u4 = layers.Concatenate()([u4, att4])
        u4 = convolution_block(u4, 256, 3)
        
        u3 = layers.UpSampling2D((2, 2))(u4)
        att3 = attention_block(u3, c3, 128)
        u3 = layers.Concatenate()([u3, att3])
        u3 = convolution_block(u3, 128, 3)

        u2 = layers.UpSampling2D((2, 2))(u3)
        att2 = attention_block(u2, c2, 64)
        u2 = layers.Concatenate()([u2, att2])
        u2 = convolution_block(u2, 64, 3)

        u1 = layers.UpSampling2D((2, 2))(u2)
        att1 = attention_block(u1, c1, 32)
        u1 = layers.Concatenate()([u1, att1])
        
        u1 = convolution_block(u1, 32, 3)
        u1 = convolution_block(u1, 32, 3)
        pooled_output = layers.GlobalAveragePooling2D()(u1)

        pooled_output = layers.Dense(32, activation='relu')(pooled_output)
        outputs = layers.Dense(1, activation='sigmoid')(pooled_output)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model


