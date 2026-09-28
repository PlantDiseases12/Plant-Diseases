import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np

from Evaluation import evaluation


# ----------------------------
# Spatio-Temporal Attention Layer
# ----------------------------
def spatio_temporal_attention(x):
    # Spatial attention
    spatial_avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    spatial_max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial_concat = tf.concat([spatial_avg_pool, spatial_max_pool], axis=-1)
    spatial_attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_concat)
    x = x * spatial_attention

    # Temporal attention (using channel-wise global pooling)
    temporal_avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    temporal_max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    temporal_concat = tf.concat([temporal_avg_pool, temporal_max_pool], axis=-1)
    temporal_attention = layers.Conv2D(x.shape[-1], kernel_size=1, activation='sigmoid')(temporal_concat)
    x = x * temporal_attention
    return x

# ----------------------------
# Generator (with STA)
# ----------------------------
def build_generator(input_shape=(64, 64, 3), num_classes=10):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(0.2)(x)

    # Apply STA
    x = spatio_temporal_attention(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inp, out, name="Generator")

# ----------------------------
# Discriminator
# ----------------------------
def build_discriminator(Activation_Function, input_shape=(64, 64, 3)):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1, activation=Activation_Function)(x)
    return Model(inp, out, name="Discriminator")


def Model_RSNetV2(Train_Datas, Train_Targets, Test_Datas, Test_Target, Activation_Function, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    # ----------------------------
    # Compile STA-CGAN
    # ----------------------------
    img_shape = Train_Datas.shape[1:]
    num_classes = Train_Targets.shape[1]

    generator = build_generator(img_shape, num_classes)
    discriminator = build_discriminator(Activation_Function, img_shape)

    # GAN training setup
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    gan_input = Input(shape=img_shape)
    gan_output = discriminator(generator(gan_input))
    gan_model = Model(gan_input, gan_output)
    gan_model.compile(optimizer='adam', loss='binary_crossentropy')
    gan_model.fit(Train_Datas, Train_Targets, epochs=20, batch_size=32, validation_split=0.2)
    predict = gan_model.predict(Test_Datas)
    Eval = evaluation(Test_Target, predict)
    return Eval, predict
