import tensorflow as tf
from tensorflow.keras import layers, models

from Evaluation import net_evaluation


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x


def dilated_transformer_block(x, num_heads=4, dilation_rate=2):
    _, h, w, c = x.shape
    x_proj = layers.Conv2D(c, 1)(x)
    x_reshape = tf.reshape(x_proj, (-1, h * w, c))

    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=c // num_heads)
    attn_output = attention(x_reshape, x_reshape)

    out = tf.reshape(attn_output, (-1, h, w, c))
    out = layers.Conv2D(c, 3, padding='same', dilation_rate=dilation_rate)(out)
    return out


def spatial_pyramid_pooling(x, pool_sizes=[1, 2, 3, 6]):
    concat = [x]
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    for size in pool_sizes:
        pooled = layers.AveragePooling2D(pool_size=(size, size), strides=(size, size), padding='same')(x)
        upsampled = layers.UpSampling2D(size=(h // size, w // size), interpolation='bilinear')(pooled)
        concat.append(upsampled)
    return layers.Concatenate()(concat)


def build_model(input_shape=(256, 256, 3), num_classes=1, base_filters=64):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D()(e1)

    e2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D()(e2)

    e3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D()(e3)

    e4 = conv_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D()(e4)

    # Bottleneck
    b = dilated_transformer_block(p4, num_heads=4)
    b = spatial_pyramid_pooling(b)

    # Decoder
    up3 = layers.UpSampling2D()(b)
    d3 = conv_block(layers.Concatenate()([up3, e4]), base_filters * 8)

    up2 = layers.UpSampling2D()(d3)
    d2 = conv_block(layers.Concatenate()([up2, e3]), base_filters * 4)

    up1 = layers.UpSampling2D()(d2)
    d1 = conv_block(layers.Concatenate()([up1, e2]), base_filters * 2)

    up0 = layers.UpSampling2D()(d1)
    d0 = conv_block(layers.Concatenate()([up0, e1]), base_filters)

    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(d0)

    return models.Model(inputs, outputs)


def Model_ESPDTUPlusPlus(images, masks):
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    pred_img = model.predict(images)
    Eval = net_evaluation(pred_img, masks)
    return Eval, pred_img

