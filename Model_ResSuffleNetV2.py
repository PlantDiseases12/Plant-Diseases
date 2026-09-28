import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from Evaluation import evaluation


# -----------------------
# Utilities
# -----------------------
def channel_shuffle(x, groups):
    # x shape: (B, H, W, C)
    batch_size, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    group_channels = c // groups
    # reshape -> (B, H, W, groups, group_channels)
    x = tf.reshape(x, [-1, h, w, groups, group_channels])
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])  # swap group and group_channels
    x = tf.reshape(x, [-1, h, w, c])
    return x


def conv_bn_relu(x, out_channels, kernel=1, stride=1, groups=1, name=None):
    x = layers.Conv2D(out_channels, kernel, strides=stride, padding='same',
                      use_bias=False, groups=groups, name=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# -----------------------
# ShuffleNetV2 Block with residual
# -----------------------
def shufflenetv2_unit(x, out_channels, stride=1, name=None):
    """
    ShuffleNetV2 unit.
    If stride == 1: split input channels in two, process right part and concat + shuffle.
    If stride == 2: apply branch on both sides and concat.
    Residual connection applied when stride==1 and in_channels == out_channels.
    """
    in_channels = x.shape[-1]
    if stride == 1:
        # channel split
        assert in_channels % 2 == 0, "Input channels must be even for split when stride=1"
        c = in_channels // 2
        x1 = layers.Lambda(lambda z: z[..., :c])(x)  # left (identity)
        x2 = layers.Lambda(lambda z: z[..., c:])(x)  # right (to transform)

        # right branch
        # 1x1 pw conv (reduce)
        x2 = layers.Conv2D(c, 1, padding='same', use_bias=False)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.ReLU()(x2)
        # 3x3 depthwise conv
        x2 = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=False)(x2)
        x2 = layers.BatchNormalization()(x2)
        # 1x1 pw conv (expand)
        x2 = layers.Conv2D(c, 1, padding='same', use_bias=False)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.ReLU()(x2)

        out = layers.Concatenate(axis=-1)([x1, x2])
    else:
        # stride == 2: both branches transform (no split)
        # left branch (depthwise then pw)
        left = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False)(x)
        left = layers.BatchNormalization()(left)
        left = layers.Conv2D(out_channels // 2, 1, padding='same', use_bias=False)(left)
        left = layers.BatchNormalization()(left)
        left = layers.ReLU()(left)

        # right branch
        right = layers.Conv2D(out_channels // 2, 1, padding='same', use_bias=False)(x)
        right = layers.BatchNormalization()(right)
        right = layers.ReLU()(right)
        right = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False)(right)
        right = layers.BatchNormalization()(right)
        right = layers.Conv2D(out_channels // 2, 1, padding='same', use_bias=False)(right)
        right = layers.BatchNormalization()(right)
        right = layers.ReLU()(right)

        out = layers.Concatenate(axis=-1)([left, right])

    # Channel shuffle
    out = layers.Lambda(lambda z: channel_shuffle(z, groups=2))(out)

    # Residual addition if shapes match (stride==1 and channels equal)
    if stride == 1 and in_channels == out_channels:
        out = layers.Add()([out, x])  # residual
    return out


# -----------------------
# Stage builder
# -----------------------
def make_stage(x, repeats, out_channels, first_stride):
    # first block may have stride=2
    x = shufflenetv2_unit(x, out_channels, stride=first_stride)
    for _ in range(repeats - 1):
        x = shufflenetv2_unit(x, out_channels, stride=1)
    return x


# -----------------------
# Network builder
# -----------------------
def build_residual_shufflenetv2(sol, input_shape=(224, 224, 3), num_classes=1000, scale=1.0):

    Activation_Function = ['Relu', 'Linear', 'Sigmoid', 'Tanh', 'Softmax']
    """
    scale: channel multiplier, typical options: 0.5, 1.0, 1.5, 2.0
    """
    # base channels per stage for 1.0x (from original ShuffleNetV2 paper)
    if scale == 0.5:
        stage_out_channels = [-1, 24, 48, 96, 192, 1024]
    elif scale == 1.0:
        stage_out_channels = [-1, 24, 116, 232, 464, 1024]
    elif scale == 1.5:
        stage_out_channels = [-1, 24, 176, 352, 704, 1024]
    elif scale == 2.0:
        stage_out_channels = [-1, 24, 244, 488, 976, 2048]
    else:
        raise ValueError("Unsupported scale value")

    inp = Input(shape=input_shape)
    x = inp

    # Initial conv + maxpool
    x = layers.Conv2D(stage_out_channels[1], kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Stages configuration: repeats per stage (from ShuffleNetV2)
    repeats = [4, 8, 4]  # you can change depending on model size (for smaller, reduce)
    # Stage 2
    x = make_stage(x, repeats[0], stage_out_channels[2], first_stride=2)
    # Stage 3
    x = make_stage(x, repeats[1], stage_out_channels[3], first_stride=2)
    # Stage 4
    x = make_stage(x, repeats[2], stage_out_channels[4], first_stride=2)

    # Final conv
    x = layers.Conv2D(stage_out_channels[5], kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global pooling + classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation=Activation_Function[int(sol[2])])(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation=Activation_Function[int(sol[2])])(x)

    model = Model(inputs=inp, outputs=out, name=f"Residual_ShuffleNetV2_{scale}x")
    return model


def Model_ResSuffleNetV2(Train_Datas, Train_Targets, Test_Datas, Test_Target, Activation_Function, sol=None):
    if sol is None:
        sol =[5, 0.01, 1]
    model = build_residual_shufflenetv2(sol, input_shape=(128, 128, 3), num_classes=10, scale=1.0)
    model.summary()

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(sol[1]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # quick test train step
    model.fit(Train_Datas, Train_Targets, epochs=10, batch_size=sol[0])

    pred = model.predict(Test_Datas)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Test_Target)
    return Eval, pred
