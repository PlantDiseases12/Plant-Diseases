from keras import Sequential
from keras.applications import EfficientNetB3
from keras.layers import Dense, LSTM
import numpy as np
from Evaluation import evaluation


def Model_EfficientNet(Train_Datas, Train_Targets, Test_Datas, Test_Target, act=None, steps_per_epoch = None):
    if act is None:
        act = 'relu'
    if steps_per_epoch is None:
        steps_per_epoch = 100
    IMG_SIZE = 32
    Train_Data = [img_data for img_data in Train_Datas if img_data is not None]
    Train_Data = np.asarray(Train_Data)
    Train_Target = [label for i, label in enumerate(Train_Targets) if Train_Datas[i] is not None]
    Train_Target = np.asarray(Train_Target)
    Test_Data = [img_data for img_data in Test_Datas if img_data is not None]
    Test_Data = np.asarray(Test_Data)

    Train_x = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_x[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False,
        pooling='max'
    )

    model = Sequential()
    model.add(efficient_net)  # add EfficientNetB7 base model
    # Add an LSTM layer with return_sequences=True for sequence data
    model.add(LSTM(16, input_shape=(Train_x.shape[1:])))
    model.add(Dense(units=Train_Target.shape[1], activation=act))
    model.add(Dense(units=Train_Target.shape[1], activation=act))
    model.add(Dense(units=Test_Target.shape[1], activation=act))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_x, Train_Target, steps_per_epoch=steps_per_epoch, epochs=50, validation_data=(Test_X, Test_Target))

    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Test_Target)
    return Eval, pred
