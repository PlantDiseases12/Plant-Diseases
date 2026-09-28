from Model_DCNN import Model_DCNN
from Model_EfficientNet import Model_EfficientNet


def Model_Efficient_CNN(Train_Datas, Train_Targets, Test_Datas, Test_Target, act):
    Eval, Pred_eff = Model_EfficientNet(Train_Datas, Train_Targets, Test_Datas, Test_Target, act)
    Eval, Pred_cnn = Model_DCNN(Train_Datas, Train_Targets, Test_Datas, Test_Target, act)
    Pred = (Pred_cnn+Pred_eff)/2
    return Eval, Pred