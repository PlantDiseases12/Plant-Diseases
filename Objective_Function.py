import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_RSNetV2 import Model_RSNetV2
from Model_ResSuffleNetV2 import Model_ResSuffleNetV2


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_ResSuffleNetV2(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(Test_Target, pred)
            Fitn[i] = (1 / Eval[13]) + Eval[11]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_ResSuffleNetV2(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(Test_Target, pred)
        Fitn = (1 / Eval[13]) + Eval[11]
        return Fitn


def objfun_pred(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_RSNetV2(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(Test_Target, pred)
            Fitn[i] = Eval[5] + Eval[4]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_RSNetV2(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(Test_Target, pred)
        Fitn =  Eval[5] + Eval[4]
        return Fitn
