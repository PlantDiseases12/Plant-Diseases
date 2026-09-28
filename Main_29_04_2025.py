import numpy as np
import os
import cv2 as cv
from numpy import matlib
import Global_Vars
from AOA import AOA
from AZO import AZO
from COA import COA
from CWO import CWO
from Model_CNN import Model_CNN
from Model_DCNN import Model_DCNN
from Model_ESPDTUPlusPlus import Model_ESPDTUPlusPlus
from Model_Efficent_CNN import Model_Efficient_CNN
from Model_EfficientNet import Model_EfficientNet
from Model_GAN import Model_GAN
from Model_RSNetV2 import Model_RSNetV2
from Model_RSNetV2_DCRF import Model_ResSuffleNetV2_DCRF
from Model_ResSuffleNetV2 import Model_ResSuffleNetV2
from Model_VGG16 import Model_VGG16
from Objective_Function import objfun, objfun_pred
from PlotResults import *
from Proposed import Proposed

# Read Dataset
an = 0
if an == 1:
    Images = []
    Masks = []
    imagefold = os.listdir('./Dataset/cb328232-31f5-4b84-a929-8e1ee551d66a/public/GrowliFlowerL/images/')
    maskfold = os.listdir('./Dataset/cb328232-31f5-4b84-a929-8e1ee551d66a/public/GrowliFlowerL/labels/')
    for n in range(len(imagefold)):
        filename_image = os.listdir('./Dataset/cb328232-31f5-4b84-a929-8e1ee551d66a/public/GrowliFlowerL/images/' + imagefold[n])
        filename_mask = os.listdir('./Dataset/cb328232-31f5-4b84-a929-8e1ee551d66a/public/GrowliFlowerL/labels/' + imagefold[n] + '/' + 'maskPlants')
        for m in range(len(filename_image)):
            image = cv.imread('./Dataset/cb328232-31f5-4b84-a929-8e1ee551d66a/public/GrowliFlowerL/images/' + imagefold[n] + '/' + filename_image[m])
            mask = cv.imread(
                './Dataset/cb328232-31f5-4b84-a929-8e1ee551d66a/public/GrowliFlowerL/labels/' + imagefold[n] + '/' + 'maskPlants' + '/' + filename_mask[m])
            Images.append(cv.resize(image, (512, 512)))
            Masks.append(mask)
    np.save('Images.npy', Images)
    np.save('Mask.npy', Masks)

# generate Ground Truth
an = 0
if an == 1:
    Mask = np.load('Mask.npy', allow_pickle=True)
    GT = []
    for n in range(len(Mask)):
        print(n)
        image = Mask[n]
        img = np.zeros(image.shape, dtype=np.uint8)
        max_val = np.max(image)
        thresh = max_val - (max_val * 0.2)
        index = np.where(image >= thresh)
        img[index[0], index[1]] = 255
        img = img.astype(np.uint8)
        # cv.imshow('im', image)
        # cv.imshow('gt', img)
        # # cv.waitKey(300)
        GT.append(img)
    np.save('Ground_Truth.npy', GT)

# Generate Target using GT
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('Ground_Truth.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        image = Ground_Truth[i]
        result = image.astype('uint8')
        a, b = np.where((result == 255))
        uniq = np.unique(result)
        if len(a) > 50000:
            Tar.append(1)
        else:
            Tar.append(0)
    Tar = np.asarray(Tar).reshape(-1, 1)
    np.save('Target.npy', np.reshape(Tar, (-1, 1)))

# Segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    GT = np.load('GT.npy', allow_pickle=True)
    Results = []
    for i in range(len(Images)):
        Segmented_image = Model_ESPDTUPlusPlus(Images[i], GT[i])
        Results.append(Segmented_image)
    np.save('Method5_Dataset.npy', Results)

# Optimization for Binary Classification
an = 0
if an == 1:
    Data = np.load('Method5_Dataset.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Ch_len = 3
    xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun
    Max_iter = 50

    print("AOA...")
    [bestfit1, fitness1, bestsol1, time1] = AOA(initsol, fname, xmin, xmax, Max_iter)

    print("AZO...")
    [bestfit2, fitness2, bestsol2, time2] = AZO(initsol, fname, xmin, xmax, Max_iter)

    print("CWO...")
    [bestfit3, fitness3, bestsol3, time3] = CWO(initsol, fname, xmin, xmax, Max_iter)

    print("COA...")
    [bestfit4, fitness4, bestsol4, time4] = COA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)
    fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
    bests = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Bestsol_binary.npy', bests)
    np.save('Fitness.npy', fitness)

# Binary Classification
an = 0
if an == 1:
    Feat = np.load('Method5_Dataset.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Bestsol = np.load('Bestsol_binary.npy', allow_pickle=True)
    Activation_Function = ['Relu', 'Linear', 'Sigmoid', 'Tanh', 'Softmax']
    EVAL = []
    for n in range(len(Activation_Function)):
        Eval = np.zeros((10, 25))
        for j in range(Bestsol.shape[0]):
            learnper = round(Feat.shape[0] * 0.75)
            sol = np.round(Bestsol[j, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Target[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Target[learnper:, :]
            Eval[j, :], pred1 = Model_ResSuffleNetV2(Train_Data, Train_Target, Test_Data, Test_Target, Activation_Function[n], sol)
        Eval[5, :], pred2 = Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target, Activation_Function[n])
        Eval[6, :], pred3 = Model_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, Activation_Function[n])
        Eval[7, :], pred4 = Model_Efficient_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Activation_Function[n])
        Eval[8, :], pred5 = Model_ResSuffleNetV2(Train_Data, Train_Target, Test_Data, Test_Target, Activation_Function[n])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_binary.npy', EVAL)


# Severity Classification
an = 0
if an == 1:
    Feat = np.load('Method5_Dataset.npy', allow_pickle=True)
    Target = np.load('Target_Abnormal.npy', allow_pickle=True)
    Batch_Size = [4, 16, 32, 64, 128]
    EVAL = []
    for n in range(len(Batch_Size)):
        Eval = np.zeros((5, 25))
        learnper = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnper, :]
        Train_Target = Target[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Target[learnper:, :]
        Eval[5, :], pred2 = Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_Size[n])
        Eval[6, :], pred3 = Model_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_Size[n])
        Eval[7, :], pred4 = Model_Efficient_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_Size[n])
        Eval[8, :], pred5 = Model_ResSuffleNetV2(Train_Data, Train_Target, Test_Data, Test_Target, Batch_Size[n])
        Eval[9, :] = Model_ResSuffleNetV2_DCRF(Feat, Target, Batch_Size[n])
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)

# Optimization for Rate of Progression Calculation - Prediction
an = 0
if an == 1:
    Data = np.load('Method5_Dataset.npy', allow_pickle=True)
    Target = np.load('Target_Prediction.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Ch_len = 3
    xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_pred
    Max_iter = 50

    print("AOA...")
    [bestfit1, fitness1, bestsol1, time1] = AOA(initsol, fname, xmin, xmax, Max_iter)

    print("AZO...")
    [bestfit2, fitness2, bestsol2, time2] = AZO(initsol, fname, xmin, xmax, Max_iter)

    print("CWO...")
    [bestfit3, fitness3, bestsol3, time3] = CWO(initsol, fname, xmin, xmax, Max_iter)

    print("COA...")
    [bestfit4, fitness4, bestsol4, time4] = COA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)
    fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
    bests = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Bestsol.npy', bests)

# Rate of Progression Calculation - Prediction
an = 0
if an == 1:
    Feat = np.load('Method5_Dataset.npy', allow_pickle=True)
    Target = np.load('Target_Prediction.npy', allow_pickle=True)
    Bestsol = np.load('Bestsol.npy', allow_pickle=True)
    Hidden_Neuron_Count = [100, 200, 300, 400, 500]
    EVAL = []
    for n in range(len(Hidden_Neuron_Count)):
        Eval = np.zeros((10, 14))
        for j in range(Bestsol.shape[0]):
            learnper = round(Feat.shape[0] * 0.75)
            sol = np.round(Bestsol[j, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Target[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Target[learnper:, :]
            Eval[j, :], pred1 = Model_RSNetV2(Train_Data, Train_Target, Test_Data, Test_Target, Hidden_Neuron_Count[n], sol)
        Eval[5, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Hidden_Neuron_Count[n])
        Eval[6, :], pred3 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target, Hidden_Neuron_Count[n])
        Eval[7, :], pred4 = Model_GAN(Train_Data, Train_Target, Test_Data, Test_Target, Hidden_Neuron_Count[n])
        Eval[8, :], pred5 = Model_RSNetV2(Train_Data, Train_Target, Test_Data, Test_Target, Hidden_Neuron_Count[n])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_Prediction.npy', EVAL)


plotConvResults()
plot_Results_binary()
plot_results_Segmentation()
Table_binary()
Table_Abnormal()
plot_Results_abnormal()
Plot_Results_Prediction()
Image_Results()
Plot_ROC_Curve()
Sample_Images()