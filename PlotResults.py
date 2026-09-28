import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.metrics import roc_curve
from itertools import cycle
import cv2 as cv
from sklearn import metrics

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    for n in range(1):
        Fitness = np.load('Fitness.npy', allow_pickle=True)[n]
        Algorithm = ['TERMS', 'AOA-ARSNetv2', 'AZO-ARSNetv2', 'CWO-ARSNetv2', 'COA-ARSNetv2', 'RNACO-ARSNetv2']
        Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset-', str(n + 1), 'Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        fig.canvas.manager.set_window_title('Convergence Curve')
        length = np.arange(Fitness.shape[1])
        plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='AOA-ARSNetv2')
        plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='AZO-ARSNetv2')
        plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='CWO-ARSNetv2')
        plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='COA-ARSNetv2')
        plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='RNACO-ARSNetv2')
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight':'bold', 'size':12})
        plt.savefig("./Results/Dataset-%s-Conv.png" % (n + 1),
    dpi=300,
    bbox_inches='tight')
        plt.show()


def plot_Results_binary():
    eval = np.load('Evaluate_all_Abnormal.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Algorithm = ['AOA-ARSNetv2', 'AZO-ARSNetv2', 'CWO-ARSNetv2', 'COA-ARSNetv2', 'RNACO-ARSNetv2']
    Method = ['EfficientNetB3', 'DCNN', 'Efficient CNN', 'RSNetv2', 'RNACO-ARSNetv2']
    Graph_Terms = [0, 5, 7, 8, 9, 10, 12]
    colors = ['y', 'c', 'g', 'orange', 'k']
    colors1 = ['#f6aa1c', '#4f772d', '#a44a3f', 'c', 'k']
    Epochs = [50, 100, 150, 200, 250]
    for i in range(eval.shape[0]):
        Graph = np.zeros((eval.shape[1], eval.shape[2]))
        for j in range(len(Graph_Terms)):
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
            fig = plt.figure(facecolor='#ffffff', figsize=(10, 6))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_facecolor("#ffffff")
            X = np.arange(Graph.shape[0])
            barWidth = 0.15
            color = ['#ef476f', '#ffd166', '#073b4c', '#669bbc', '#5f0f40']
            for l in range(Graph.shape[1] - 5):
                ax.bar(X + (l * barWidth), Graph[:, l], color=color[l - 5], width=barWidth, label=Algorithm[l])
            plt.xticks(X + (((len(Algorithm)) * barWidth) * 0.4), ('100', '200', '300', '400', '500'),
                       fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('No of Epochs', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=c, markersize=8) for c
                           in color]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.13), fontsize=11,
                       frameon=False, ncol=3,prop={'weight': 'bold', 'size': 12})
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            ax.grid(which='major', axis='y', linestyle='-')
            path1 = "./Results/Binary - Alg-%s.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path1,
    dpi=300,
    bbox_inches='tight')
            plt.show()

            # -----------------------------------------------Classifier------------------------------------------------

            fig = plt.figure(facecolor='#ffffff', figsize=(10, 6))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_facecolor("#ffffff")
            X = np.arange(Graph.shape[0])
            barWidth = 0.15
            color = ['#e9ecef', '#ced4da', '#6c757d', '#495057', '#212529']
            for m in range(5, Graph.shape[1]):
                ax.bar(X + (m * barWidth), Graph[:, m], color=color[m - 5], width=barWidth, label=Method[m - 5], edgecolor='k')
            plt.xticks(X + (((len(Method)) * barWidth) * 1.4), ('100', '200', '300', '400', '500'),
                       fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('No of Epochs',
                       fontname="Arial", fontsize=12, fontweight='bold', color='k')
            # ax.tick_params(axis='x', labelrotation=45)
            plt.ylabel(Terms[Graph_Terms[j]],
                       fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            # plt.ylim([60, 100])
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=True, shadow=True, facecolor='#ecf8f8')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=c, markersize=8) for c
                           in color]
            plt.legend(dot_markers, Method, loc='upper center', bbox_to_anchor=(0.5, 1.13), fontsize=12,
                       frameon=False, ncol=3, prop={'weight':'bold', 'size':12})
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            # ax.grid(which='major', axis='y', linestyle='-')
            path1 = "./Results/Binary - bar_%s.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path1,
    dpi=300,
    bbox_inches='tight')
            plt.show()


def Table_binary():
    eval = np.load('Eval_all_binary.npy', allow_pickle=True)
    Algorithm = ['TERMS/Batch Size', 'AOA-ARSNetv2', 'AZO-ARSNetv2', 'CWO-ARSNetv2', 'COA-ARSNetv2', 'RNACO-ARSNetv2']
    Method = ['TERMS/Batch Size', 'EfficientNetB3', 'DCNN', 'Efficient CNN', 'RSNetv2', 'RNACO-ARSNetv2']

    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Table_Terms = [0, 3, 9, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    Batch_size = [4, 16, 32, 64, 128]
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batch_size)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison for Binary Classification',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Method[0], Batch_size)
            for j in range(len(Method) - 1):
                Table.add_column(Method[j + 1], value[:, len(Algorithm) + j - 1, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison for Binary Classification',
                  '---------------------------------------')
            print(Table)


def plot_Results_abnormal():
    eval = np.load('Evaluate_all_Abnormal.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Method = ['SVM', 'VGG-16', 'ARSNetv2', 'DCRF', 'RNACO-ARSNetv2-DCRF']
    Graph_Terms = [0, 5, 8, 10, 12]
    colors = ['y', 'c', 'g', 'orange', 'k']
    colors1 = ['#f6aa1c', '#4f772d', '#a44a3f', 'c', 'k']
    for i in range(eval.shape[0]):
        Graph = np.zeros((eval.shape[1], eval.shape[2]))
        for j in range(len(Graph_Terms)):
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
            fig = plt.figure(figsize=(8, 7), facecolor='#dcf8e3')
            ax = fig.add_axes([0.12, 0.15, 0.8, 0.8])
            ax.set_facecolor("#dcf8e3")

            X = np.arange(len(Method))
            barWidth = 0.35
            coloru = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c']
            bars = ax.bar(X, Graph[:, 4], color=coloru, width=barWidth, label='Proposed Method')

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.02),
                         f"{np.round(height, 2)}", ha='center', va='bottom', fontsize=8.5, weight='bold')

            ax.spines['left'].set_position(('data', X[0] - barWidth / 2 - 0.22))
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0, max(Graph[:, 4]) * 1.3)

            # Adding arrow at the end of X-axis
            ax.annotate('', xy=(1.03, 0), xytext=(1, 0),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.9))

            # Adding arrow at the end of Y-axis
            ax.annotate('', xy=(0, 1.03), xytext=(0, 1),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.9))
            plt.xticks(X + 0.05, ('16', '32', '64', '128', '256'), fontname="Arial", fontsize=12,
                       fontweight='bold', color='k')
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylim(0)
            path = "./Results/Abnormal - bar-%s.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path,
    dpi=300,
    bbox_inches='tight')
            plt.show()


def Table_Abnormal():
    eval = np.load('Eval_all_Abnormal.npy', allow_pickle=True)
    Method = ['TERMS/Activation Function', 'SVM', 'VGG-16', 'ARSNetv2', 'DCRF', 'RNACO-ARSNetv2-DCRF']

    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Table_Terms = [0, 3, 9, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    Activation_Function = ['Relu', 'Linear', 'Sigmoid', 'Tanh', 'Softmax']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Method[0], Activation_Function)
            for j in range(len(Method) - 1):
                Table.add_column(Method[j + 1], value[:, j, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Method Comparison for Multi Classification',
                  '---------------------------------------')
            print(Table)


def Plot_ROC_Curve():
    lw = 2
    cls = ['SVM', 'VGG-16', 'ARSNetv2', 'DCRF', 'RNACO-ARSNetv2-DCRF']
    for a in range(1):  # For 5 Datasets
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        Actual = np.load('Target.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
        path1 = "./Results/ROC-%s.png" % (a + 1)
        plt.savefig(path1,
    dpi=300,
    bbox_inches='tight')
        plt.show()


def plot_results_Segmentation():  # For Segmentation
    Eval_all = np.load('Eval_all_Segmentation.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :], color='c', width=0.10, label="Unet")
            ax.bar(X + 0.10, stats[i, 1, :], color='y', width=0.10, label="Unet3+")
            ax.bar(X + 0.20, stats[i, 2, :], color='b', width=0.10, label="FCN")
            ax.bar(X + 0.30, stats[i, 3, :], color=[0.5, 0.2, 0.4], width=0.10, label="DenseUnet")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="ESPDTU++")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'), fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
            #            ncol=2, fancybox=True, shadow=True)
            plt.legend(loc=10, prop={'weight': 'bold', 'size': 12})
            path1 = "./Results/Dataset_%s_%s_alg.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1,
    dpi=300,
    bbox_inches='tight')
            plt.show()


def Plot_Results_Prediction():
    Eval_all = np.load('Eval_all_Prediction.npy', allow_pickle=True)
    Terms = ['MSE', 'MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'ONE-NORM', 'TWO-NORM', 'INFINITY-NORM']
    Algorithm = ['GTO', 'BCO', 'FOA', 'ABOA', 'Proposed']
    Classifier = ['GAN', 'REF2', 'REF4', 'STA-CGAN', 'PROPOSED']
    for u in range(len(Eval_all)):

        for j in range(len(Terms)):
            val = np.zeros((5, 5))
            for k in range(len(Algorithm)):
                val[k, :] = Eval_all[u, :, k, j]

            x = [5, 10, 15, 20, 25]

            data = val
            plt.plot(x, data[0, :], color='black', linewidth=4, marker='o', markerfacecolor='c', markersize=13,
                     label="AOA-ASTA-CGAN")
            plt.plot(x, data[1, :], color='black', linewidth=4, marker='o', markerfacecolor='orange', markersize=13,
                     label="AZO-ASTA-CGAN")
            plt.plot(x, data[2, :], color='black', linewidth=4, marker='o', markerfacecolor='y', markersize=13,
                     label="CWO-ASTA-CGAN")
            plt.plot(x, data[3, :], color='black', linewidth=4, marker='o', markerfacecolor='#495057', markersize=13,
                     label="COA-ASTA-CGAN")
            plt.plot(x, data[4, :], color='black', linewidth=4, marker='o', markerfacecolor='k', markersize=13,
                     label="RNACO-ASTA-CGAN")
            plt.ylabel(Terms[j], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(x, ('5', '10', '15', '20', '25'), fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('No of Days', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),
                       ncol=3, fancybox=True, shadow=True, prop={'weight':'bold', 'size': 12}, frameon=False)
            path1 = "./Results/Dataset_%s-perfalg_%s.png" % (u, Terms[j])
            plt.savefig(path1,
    dpi=300,
    bbox_inches='tight')
            plt.show()

        for j in range(len(Terms)):
            val = np.zeros((6, 5))
            for k in range(len(Classifier)):
                val[k, :] = Eval_all[u, :, k + 5, j]

            n_groups = 5
            data = val
            plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.10
            opacity = 1
            plt.bar(index, data[0, :], bar_width,
                    alpha=opacity,
                    color='orange',
                    label='CNN')
            plt.bar(index + bar_width, data[1, :], bar_width,
                    alpha=opacity,
                    color='c',
                    label='Efficient CNN')
            plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                    alpha=opacity,
                    color='y',
                    label='CGAN')
            plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                    alpha=opacity,
                    color='#495057',
                    label='STA-CGAN')
            plt.bar(index + 4 * bar_width, data[4, :], bar_width,
                    alpha=opacity,
                    color='k',
                    label='RNACO-ASTA-CGAN')
            plt.xticks(index + 0.25, ('5', '10', '15', '20', '25'), fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[j], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('No of Days', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),
                       ncol=3, fancybox=True, shadow=True, prop={'weight': 'bold', 'size': 12}, frameon=False)
            plt.tight_layout()
            path1 = "./Results/Dataset_%s-perfcls_%s.png" % (u, Terms[j])
            plt.savefig(path1,
    dpi=300,
    bbox_inches='tight')
            plt.show()


def Image_Results():
    I = [2189, 2191, 2192, 2195, 2197]
    Images = np.load('Images.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    UNet = np.load('Method1_Dataset.npy', allow_pickle=True)
    ResUNetPlusPlus = np.load('Method2_Dataset.npy', allow_pickle=True)
    Trans_ResUNet = np.load('Method3_Dataset.npy', allow_pickle=True)
    TDR2UNetPlusPlus = np.load('Method4_Dataset.npy', allow_pickle=True)
    for i in range(len(I)):
        plt.subplot(2, 3, 1)
        plt.title('Original')
        plt.imshow(Images[I[i]])
        plt.subplot(2, 3, 2)
        plt.title('GT')
        plt.imshow(GT[I[i]])
        plt.subplot(2, 3, 3)
        plt.title('UNet')
        plt.imshow(UNet[I[i]])
        plt.subplot(2, 3, 4)
        plt.title('ResUNetPlusPlus')
        plt.imshow(ResUNetPlusPlus[I[i]])
        plt.subplot(2, 3, 5)
        plt.title('Trans_ResUNet')
        plt.imshow(Trans_ResUNet[I[i]])
        plt.subplot(2, 3, 6)
        plt.title('TDR2UNetPlusPlus')
        plt.imshow(TDR2UNetPlusPlus[I[i]])
        plt.tight_layout()
        plt.show()
        # cv.imwrite('./Results/Image_Results/Dataset-' + 'orig-' + str(i + 1) + '.png', Images[I[i]])
        # cv.imwrite('./Results/Image_Results/Dataset-' + 'gt-' + str(i + 1) + '.png', GT[I[i]])
        # cv.imwrite('./Results/Image_Results/Dataset-' + 'unet3+-' + str(i + 1) + '.png', UNet[I[i]])
        # cv.imwrite('./Results/Image_Results/Dataset-' + 'Resunet++-' + str(i + 1) + '.png',
        #            ResUNetPlusPlus[I[i]])
        # cv.imwrite('./Results/Image_Results/Dataset-' + 'Trans-ResUnet-' + str(i + 1) + '.png',
        #            Trans_ResUNet[I[i]])
        # cv.imwrite('./Results/Image_Results/Dataset-' + 'Proposed-' + str(i + 1) + '.png',
        #            TDR2UNetPlusPlus[I[i]])


def Sample_Images():
    Orig = np.load('Images.npy', allow_pickle=True)
    ind = [318, 319, 320, 321, 322, 323]
    fig, ax = plt.subplots(2, 3)
    plt.suptitle("Sample Images from Dataset " + str(0 + 1))
    plt.subplot(2, 3, 1)
    plt.title('Image-1')
    plt.imshow(Orig[ind[0]])
    plt.subplot(2, 3, 2)
    plt.title('Image-2')
    plt.imshow(Orig[ind[1]])
    plt.subplot(2, 3, 3)
    plt.title('Image-3')
    plt.imshow(Orig[ind[2]])
    plt.subplot(2, 3, 4)
    plt.title('Image-4')
    plt.imshow(Orig[ind[3]])
    plt.subplot(2, 3, 5)
    plt.title('Image-5')
    plt.imshow(Orig[ind[4]])
    # plt.show()
    plt.subplot(2, 3, 6)
    plt.title('Image-6')
    plt.imshow(Orig[ind[5]])
    plt.show()
    # cv.imwrite('./Results/Sample_Images/Abnormal/Dataset' + str(n + 1) + '-img-' + str(0 + 1) + '.png', Orig[ind[0]])
    # cv.imwrite('./Results/Sample_Images/Abnormal/Dataset' + str(n + 1) + '-img-' + str(1 + 1) + '.png', Orig[ind[1]])
    # cv.imwrite('./Results/Sample_Images/Abnormal/Dataset' + str(n + 1) + '-img-' + str(2 + 1) + '.png', Orig[ind[2]])
    # cv.imwrite('./Results/Sample_Images/Abnormal/Dataset' + str(n + 1) + '-img-' + str(3 + 1) + '.png', Orig[ind[3]])
    # cv.imwrite('./Results/Sample_Images/Abnormal/Dataset' + str(n + 1) + '-img-' + str(4 + 1) + '.png', Orig[ind[4]])
    # cv.imwrite('./Results/Sample_Images/Abnormal/Dataset' + str(n + 1) + '-img-' + str(5 + 1) + '.png', Orig[ind[5]])

def Confusion_Binary():
    # for n in range(No_of_Dataset):
    Actual = np.load('Binary_Actual.npy', allow_pickle=True).astype(np.int32)
    Predict = np.load('Binary_Predicted.npy', allow_pickle=True).astype(np.int32)
    classes = ['Normal', 'Abnormal']
    class_2 = ['Adenocarcinoma', 'Small Cell \n Carcinoma', 'Large Cell \n Carcinoma', 'Squamous']
    class_3 = ['Bengin', 'Malignant', 'Normal']
    # classes = [class_1, class_2, class_3]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.canvas.manager.set_window_title('Confusion Matrix')
    confusion_matrix = metrics.confusion_matrix(Actual, Predict)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    cm_display.plot(ax=ax)
    for labels in cm_display.text_.ravel():
        labels.set_fontsize(16)  # Set desired font size
    path = "./Results/Confusion_Binary.png"
    plt.title("Confusion Matrix")
    plt.ylabel('Actual', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xlabel('Predicted', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=11, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=11, fontweight='bold', color='k')
    plt.savefig(path)
    plt.show()

def Confusion_Severity():
    # for n in range(No_of_Dataset):
    Actual = np.load('Severity_Actual.npy', allow_pickle=True).astype(np.int32)
    Predict = np.load('Severity_Predicted.npy', allow_pickle=True).astype(np.int32)
    classes = ['Earlier', 'Mild', 'Moderate', 'Severe']
    class_2 = ['Adenocarcinoma', 'Small Cell \n Carcinoma', 'Large Cell \n Carcinoma', 'Squamous']
    class_3 = ['Bengin', 'Malignant', 'Normal']
    # classes = [class_1, class_2, class_3]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.canvas.manager.set_window_title('Confusion Matrix')
    confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    cm_display.plot(ax=ax)
    for labels in cm_display.text_.ravel():
        labels.set_fontsize(16)  # Set desired font size
    path = "./Results/Confusion_Severity.png"
    plt.title("Confusion Matrix")
    plt.ylabel('Actual', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xlabel('Predicted', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=11, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=11, fontweight='bold', color='k')
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    # # Confusion_Binary()
    # # Confusion_Severity()
    # plotConvResults()
    # plot_Results_binary()
    # plot_results_Segmentation()
    # Table_binary()
    # Table_Abnormal()
    # plot_Results_abnormal()
    Plot_Results_Prediction()
    # Image_Results()
    Plot_ROC_Curve()
    # Sample_Images()
