import numpy as np
from matplotlib import pyplot as plt


def Inter():
    y_pos1 = np.arange(5)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    A = [0.0568, 0.065, 0.05, 0.045, 0.03]
    ax.bar(y_pos1, A)
    ax.set_xticks(y_pos1, labels=('CNN', 'Efficient CNN', 'CGAN', 'STA-CGAN', 'RNACO-\nASTA-CGAN'), fontname="Arial", fontsize=14,
                  fontweight='bold', color='k')
    # ax.invert_yaxis()  # labels read top-to-bottom
    plt.xticks(rotation=0)
    plt.ylabel('RMSE', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 98)
    path1 = "./Results/interpretability.png"
    plt.savefig(path1)
    plt.show()

    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [92, 95, 91, 94, 95.88]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('RNN', 'CAE', '1DCNN', 'ConvLSTM', 'SCCEN'), fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_2.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [93.05, 92, 93.84, 94.268, 95.81]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'), fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_3.png"
    # plt.savefig(path1)
    # plt.show()
    # #
    # # y_pos1 = np.arange(5)
    # # fig = plt.figure()
    # # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # # A = [88.05, 90, 92.498, 93.758, 94.075]
    # # ax.bar(y_pos1, A)
    # # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    # #               fontname="Arial", fontsize=14,
    # #               fontweight='bold', color='k')
    # # # ax.invert_yaxis()  # labels read top-to-bottom
    # # plt.xticks(rotation=15)
    # # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # # plt.ylim(89, 99)
    # # path1 = "./Results/interpretability_4.png"
    # # plt.savefig(path1)
    # # plt.show()
    # #
    # # y_pos1 = np.arange(5)
    # # fig = plt.figure()
    # # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # # A = [90.755, 92.756, 93.105, 94.7456, 95.1077]
    # # ax.bar(y_pos1, A)
    # # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    # #               fontname="Arial", fontsize=14,
    # #               fontweight='bold', color='k')
    # # # ax.invert_yaxis()  # labels read top-to-bottom
    # # plt.xticks(rotation=15)
    # # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # # plt.ylim(89, 99)
    # # path1 = "./Results/interpretability_5.png"
    # # plt.savefig(path1)
    # # plt.show()
    # #
    # # y_pos1 = np.arange(5)
    # # fig = plt.figure()
    # # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # # A = [93.05, 92, 93.84, 94.268, 95.81]
    # # ax.bar(y_pos1, A)
    # # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    # #               fontname="Arial", fontsize=14,
    # #               fontweight='bold', color='k')
    # # # ax.invert_yaxis()  # labels read top-to-bottom
    # # plt.xticks(rotation=15)
    # # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # # plt.ylim(89, 99)
    # # path1 = "./Results/interpretability_6.png"
    # # plt.savefig(path1)
    # # plt.show()


def Inter1():
    y_pos1 = np.arange(5)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    A = [94.08688, 95.041588, 94.874544, 95.99411, 96.07845]
    ax.bar(y_pos1, A)
    ax.set_xticks(y_pos1, labels=('LSTM', '1DCNN', 'DTCN', 'LSTM-\n1DCNN-\nDTCN-EL', 'ELS-HHO-\nIDENet '), fontname="Arial", fontsize=14,
                  fontweight='bold', color='k')
    # ax.invert_yaxis()  # labels read top-to-bottom
    plt.xticks(rotation=0)
    plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylim(89, 98)
    path1 = "./Results/Capter_3_interpretability_1.png"
    plt.savefig(path1)
    plt.show()

    y_pos1 = np.arange(5)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    A = [92.0545, 95.0525, 91.555, 94.545, 95.88854]
    ax.bar(y_pos1, A)
    ax.set_xticks(y_pos1, labels=('LSTM', '1DCNN', 'DTCN', 'LSTM-\n1DCNN-\nDTCN-EL', 'ELS-HHO-\nIDENet '), fontname="Arial", fontsize=14,
                  fontweight='bold', color='k')
    # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylim(89, 99)
    path1 = "./Results/Capter_3_interpretability_2.png"
    plt.savefig(path1)
    plt.show()

    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [93.05, 92, 93.84, 94.268, 95.81]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'), fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_3.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [88.05, 90, 92.498, 93.758, 94.075]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    #               fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_4.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [90.755, 92.756, 93.105, 94.7456, 95.1077]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    #               fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_5.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [93.05, 92, 93.84, 94.268, 95.81]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    #               fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_6.png"
    # plt.savefig(path1)
    # plt.show()


def Inter2():
    y_pos1 = np.arange(6)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    A = [94.06588, 95.013588, 94.895544, 95.99001, 96.58845, 97.05874]
    ax.bar(y_pos1, A)
    ax.set_xticks(y_pos1, labels=('XGBoost', 'Light\nGBM', 'CNN', 'RNN', 'FNN ', 'Deep\nEnsembleNet '), fontname="Arial", fontsize=14,
                  fontweight='bold', color='k')
    # ax.invert_yaxis()  # labels read top-to-bottom
    plt.xticks(rotation=0)
    plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylim(89, 98)
    path1 = "./Results/Chapter5_interpretability_1.png"
    plt.savefig(path1)
    plt.show()

    y_pos1 = np.arange(6)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    A = [92.9854, 95.9465, 91.945, 94.9141, 95.1056, 96.54514]
    ax.bar(y_pos1, A)
    ax.set_xticks(y_pos1, labels=('XGBoost', 'Light\nGBM', 'CNN', 'RNN', 'FNN ', 'Deep\nEnsembleNet '), fontname="Arial", fontsize=14,
                  fontweight='bold', color='k')
    # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylim(89, 99)
    path1 = "./Results/Chapter5_interpretability_2.png"
    plt.savefig(path1)
    plt.show()

    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [93.05, 92, 93.84, 94.268, 95.81]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'), fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_3.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [88.05, 90, 92.498, 93.758, 94.075]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    #               fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_4.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [90.755, 92.756, 93.105, 94.7456, 95.1077]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    #               fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_5.png"
    # plt.savefig(path1)
    # plt.show()
    #
    # y_pos1 = np.arange(5)
    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # A = [93.05, 92, 93.84, 94.268, 95.81]
    # ax.bar(y_pos1, A)
    # ax.set_xticks(y_pos1, labels=('Transformer\nNet', 'Efficient\nNet', 'SVM', 'GenNet', 'RUPBO-\nA-GenNet'),
    #               fontname="Arial", fontsize=14,
    #               fontweight='bold', color='k')
    # # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks(rotation=15)
    # plt.ylabel('Accuracy', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    # plt.ylim(89, 99)
    # path1 = "./Results/interpretability_6.png"
    # plt.savefig(path1)
    # plt.show()


Inter()
# Inter1()
# Inter2()
