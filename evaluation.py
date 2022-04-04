import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import os


def plot_roc(fpr, tpr, auc_score, filename):
    f = plt.figure(figsize=(5, 3.5))
    plt.plot(fpr, tpr, color='r')
    # plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
    # plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
    # plt.tick_params(labelsize=23)
    # plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
    # plt.xticks(np.arange(0, 1.01, step=0.2))
    # plt.xlim([-0.001, 0.011])
    # plt.xticks(np.arange(0, 0.01, step=0.001))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if 'feco' in filename:
        line_label = 'feco'
    else:
        line_label = os.path.split(filename)[-1]
    plt.legend([line_label + ' (AUC)={0:5.2f}'.format(auc_score), 'None'])
    plt.show()
    f.savefig(filename + '.pdf')


def split_evaluate(y, scores, plot=False, filename=None, manual_th=None, perform_dict=None):
    # compute FPR TPR
    pos_idx = np.where(y==1)
    neg_idx = np.where(y==0)
    score_pos = np.sum(scores[pos_idx])/len(pos_idx)
    score_neg = np.sum(scores[neg_idx])/len(neg_idx)

    if score_pos < score_neg:
        fpr, tpr, thresholds = roc_curve(y, -scores)
        thresholds = -thresholds
    else:
        fpr, tpr, thresholds = roc_curve(y, scores)
        thresholds = thresholds

    auc_score = auc(fpr, tpr)
    if plot:
        plot_roc(fpr, tpr, auc_score, filename)

    pos1 = np.where(fpr <= 0.005)[0]
    pos2 = np.where(fpr <= 0.01)[0]
    print(f'AUC: {auc_score}')
    print(f'TPR(FPR=0.005): {tpr[pos1[-1]]:.4f}, threshold: {thresholds[pos1[-1]]}\n')
    print(f'TPR(FPR=0.01): {tpr[pos2[-1]]:.4f}, threshold: {thresholds[pos2[-1]]}\n')
    pos1 = np.where(fpr <= 0.0005)[0]
    pos2 = np.where(fpr <= 0.001)[0]
    print(f'TPR(FPR=0.0005): {tpr[pos1[-1]]:.4f}, threshold: {thresholds[pos1[-1]]}\n')
    print(f'TPR(FPR=0.001): {tpr[pos2[-1]]:.4f}, threshold: {thresholds[pos2[-1]]}\n')

    # if the labels y is with 1, -1, then transform them to 1, 0
    if -1 in y:
        y = ((y + 1)/2).astype(int)

    # save scores to file
    labels_scores = np.concatenate((y.reshape(-1, 1), scores.reshape(-1, 1)), axis=1)
    if filename is not None:
        np.save(file=filename+'_labels_scores.npy', arr=labels_scores)

    # get the accuracy
    total_a = np.sum(y)
    total_n = len(y) - total_a
    best_acc = 0

    # evaluate the accuracy of normal set and anormal set separately using various threshold
    # acc = 0
    # total_correct_a = np.zeros(len(thresholds))
    # total_correct_n = np.zeros(len(thresholds))
    # if n_th > 500:
    #     thresholds_new = [thresholds[i] for i in range(n_th) if tpr[i]<1]

    # for i, th in enumerate(thresholds):
    #     if i % 500 == 0:
    #         print('evaluating threshold {}/{}'.format(i, len(thresholds)))
    #     y_pred = scores <= th
    #     correct = y_pred == y
    #     total_correct_a[i] += np.sum(correct[np.where(y == 1)])
    #     total_correct_n[i] += np.sum(correct[np.where(y == 0)])
    #
    # acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    # acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    # acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(thresholds))]
    # best_acc = np.max(acc)
    # idx = np.argmax(acc)
    # best_threshold = thresholds[idx]
    #
    # print('Best ACC: {:.4f} | Threshold: {:.4f} | ACC_normal={:.4f} | ACC_anormal={:.4f}\n'.
    #       format(best_acc, best_threshold, acc_n[idx], acc_a[idx]))

    if manual_th is not None:
        print(f'Manually choose decision threshold: {manual_th}')
    else:
        idxs = np.where(tpr >= 0.9998)
        id = idxs[0][0]
        manual_th = thresholds[id]
        print(f'choose decision threshold: {manual_th}')
    if score_pos < score_neg:
        y_pred = scores <= manual_th
    else:
        y_pred = scores >= manual_th
    correct = y_pred == y
    correct_a = np.sum(correct[np.where(y == 1)])
    correct_n = np.sum(correct[np.where(y == 0)])

    acc_n = correct_n / total_n
    acc_a = correct_a / total_a
    acc = (correct_n + correct_a) / (total_n + total_a)
    print('ACC: {:.4f} | Threshold: {:.4f} | ACC_normal={:.4f} | ACC_anormal={:.4f}\n'.
          format(acc, manual_th, acc_n, acc_a))

    recall = correct_a/total_a
    precision = correct_a/(total_n-correct_n+correct_a)
    fpr = (total_n - correct_n)/total_n

    print('Recall: {:.4f} | Precision: {:.4f} | fpr={:.4f} | f1={:.4f} \n'.
          format(recall, precision, fpr, 2*recall*precision/(recall+precision)))
    if perform_dict is not None:
        perform_dict['threshold'] = manual_th
        perform_dict['auc'] = auc_score
        perform_dict['acc'] = acc
        perform_dict['recall'] = recall
        perform_dict['precision'] = precision
        perform_dict['fpr'] = fpr

    return best_acc, acc, auc_score
