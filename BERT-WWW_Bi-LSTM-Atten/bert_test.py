import os
import torch
from bert_get_data_model import BertClassifier, GenerateDate, BertClassifier1, BertClassifierWithAttention
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(y_true, y_pred, classes, normlize=False,
                          title='Confusion Matrix', cmap=plt.cm.Blues,
                          save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if normlize:
        cm = cm.astype('float') / cm.sum(axis=1).reshape(-1, 1)
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt=fmt,
                     cmap=cmap, square=True,
                     xticklabels=classes,
                     yticklabels=classes,
                     cbar=True)
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()



def confusion_matrix(y_true, y_pred):
    """手动计算混淆矩阵"""
    num_classes = len(np.unique(y_true))
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return cm


def calculate_metrics(cm):
    """手动计算各类别的准确率、召回率、F1值以及加权平均值"""
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.sum(cm, axis=1)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    # 计算加权平均值
    weights = support / np.sum(support)
    weighted_precision = np.sum(precision * weights)
    weighted_recall = np.sum(recall * weights)
    weighted_f1 = np.sum(f1 * weights)

    return precision, recall, f1, support, weighted_precision, weighted_recall, weighted_f1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = './bert_checkpoint_waimai'
path = '_atten'
name = 'BERT'
if path == '_atten':
    model = BertClassifierWithAttention()
    name = 'BERT_Bi-LSTM_Atten'
elif path == '_1':
    name = 'BERT_Bi-LSTM'
    model = BertClassifier1()
else:
    model = BertClassifier()

model.load_state_dict(torch.load(os.path.join(save_path, f'best{path}.pt')))
model.to(device)
model.eval()

y_pred = []
y_true = []
classes = ["消极", "积极"]


def evaluate(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=16)
    total_acc_test = 0
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            input_ids = test_inputs['input_ids'].squeeze(1).to(device)
            mask = test_inputs['attention_mask'].to(device)
            test_labels = test_labels.to(device)
            output = model(input_ids, mask)
            acc = (output.argmax(dim=1) == test_labels).sum().item()
            total_acc_test += acc

            y_true.extend(test_labels.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算各项指标
    precision, recall, f1, support, w_precision, w_recall, w_f1 = calculate_metrics(cm)

    # 输出结果
    print("\n" + "=" * 60)
    print(f"{name}_Classification Report:")
    print("=" * 60)
    print(f"{'类别':<10}{'精确率':<10}{'召回率':<10}{'F1值':<10}{'样本数'}")
    for i in range(len(classes)):
        print(f"{classes[i]:<10}{precision[i]:<10.4f}{recall[i]:<10.4f}{f1[i]:<10.4f}{support[i]:<10}")

    print("\n加权平均值:")
    print(f"{'精确率':<10}{'召回率':<10}{'F1值':<10}")
    print(f"{w_precision:<10.4f}{w_recall:<10.4f}{w_f1:<10.4f}")

    print(f'\n测试准确率: {total_acc_test / len(dataset):.4f}')


test_dataset = GenerateDate("test")
evaluate(model, test_dataset)

plot_confusion_matrix(y_true, y_pred, classes=classes,save_path="./")