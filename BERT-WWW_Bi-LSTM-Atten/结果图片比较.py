import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ==================== 数据配置 ====================
datasets = ['ChnSentiCorp', 'Waimai_10k']
models = ['BERT', 'BERT+BiLSTM', 'BERT+BiLSTM+Attention']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

# 示例数据 (请替换为实际数据)
data = {
    'ChnSentiCorp': {
        'BERT': [0.8970, 0.8961, 0.8970, 0.8958],
        'BERT+BiLSTM': [0.9003, 0.8994, 0.9003, 0.8996],
        'BERT+BiLSTM+Attention': [0.9062, 0.9058, 0.9062, 0.9058]
    },
    'Waimai_10k': {
        'BERT': [0.9145, 0.9157, 0.9145, 0.9149],
        'BERT+BiLSTM': [0.9174, 0.9185, 0.9174, 0.9178],
        'BERT+BiLSTM+Attention': [0.9212, 0.9211, 0.9212, 0.9211]
    }
}

# ==================== 绘图设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']  # 蓝, 绿, 红, 橙

# 计算全局最小值和最大值来确定y轴范围
all_values = np.array([value for dataset in data.values() for model in dataset.values() for value in model])
global_min = np.min(all_values) - 0.02
global_max = np.max(all_values) + 0.02

# ==================== 绘制每个子图 ====================
for ax, dataset in zip(axes, datasets):
    # 计算柱状图位置
    x = np.arange(len(models))
    width = 0.18  # 稍微减小宽度

    # 绘制每组指标
    for i, metric in enumerate(metrics):
        values = [data[dataset][model][i] for model in models]
        # 使用更明显的颜色和样式
        bars = ax.bar(x + i * width, values, width, color=colors[i], label=metric,
                      edgecolor='black', linewidth=0.8, alpha=0.9)

        # 添加数值标签，使用更醒目的格式
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.003,
                    f'{height:.4f}',  # 显示4位小数
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 学术化样式调整
    ax.set_title(dataset, pad=12, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=0, fontweight='bold')
    ax.set_ylim(global_min, global_max)  # 使用全局范围
    ax.yaxis.set_major_locator(MaxNLocator(10))  # 增加刻度数量
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加基准线
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# ==================== 全局设置 ====================
plt.suptitle("Performance Comparison of Three Models on Two Datasets",
             y=1.02, fontsize=16, fontweight='bold')
axes[0].set_ylabel('Score', labelpad=12, fontsize=13)

# 优化图例位置和样式
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.92, 0.95),
           ncol=4, fontsize=11, framealpha=1, borderpad=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局
plt.savefig('model_comparison_enhanced.png', bbox_inches='tight', dpi=600)
plt.show()