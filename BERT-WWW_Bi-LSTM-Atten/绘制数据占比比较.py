import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置全局样式（论文级美观设置）
# plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold'
})


df1 = pd.read_csv('./data/ChnSentiCorp_htl_all.csv')
df2 = pd.read_csv('./data/waimai_10k.csv')

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Dataset Label Distribution Comparison', y=1.05)


# 子图1：饼图
def plot_pie(ax, df, title):
    counts = df['label'].value_counts()
    colors = ['#66b3ff', '#ff9999']
    labels = ['Negative (0)', 'Positive (1)']

    wedges, texts, autotexts = ax.pie(
        counts,
        colors=colors,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
    )

    # 美化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    ax.set_title(title, pad=20)
    ax.axis('equal')  # 保持圆形


plot_pie(ax1, df1, 'ChnSentiCorp_htl_all')
plot_pie(ax2, df2, 'waimai_10k')

# 子图2：直方图（保存为单独图片）
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Label Distribution Histogram', y=1.05)


def plot_hist(ax, df, title):
    sns.countplot(
        x='label',
        data=df,
        ax=ax,
        palette=['#66b3ff', '#ff9999'],
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_xlabel('')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title(title, pad=15)

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )


plot_hist(ax3, df1, 'ChnSentiCorp_htl_all')
plot_hist(ax4, df2, 'waimai_10k')

# 调整布局并保存
plt.tight_layout()
fig.savefig('label_distribution_pie.png', bbox_inches='tight', transparent=True)
fig2.savefig('label_distribution_hist.png', bbox_inches='tight', transparent=True)
plt.close('all')

print("可视化图表已保存为高分辨率PNG文件")