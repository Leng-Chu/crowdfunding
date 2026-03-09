import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据并计算 5 次实验的平均值
df = pd.read_csv('experiments\\all.csv')
mean_df = df.groupby('mode')[['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']].mean().reset_index()

# 2. 筛选对比模型
target_models = ['NoAttr-MSTC', 'MSTC']
# target_models = ['Mean-Pool', 'MDL', 'Late-MSTC', 'MSTC']
# target_models = ['Attn-Pool', 'NoPos-MSTC', 'Shuffled-MSTC', 'MSTC']
mean_df = mean_df[mean_df['mode'].isin(target_models)].copy()
mean_df['mode'] = pd.Categorical(mean_df['mode'], categories=target_models, ordered=True)
mean_df = mean_df.sort_values('mode')

# 3. 字体乱码终极解决方案 (针对 Windows)
# 强制设置全局字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# 设置 Seaborn 主题，并显式指定字体，防止其覆盖 Matplotlib 设置
sns.set_theme(style="whitegrid", font='Microsoft YaHei')

# 4. 配置子图参数
metrics_config = [
    ('test_accuracy', '准确率 (Accuracy)', (0.82, 0.84)),
    ('test_precision', '精确率 (Precision)', (0.785, 0.815)),
    ('test_recall', '召回率 (Recall)', (0.87, 0.885)),
    ('test_f1', 'F1分数 (F1-score)', (0.83, 0.845)),
    ('test_auc', 'AUC', (0.91, 0.925))
]

# 5. 开始画图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (col, name, ylim) in enumerate(metrics_config):
    ax = axes[i]
    
    # --- 修复 FutureWarning 的核心位置 ---
    # 添加 hue='mode' 并设置 legend=False
    sns.barplot(
        data=mean_df, 
        x='mode', 
        y=col, 
        hue='mode',       # 将 x 轴变量也赋值给 hue
        palette='viridis', 
        edgecolor='black', 
        ax=ax,
        legend=False      # 隐藏子图内的图例，因为横轴已经有标签了
    )
    
    # 细节设置
    # 使用 fontdict 显式指定字体属性，确保标题不会乱码
    ax.set_title(name, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(ylim)
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    # 数值标注
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.4f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 8), textcoords='offset points', fontsize=10)

# 隐藏多余的第6个子图
axes[5].axis('off')

# 整体大标题
plt.suptitle('各模型平均性能对比', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存并显示
# bbox_inches='tight' 非常重要，确保边缘文字不被截断
plt.savefig('442.png', dpi=300, bbox_inches='tight')
plt.show()