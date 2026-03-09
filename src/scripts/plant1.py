import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('experiments\\all.csv')

# 2. 筛选对比模型
# target_models = ['Mean-Pool', 'MDL', 'Late-MSTC', 'MSTC']
target_models = ['Attn-Pool', 'NoPos-MSTC', 'Shuffled-MSTC', 'MSTC']
df_plot = df[df['mode'].isin(target_models)].copy()
df_plot['mode'] = pd.Categorical(df_plot['mode'], categories=target_models, ordered=True)

# 3. 解决中文乱码的关键配置
# 优先使用微软雅黑(Microsoft YaHei)，其次黑体(SimHei)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Tahoma', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sns.set_theme(style="whitegrid")
# 强制对 seaborn 也应用字体设置
sns.set(font='Microsoft YaHei', style="whitegrid")

def save_grouped_bar_zh(metric_column, metric_name, file_name, ylim):
    # 创建画布，增加宽度以容纳右侧图例
    plt.figure(figsize=(11, 6))
    
    # 绘制柱状图
    ax = sns.barplot(
        data=df_plot, 
        x='seed', 
        y=metric_column, 
        hue='mode', 
        palette='muted',
        edgecolor='black',
        linewidth=0.8
    )
    
    # 设置中文标题和标签
    plt.title(f'不同随机种子下的{metric_name}对比', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel('数据集随机种子', fontsize=12)
    
    # 调整纵轴范围
    plt.ylim(ylim)
    
    # 图例设置：放到图表右侧外部 (bbox_to_anchor)，确保不遮挡柱子
    # borderaxespad=0.5 增加图例与图表的间距
    plt.legend(title='模型', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.5)
    
    # 自动调整布局，防止标签和图例被截断
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"成功保存: {file_name}")
    plt.show()
    plt.close()

# 4. 执行绘图
# 注意：ylim 的范围根据你的数据微调，确保柱状图有明显的起伏感
save_grouped_bar_zh('test_accuracy', '准确率 (Accuracy) ', 'accuracy_441.png', (0.80, 0.845))
save_grouped_bar_zh('test_f1', ' F1-score ', 'f1_441.png', (0.815, 0.85))
save_grouped_bar_zh('test_auc', ' AUC ', 'auc_441.png', (0.90, 0.925))