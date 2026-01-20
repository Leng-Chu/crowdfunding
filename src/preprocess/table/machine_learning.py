import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 统一使用以仓库根目录为基准的相对路径（避免不同机器/不同工作目录下路径失效）
def _get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


REPO_ROOT = _get_repo_root()

# 读取csv文件
df = pd.read_csv(REPO_ROOT / "data" / "metadata" / "now_processed.csv")
# 结果保存目录
output_dir = REPO_ROOT / "experiments" / "meta_ml"
output_dir.mkdir(parents=True, exist_ok=True)

# 提取特征和标签
X = df.drop(['state', 'project_id', 'time'], axis=1)  # 删除标签列和不需要的列
y = df['state']  # 标签列
print(X.columns)

# 手动指定数值特征和类别特征
numerical_cols = ['duration_days', 'log_usd_goal']  # 例如：你认为这些是数值型特征
categorical_cols = ['category', 'currency', 'country']  # 你可以手动指定所有的类别特征

# 对类别特征进行 one-hot 编码
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)  # one-hot 编码

# 对数值特征进行标准化
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

def evaluate_models(X_train, X_test, y_train, y_test, split_method):
    """评估模型的函数"""
    # 定义多个机器学习算法
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # 创建结果报告
    results_report = []
    results_report.append(f"Split Method: {split_method}")
    results_report.append("=" * 50)

    # 训练模型并评估
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算模型性能
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, digits=6)
        
        # 输出模型性能
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{class_report}")
        print("-" * 50)
        
        # 添加到结果报告
        results_report.append(f"Model: {name}")
        results_report.append(f"Accuracy: {accuracy}")
        results_report.append(f"Classification Report:\n{class_report}")
        results_report.append("-" * 50)

    return results_report

# 1. 随机划分数据集
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, test_size=0.2, random_state=42
)

random_results = evaluate_models(
    X_train_random, X_test_random, y_train_random, y_test_random, 
    "Random Split"
)

# 2. 按CSV顺序划分（前80%作为训练集，后20%作为测试集）
split_idx = int(len(X) * 0.8)
X_train_sequential = X.iloc[:split_idx]
X_test_sequential = X.iloc[split_idx:]
y_train_sequential = y.iloc[:split_idx]
y_test_sequential = y.iloc[split_idx:]

sequential_results = evaluate_models(
    X_train_sequential, X_test_sequential, y_train_sequential, y_test_sequential, 
    "Sequential Split (by CSV order)"
)

# 合并两种方法的结果
all_results = random_results + ["\n"] + sequential_results

# 保存结果到文件
output_path = output_dir / "ml_results_report.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write('\n'.join(all_results))

print(f"Results saved to {output_path}")
