# crowdfunding

本仓库用于管理众筹结果预测项目。

## 目录结构

- src/ 代码入口
  - crawlers/ 爬虫
  - pipelines/ 清洗、切分、对齐等数据流程
  - storage/ 存储读写与统一接口
  - utils/ 通用工具
- data/ 数据与缓存
  - projects/ 项目数据（一个项目一个文件夹）
    - <project_id>/ 项目目录
      - cover/ 封面图像
      - photo/ 图像素材
      - word/ 文本内容
      - sequence_index.json 图文顺序标识文件
  - cache/ 训练与特征缓存
    - features/ 特征缓存
    - training/ 训练所需临时数据
  - metadata/ 元数据
    - years/ 按年存储
    - all.json 来自webrobots.io/kickstarter-datasets，2025-12-18
    - all.csv 筛选字段后的总数据
- experiments/ 实验管理
  - runs/ 实验运行产物
  - logs/ 实验日志
- models/ 模型产物
  - checkpoints/ 训练中断点
  - exports/ 导出模型
- artifacts/ 其他产物
  - reports/ 报告
  - figures/ 可视化图表
- docs/ 文档
- scripts/ 脚本
