# crowdfunding

本仓库用于管理众筹结果预测项目。

## 目录结构

- src/ 代码入口
  - crawlers/ 爬虫
  - preprocess/ 数据预处理流程
    - clean/ 清理爬虫得到的数据
    - embedding/ 文本、图片向量化
  - scipts/ 临时脚本
- data/ 数据与缓存
  - projects/ 项目数据（一个项目一个文件夹）
    - <project_id>/ 项目目录
      - cover/ 封面图像和视频
      - photo/ story-content中的图像素材
      - content.json 项目内容标识文件
      - page.html 项目html
  - metadata/ 元数据
  - - years/ 按年存储
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
