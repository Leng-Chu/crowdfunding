# crowdfunding

本仓库用于管理众筹结果预测项目。

## 目录结构

- src/ 代码入口
  - crawlers/ 爬虫
  - preprocess/ 数据预处理流程
    - clean/ 清理爬虫得到的数据
    - embedding/ 文本、图片向量化
    - table/ 表格数据处理与特征工程
  - dl/ 深度学习代码
  - scipts/ 临时脚本
- data/ 数据与缓存
  - projects/ 项目数据（一个项目一个文件夹）
    - <project_id>/ 项目目录
      - cover/ 封面图像和视频
      - photo/ story-content中的图像素材
      - content.json 项目内容标识文件
      - page.html 项目html
  - metadata/ 元数据
- experiments/ 实验管理
  - <experiment_name>/ 实验名
    - model/ 实验运行产物
    - log/ 实验日志
- docs/ 文档
