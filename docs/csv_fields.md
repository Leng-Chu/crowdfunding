# CSV 字段说明

本文件说明由 `src/crawlers/json_to_table.py` 生成的 CSV 中字段含义。读取的 JSON 文件通常来源于 https://webrobots.io/kickstarter-datasets/ 。
时间字段已转换为 `YYYY-MM-DD`。

## 字段解释

- project_id: 项目 ID
- creator_id: 创建者 ID
- title: 项目标题
- blurb: 项目简介/一句话描述
- category: 类别名称
- category_parent: 父级类别名称
- staff_pick: 是否平台精选
- country: 项目国家代码
- currency: 原币种代码
- static_usd_rate: 静态 USD 汇率
- usd_goal: 目标金额换算为 USD（goal * static_usd_rate）
- launched_at: 上线/发起时间（日期）
- deadline: 截止时间（日期）
- creator_profile_url: 创建者主页 URL
- project_url: 项目页面 URL
- backers_count: 支持者数量
- percent_funded: 完成比例（百分比数值，例如 103.6 表示 103.6%）
- usd_pledged: 以 USD 计的募集金额
- state: 项目状态（successful/failed）
