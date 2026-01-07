from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter


def get_by_path(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    if "|" in path:
        for candidate in path.split("|"):
            value = get_by_path(obj, candidate.strip(), default=None)
            if value is not None:
                return value
        return default

    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def format_timestamp(value: Any) -> Any:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.utcfromtimestamp(value).strftime("%Y-%m-%d")
    return value


def format_timestamp_to_year(value: Any) -> Any:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.utcfromtimestamp(value).strftime("%Y")
    return value


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_mapping() -> List[Tuple[str, str]]:
    return [
        ("project_id", "data.id"),
        ("creator_id", "data.creator.id"),
        ("title", "data.name"),
        ("blurb", "data.blurb"),
        ("category", "data.category.name"),
        ("category_parent", "data.category.parent_name"),
        ("staff_pick", "data.staff_pick"),
        ("country", "data.country"),
        ("currency", "data.currency"),
        ("static_usd_rate", "data.static_usd_rate"),
        ("usd_goal", "__computed__"),
        ("launched_at", "data.launched_at"),
        ("deadline", "data.deadline"),
        ("creator_profile_url", "data.creator.urls.web.user"),
        ("project_url", "data.urls.web.project"),
        ("backers_count", "data.backers_count"),
        ("percent_funded", "data.percent_funded"),
        ("usd_pledged", "data.usd_pledged"),
        ("state", "data.state"),
    ]


def main() -> None:
    input = "data/metadata/all.json"
    all_output = "data/metadata/all.csv"
    mapping = build_mapping()
    headers = [name for name, _ in mapping]

    in_path = Path(input)
    all_out_path = Path(all_output)

    # 按年份分组存储数据
    yearly_data = {}
    yearly_counters = Counter()

    # 打开all.csv文件
    with all_out_path.open("w", encoding="utf-8", newline="") as all_f:
        all_writer = csv.writer(all_f)
        all_writer.writerow(headers)
        
        for row in iter_rows(in_path):
            state = get_by_path(row, "data.state", "")
            if state not in {"successful", "failed"}:
                continue
                
            # 获取项目年份
            launched_at_raw = get_by_path(row, "data.launched_at", None)
            if launched_at_raw and isinstance(launched_at_raw, (int, float)):
                year = datetime.utcfromtimestamp(launched_at_raw).strftime("%Y")
                
                # 剔除2014年及之前的项目
                if int(year) <= 2014:
                    continue
                    
                # 构建行数据
                row_map: Dict[str, Any] = {}
                for name, path in mapping:
                    value = get_by_path(row, path, "")
                    if name in {"launched_at", "deadline"}:
                        value = format_timestamp(value)
                    row_map[name] = value

                goal = get_by_path(row, "data.goal", None)
                static_rate = row_map.get("static_usd_rate")
                if isinstance(goal, (int, float)) and isinstance(static_rate, (int, float)):
                    row_map["usd_goal"] = goal * static_rate
                else:
                    row_map["usd_goal"] = ""

                values = [row_map.get(name, "") for name, _ in mapping]
                
                # 写入all.csv
                all_writer.writerow(values)
                
                # 按年份分组存储，用于后续创建年度文件
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(values)
                yearly_counters[year] += 1

    # 确保年度目录存在
    years_dir = Path("data/metadata/years")
    years_dir.mkdir(parents=True, exist_ok=True)

    # 写入每年的数据到对应的CSV文件
    for year, data in yearly_data.items():
        count = yearly_counters[year]
        filename = f"data/metadata/years/{year}_{count}.csv"
        file_path = Path(filename)
        
        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)
        
        print(f"已创建文件: {filename}，包含 {count} 个项目")

    print(f"已创建 all.csv 文件，包含 {sum(yearly_counters.values())} 个项目")
    print("处理完成！")


if __name__ == "__main__":
    main()