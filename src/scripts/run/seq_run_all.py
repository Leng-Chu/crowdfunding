"""
SEQ模型实验运行脚本

使用样例:

# 1. 运行所有baseline模式，使用指定种子（默认为42）
python seq_run_all.py all --seed 42

# 2. 运行特定baseline模式，使用不同种子范围
#    同时运行有meta和无meta的实验
python seq_run_all.py single --baseline-mode trm_pos --start-seed 42 --end-seed 46

# 3. 运行set_mean模式，使用种子40到50
python seq_run_all.py single --baseline-mode set_mean --start-seed 40 --end-seed 50

# 4. 运行所有baseline，并关闭文本长度/图片面积属性
python seq_run_all.py all --seed 42 --no-use-seq-attr
"""

import subprocess
import sys
import threading
import time
import argparse
from typing import List, Optional


def run_command(cmd: List[str], experiment_name: str):
    """
    运行单个命令
    """
    print(f"正在启动实验: {experiment_name}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, shell=False, check=True)
        print(f"实验 '{experiment_name}' 完成，返回码: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"实验 '{experiment_name}' 失败，返回码: {e.returncode}")


def _run_command_group(commands: List[tuple[str, str]], group_name: str) -> None:
    print(group_name)
    print("-" * 50)

    threads = []
    for cmd_str, experiment_name in commands:
        cmd = cmd_str.split()
        print(f"启动实验: {experiment_name}")
        thread = threading.Thread(target=run_command, args=(cmd, experiment_name))
        threads.append(thread)
        thread.start()

        # 等待一小段时间再启动下一个实验，避免瞬时资源冲突
        time.sleep(2)

    for thread in threads:
        thread.join()


def _format_seq_attr_flag(use_seq_attr: Optional[bool]) -> str:
    """
    生成 seq_attr 参数字符串：
    - True  -> --use-seq-attr
    - False -> --no-use-seq-attr
    - None  -> 不显式传参（沿用 config.py 默认值）
    """
    if use_seq_attr is True:
        return " --use-seq-attr"
    if use_seq_attr is False:
        return " --no-use-seq-attr"
    return ""


def _format_seq_attr_label(use_seq_attr: Optional[bool]) -> str:
    """
    生成用于打印/实验名的属性标签。
    """
    if use_seq_attr is True:
        return "seq_attr=on"
    if use_seq_attr is False:
        return "seq_attr=off"
    return "seq_attr=default"


def generate_all_commands(seed: int = 42, use_seq_attr: Optional[bool] = None) -> List[tuple[str, str]]:
    """
    生成所有命令，可指定随机数种子
    """
    seq_attr_flag = _format_seq_attr_flag(use_seq_attr)
    seq_attr_label = _format_seq_attr_label(use_seq_attr)
    return [
        # 使用meta数据的不同baseline模式
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode set_mean --use-meta --device cuda:0{seq_attr_flag}", 
         f"Seq: CLIP Image+Text with Meta - Set Mean (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode set_attn --use-meta --device cuda:1{seq_attr_flag}", 
         f"Seq: CLIP Image+Text with Meta - Set Attention (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode trm_no_pos --use-meta --device cuda:2{seq_attr_flag}", 
         f"Seq: CLIP Image+Text with Meta - Transformer No Position (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode trm_pos --use-meta --device cuda:3{seq_attr_flag}", 
         f"Seq: CLIP Image+Text with Meta - Transformer With Position (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode trm_pos_shuffled --use-meta --device cuda:0{seq_attr_flag}", 
         f"Seq: CLIP Image+Text with Meta - Transformer With Shuffled Position (seed {seed}, {seq_attr_label})"),
         
        # 不使用meta数据的不同baseline模式
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode set_mean --no-use-meta --device cuda:1{seq_attr_flag}", 
         f"Seq: CLIP Image+Text - Set Mean (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode set_attn --no-use-meta --device cuda:2{seq_attr_flag}", 
         f"Seq: CLIP Image+Text - Set Attention (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode trm_no_pos --no-use-meta --device cuda:3{seq_attr_flag}", 
         f"Seq: CLIP Image+Text - Transformer No Position (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode trm_pos --no-use-meta --device cuda:0{seq_attr_flag}", 
         f"Seq: CLIP Image+Text - Transformer With Position (seed {seed}, {seq_attr_label})"),
        (f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode trm_pos_shuffled --no-use-meta --device cuda:1{seq_attr_flag}", 
         f"Seq: CLIP Image+Text - Transformer With Shuffled Position (seed {seed}, {seq_attr_label})")
    ]


def generate_single_baseline_commands(
    baseline_mode: str,
    start_seed: int,
    end_seed: int,
    use_seq_attr: Optional[bool] = None,
) -> List[tuple[str, str]]:
    """
    生成特定baseline模式的不同随机数命令，同时包括use-meta和no-use-meta的情况
    """
    commands = []
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    seq_attr_flag = _format_seq_attr_flag(use_seq_attr)
    seq_attr_label = _format_seq_attr_label(use_seq_attr)
    
    # 为每个种子分别创建有meta和无meta的命令
    for seed in range(start_seed, end_seed + 1):
        # 有meta的命令
        device_with_meta = devices[len(commands) % len(devices)]
        cmd_with_meta = f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode {baseline_mode} --use-meta --device {device_with_meta}{seq_attr_flag}"
        exp_name_with_meta = f"Seq: {baseline_mode}+meta (seed {seed}, {seq_attr_label})"
        commands.append((cmd_with_meta, exp_name_with_meta))
        
        # 无meta的命令
        device_without_meta = devices[len(commands) % len(devices)]
        cmd_without_meta = f"conda run -n crowdfunding python src/dl/seq/main.py --run-name {seed} --seed {seed} --baseline-mode {baseline_mode} --no-use-meta --device {device_without_meta}{seq_attr_flag}"
        exp_name_without_meta = f"Seq: {baseline_mode} (seed {seed}, {seq_attr_label})"
        commands.append((cmd_without_meta, exp_name_without_meta))
    
    return commands


def run_all_experiments(args) -> None:
    """
    运行所有seq实验
    """
    if args.mode == "all":
        # 运行所有命令，使用指定的随机数种子
        all_commands = generate_all_commands(args.seed, args.use_seq_attr)
        _run_command_group(
            all_commands,
            f"所有baseline (seed {args.seed}, {_format_seq_attr_label(args.use_seq_attr)})",
        )
    elif args.mode == "single":
        # 运行特定baseline模式，使用不同随机数，同时包括use-meta和no-use-meta
        single_commands = generate_single_baseline_commands(
            args.baseline_mode, 
            args.start_seed, 
            args.end_seed,
            args.use_seq_attr,
        )
        _run_command_group(
            single_commands,
            (
                f"Baseline {args.baseline_mode}, seeds {args.start_seed}-{args.end_seed} "
                f"(+meta & -meta, {_format_seq_attr_label(args.use_seq_attr)})"
            ),
        )
    
    print("所有实验已完成！")


def main():
    parser = argparse.ArgumentParser(description="运行SEQ模型的所有实验")
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 所有命令模式
    all_parser = subparsers.add_parser('all', help='运行所有baseline模式')
    all_parser.add_argument('--seed', type=int, default=42, help='随机数种子，默认为42')
    all_parser.add_argument(
        '--use-seq-attr',
        default=None,
        action=argparse.BooleanOptionalAction,
        help='是否启用文本长度/图片面积属性（不传则沿用 config.py 默认值）',
    )
    
    # 单一baseline模式
    single_parser = subparsers.add_parser('single', help='运行单一baseline模式')
    single_parser.add_argument('--baseline-mode', required=True, choices=['set_mean', 'set_attn', 'trm_no_pos', 'trm_pos', 'trm_pos_shuffled'], 
                              help='选择baseline模式')
    single_parser.add_argument('--start-seed', type=int, default=42, help='起始随机数种子，默认为42')
    single_parser.add_argument('--end-seed', type=int, default=46, help='结束随机数种子，默认为46')
    single_parser.add_argument(
        '--use-seq-attr',
        default=None,
        action=argparse.BooleanOptionalAction,
        help='是否启用文本长度/图片面积属性（不传则沿用 config.py 默认值）',
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        # 如果没有指定模式，默认运行所有命令，种子为42
        args.mode = "all"
        args.seed = 42
        args.use_seq_attr = None
    
    print(f"一键运行SEQ实验脚本 (模式: {args.mode})")
    print("="*50)
    run_all_experiments(args)


if __name__ == "__main__":
    main()
