"""
LATE模型实验运行脚本

使用样例:

# 1. 运行所有baseline模式，使用指定种子（默认为42）
python late_run_all.py all --seed 42

# 2. 运行特定baseline模式，使用不同种子范围
#    同时运行有meta和无meta的实验
python late_run_all.py single --baseline-mode trm_pos --start-seed 42 --end-seed 46

# 3. 运行mean_pool模式，使用种子40到50
python late_run_all.py single --baseline-mode mean_pool --start-seed 40 --end-seed 50
"""

import subprocess
import threading
import time
import argparse
from typing import List


def run_command(cmd: List[str], experiment_name: str) -> None:
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


def generate_all_commands(seed: int = 42) -> List[tuple[str, str]]:
    """
    生成所有命令，可指定随机数种子
    """
    return [
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode mean_pool --use-meta --device cuda:0",
         f"Late: late_mean_pool+meta (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode attn_pool --use-meta --device cuda:1",
         f"Late: late_attn_pool+meta (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode trm_no_pos --use-meta --device cuda:2",
         f"Late: late_trm_no_pos+meta (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode trm_pos --use-meta --device cuda:3",
         f"Late: late_trm_pos+meta (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode mean_pool --no-use-meta --device cuda:0",
         f"Late: late_mean_pool (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode attn_pool --no-use-meta --device cuda:1",
         f"Late: late_attn_pool (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode trm_no_pos --no-use-meta --device cuda:2",
         f"Late: late_trm_no_pos (seed {seed})"),
        (f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode trm_pos --no-use-meta --device cuda:3",
         f"Late: late_trm_pos (seed {seed})")
    ]


def generate_single_baseline_commands(baseline_mode: str, start_seed: int, end_seed: int) -> List[tuple[str, str]]:
    """
    生成特定baseline模式的不同随机数命令，同时包括use-meta和no-use-meta的情况
    """
    commands = []
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    
    # 为每个种子分别创建有meta和无meta的命令
    for seed in range(start_seed, end_seed + 1):
        # 有meta的命令
        device_with_meta = devices[len(commands) % len(devices)]
        cmd_with_meta = f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode {baseline_mode} --use-meta --device {device_with_meta}"
        exp_name_with_meta = f"Late: late_{baseline_mode}+meta (seed {seed})"
        commands.append((cmd_with_meta, exp_name_with_meta))
        
        # 无meta的命令
        device_without_meta = devices[len(commands) % len(devices)]
        cmd_without_meta = f"conda run -n crowdfunding python src/dl/late/main.py --run-name {seed} --seed {seed} --baseline-mode {baseline_mode} --no-use-meta --device {device_without_meta}"
        exp_name_without_meta = f"Late: late_{baseline_mode} (seed {seed})"
        commands.append((cmd_without_meta, exp_name_without_meta))
    
    return commands


def run_all_experiments(args) -> None:
    """
    运行所有late实验
    """
    if args.mode == "all":
        # 运行所有命令，使用指定的随机数种子
        all_commands = generate_all_commands(args.seed)
        _run_command_group(all_commands, f"所有baseline (seed {args.seed})")
    elif args.mode == "single":
        # 运行特定baseline模式，使用不同随机数
        single_commands = generate_single_baseline_commands(
            args.baseline_mode, 
            args.start_seed, 
            args.end_seed, 
            args.use_meta
        )
        meta_desc = "+meta" if args.use_meta else ""
        _run_command_group(single_commands, f"Baseline {args.baseline_mode}{meta_desc}, seeds {args.start_seed}-{args.end_seed}")
    
    print("所有实验已完成！")


def main():
    parser = argparse.ArgumentParser(description="运行LATE模型的所有实验")
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 所有命令模式
    all_parser = subparsers.add_parser('all', help='运行所有baseline模式')
    all_parser.add_argument('--seed', type=int, default=42, help='随机数种子，默认为42')
    
    # 单一baseline模式
    single_parser = subparsers.add_parser('single', help='运行单一baseline模式')
    single_parser.add_argument('--baseline-mode', required=True, choices=['mean_pool', 'attn_pool', 'trm_no_pos', 'trm_pos'], 
                              help='选择baseline模式')
    single_parser.add_argument('--start-seed', type=int, default=42, help='起始随机数种子，默认为42')
    single_parser.add_argument('--end-seed', type=int, default=46, help='结束随机数种子，默认为46')
    
    args = parser.parse_args()
    
    if not args.mode:
        # 如果没有指定模式，默认运行所有命令，种子为42
        args.mode = "all"
        args.seed = 42
    
    print(f"一键运行LATE实验脚本 (模式: {args.mode})")
    print("="*50)
    run_all_experiments(args)


if __name__ == "__main__":
    main()