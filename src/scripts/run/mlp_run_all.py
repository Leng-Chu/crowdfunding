"""
MLP 模型实验运行脚本

使用示例:

# 1. 运行单个 seed 的全部组合（默认 seed=42）
python src/scripts/run/mlp_run_all.py all --seed 42

# 2. 在一个 seed 区间内同时运行全部组合
python src/scripts/run/mlp_run_all.py single --start-seed 42 --end-seed 46

# 3. 只运行区间内的 image_text_meta 组合
python src/scripts/run/mlp_run_all.py single --experiment-mode image_text_meta --start-seed 40 --end-seed 50
"""

import argparse
import subprocess
import threading
import time
from typing import List, Tuple


CommandItem = Tuple[str, str]


def run_command(cmd: List[str], experiment_name: str) -> None:
    """
    运行单个命令
    """
    print(f"正在启动实验: {experiment_name}")
    print(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, shell=False, check=True)
        print(f"实验 '{experiment_name}' 完成，返回码: {result.returncode}")
    except subprocess.CalledProcessError as err:
        print(f"实验 '{experiment_name}' 失败，返回码: {err.returncode}")


def _run_command_group(commands: List[CommandItem], group_name: str) -> None:
    print(group_name)
    print("-" * 50)

    threads = []
    for cmd_str, experiment_name in commands:
        cmd = cmd_str.split()
        print(f"启动实验: {experiment_name}")
        thread = threading.Thread(target=run_command, args=(cmd, experiment_name))
        threads.append(thread)
        thread.start()

        # 适当错峰启动，降低瞬时资源冲突风险
        time.sleep(2)

    for thread in threads:
        thread.join()


def _build_command(seed: int, device: str, experiment_mode: str) -> CommandItem:
    base_prefix = (
        "conda run -n crowdfunding python src/dl/mlp/main.py "
        f"--run-name {seed} --seed {seed} "
        "--image-embedding-type clip --text-embedding-type clip "
    )
    mode_to_suffix = {
        "image_text_meta": "--use-meta",
        "image_text": "--no-use-meta",
    }
    mode_to_name = {
        "image_text_meta": "MLP: clip image+text+meta",
        "image_text": "MLP: clip image+text",
    }
    cmd = f"{base_prefix}{mode_to_suffix[experiment_mode]} --device {device}"
    return cmd, f"{mode_to_name[experiment_mode]} (seed {seed})"


def generate_all_commands(seed: int = 42) -> List[CommandItem]:
    """
    生成单个 seed 下的全部命令
    """
    experiment_modes = [
        "image_text_meta",
        "image_text",
    ]
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    commands: List[CommandItem] = []
    for idx, mode in enumerate(experiment_modes):
        commands.append(_build_command(seed, devices[idx % len(devices)], mode))
    return commands


def generate_single_variant_commands(experiment_mode: str, start_seed: int, end_seed: int) -> List[CommandItem]:
    """
    生成指定 seed 区间命令。
    experiment_mode:
    - all: 每个 seed 同时生成全部实验组合
    - 其他值: 每个 seed 仅生成指定实验组合
    """
    commands: List[CommandItem] = []
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    all_modes = [
        "image_text_meta",
        "image_text",
    ]

    for seed in range(start_seed, end_seed + 1):
        modes_for_seed = all_modes if experiment_mode == "all" else [experiment_mode]
        for mode in modes_for_seed:
            device = devices[len(commands) % len(devices)]
            commands.append(_build_command(seed, device, mode))

    return commands


def run_all_experiments(args: argparse.Namespace) -> None:
    """
    运行 MLP 实验
    """
    if args.mode == "all":
        all_commands = generate_all_commands(args.seed)
        _run_command_group(all_commands, f"全部实验 (seed {args.seed})")
    elif args.mode == "single":
        if args.start_seed > args.end_seed:
            raise ValueError("start-seed 不能大于 end-seed")

        single_commands = generate_single_variant_commands(
            args.experiment_mode,
            args.start_seed,
            args.end_seed,
        )
        _run_command_group(
            single_commands,
            f"MLP {args.experiment_mode}, seeds {args.start_seed}-{args.end_seed}",
        )

    print("所有实验已完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MLP 模型实验脚本")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    all_parser = subparsers.add_parser("all", help="运行单个 seed 的全部实验")
    all_parser.add_argument("--seed", type=int, default=42, help="随机数种子，默认 42")

    single_parser = subparsers.add_parser("single", help="运行 seed 区间实验")
    single_parser.add_argument(
        "--experiment-mode",
        choices=[
            "all",
            "image_text_meta",
            "image_text",
        ],
        default="all",
        help="实验组合类型，默认 all（每个 seed 运行全部组合）",
    )
    single_parser.add_argument("--start-seed", type=int, default=42, help="起始随机数种子，默认 42")
    single_parser.add_argument("--end-seed", type=int, default=46, help="结束随机数种子，默认 46")

    args = parser.parse_args()
    if not args.mode:
        args.mode = "all"
        args.seed = 42

    print(f"一键运行 MLP 实验脚本 (模式: {args.mode})")
    print("=" * 50)
    run_all_experiments(args)


if __name__ == "__main__":
    main()
