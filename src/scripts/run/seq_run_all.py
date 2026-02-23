"""
SEQ模型实验运行脚本

使用样例:

# 1. 运行所有baseline模式，使用指定种子（默认为42）
python src/scripts/run/seq_run_all.py all --seed 42 --use-meta --use-attr

# 2. 运行所有baseline模式，使用种子范围
python src/scripts/run/seq_run_all.py all --start-seed 42 --end-seed 46 --use-meta --use-attr

# 3. 运行特定baseline模式，使用不同种子范围
python src/scripts/run/seq_run_all.py single --baseline-mode trm_pos --start-seed 42 --end-seed 46 --no-use-meta --use-attr
"""

import argparse
import subprocess
import threading
import time
from typing import List, Tuple


CommandItem = Tuple[str, str]
DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
BASELINE_MODES = ["set_mean", "set_attn", "trm_no_pos", "trm_pos", "trm_pos_shuffled"]


def run_command(cmd: List[str], experiment_name: str) -> None:
    """运行单个命令。"""
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

    threads: List[threading.Thread] = []
    for cmd_str, experiment_name in commands:
        cmd = cmd_str.split()
        print(f"启动实验: {experiment_name}")
        thread = threading.Thread(target=run_command, args=(cmd, experiment_name))
        threads.append(thread)
        thread.start()

        # 错峰启动，降低瞬时资源冲突
        time.sleep(2)

    for thread in threads:
        thread.join()


def _meta_flag(use_meta: bool) -> str:
    return "--use-meta" if bool(use_meta) else "--no-use-meta"


def _attr_flag(use_attr: bool) -> str:
    return "--use-attr" if bool(use_attr) else "--no-use-attr"


def _meta_label(use_meta: bool) -> str:
    return "meta" if bool(use_meta) else "no_meta"


def _attr_label(use_attr: bool) -> str:
    return "attr" if bool(use_attr) else "no_attr"


def _build_command(seed: int, baseline_mode: str, device: str, use_meta: bool, use_attr: bool) -> CommandItem:
    cmd = (
        "conda run -n crowdfunding python src/dl/seq/main.py "
        f"--run-name {seed} --seed {seed} --baseline-mode {baseline_mode} "
        f"{_meta_flag(use_meta)} {_attr_flag(use_attr)} --device {device}"
    )
    exp_name = f"Seq: {baseline_mode}+{_meta_label(use_meta)}+{_attr_label(use_attr)} (seed {seed})"
    return cmd, exp_name


def _resolve_all_mode_seed_range(args: argparse.Namespace) -> Tuple[int, int]:
    """解析all模式种子区间；未指定区间时退化为单seed。"""
    start_seed = getattr(args, "start_seed", None)
    end_seed = getattr(args, "end_seed", None)

    if (start_seed is None) != (end_seed is None):
        raise ValueError("all 模式下 start-seed 和 end-seed 必须同时提供")

    if start_seed is None:
        single_seed = int(getattr(args, "seed", 42))
        start_seed = single_seed
        end_seed = single_seed

    if start_seed > end_seed:
        raise ValueError("start-seed 不能大于 end-seed")

    return int(start_seed), int(end_seed)


def _format_seed_range_label(start_seed: int, end_seed: int) -> str:
    return f"seed {start_seed}" if start_seed == end_seed else f"seeds {start_seed}-{end_seed}"


def generate_all_commands(
    start_seed: int = 42,
    end_seed: int = 42,
    use_meta: bool = True,
    use_attr: bool = True,
) -> List[CommandItem]:
    """生成指定seed区间下的全部baseline命令。"""
    commands: List[CommandItem] = []
    for seed in range(start_seed, end_seed + 1):
        for idx, baseline_mode in enumerate(BASELINE_MODES):
            device = DEVICES[idx % len(DEVICES)]
            commands.append(_build_command(seed, baseline_mode, device, use_meta, use_attr))
    return commands


def generate_single_baseline_commands(
    baseline_mode: str,
    start_seed: int,
    end_seed: int,
    use_meta: bool = True,
    use_attr: bool = True,
) -> List[CommandItem]:
    """生成某个baseline在指定seed区间的命令。"""
    commands: List[CommandItem] = []
    for seed in range(start_seed, end_seed + 1):
        device = DEVICES[len(commands) % len(DEVICES)]
        commands.append(_build_command(seed, baseline_mode, device, use_meta, use_attr))
    return commands


def run_all_experiments(args: argparse.Namespace) -> None:
    """运行SEQ实验。"""
    if args.mode == "all":
        start_seed, end_seed = _resolve_all_mode_seed_range(args)
        all_commands = generate_all_commands(
            start_seed,
            end_seed,
            use_meta=bool(args.use_meta),
            use_attr=bool(args.use_attr),
        )
        _run_command_group(
            all_commands,
            (
                f"所有baseline ({_format_seed_range_label(start_seed, end_seed)}, "
                f"{_meta_label(args.use_meta)}, {_attr_label(args.use_attr)})"
            ),
        )
    elif args.mode == "single":
        if args.start_seed > args.end_seed:
            raise ValueError("start-seed 不能大于 end-seed")

        single_commands = generate_single_baseline_commands(
            args.baseline_mode,
            args.start_seed,
            args.end_seed,
            use_meta=bool(args.use_meta),
            use_attr=bool(args.use_attr),
        )
        _run_command_group(
            single_commands,
            (
                f"Baseline {args.baseline_mode}, seeds {args.start_seed}-{args.end_seed} "
                f"({_meta_label(args.use_meta)}, {_attr_label(args.use_attr)})"
            ),
        )

    print("所有实验已完成！")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行SEQ模型实验脚本")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    all_parser = subparsers.add_parser("all", help="运行所有baseline模式（支持seed区间）")
    all_parser.add_argument("--seed", type=int, default=42, help="随机数种子，默认42；未指定区间时生效")
    all_parser.add_argument("--start-seed", type=int, default=None, help="起始随机数种子（可选，需与--end-seed同时提供）")
    all_parser.add_argument("--end-seed", type=int, default=None, help="结束随机数种子（可选，需与--start-seed同时提供）")
    all_parser.add_argument(
        "--use-meta",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="是否启用meta分支，默认启用",
    )
    all_parser.add_argument(
        "--use-attr",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="是否启用属性注入，默认启用",
    )

    single_parser = subparsers.add_parser("single", help="运行单一baseline模式")
    single_parser.add_argument(
        "--baseline-mode",
        required=True,
        choices=BASELINE_MODES,
        help="选择baseline模式",
    )
    single_parser.add_argument("--start-seed", type=int, default=42, help="起始随机数种子，默认42")
    single_parser.add_argument("--end-seed", type=int, default=46, help="结束随机数种子，默认46")
    single_parser.add_argument(
        "--use-meta",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="是否启用meta分支，默认启用",
    )
    single_parser.add_argument(
        "--use-attr",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="是否启用属性注入，默认启用",
    )

    args = parser.parse_args()

    if not args.mode:
        args.mode = "all"
        args.seed = 42
        args.start_seed = None
        args.end_seed = None
        args.use_meta = True
        args.use_attr = True

    print(f"一键运行SEQ实验脚本 (模式: {args.mode})")
    print("=" * 50)
    run_all_experiments(args)


if __name__ == "__main__":
    main()
