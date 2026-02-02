import subprocess
import threading
import time
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


def run_all_experiments() -> None:
    """
    运行所有late实验
    """
    commands = [
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode mean_pool --use-meta --device cuda:0",
         "Late: late_mean_pool+meta (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode attn_pool --use-meta --device cuda:1",
         "Late: late_attn_pool+meta (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode trm_no_pos --use-meta --device cuda:2",
         "Late: late_trm_no_pos+meta (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode trm_pos --use-meta --device cuda:3",
         "Late: late_trm_pos+meta (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode mean_pool --no-use-meta --device cuda:0",
         "Late: late_mean_pool (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode attn_pool --no-use-meta --device cuda:1",
         "Late: late_attn_pool (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode trm_no_pos --no-use-meta --device cuda:2",
         "Late: late_trm_no_pos (CLIP)"),
        ("conda run -n crowdfunding python src/dl/late/main.py --run-name 42 --seed 42 --baseline-mode trm_pos --no-use-meta --device cuda:3",
         "Late: late_trm_pos (CLIP)")
    ]

    _run_command_group(commands, group_name="")
    print("所有实验已完成！")


if __name__ == "__main__":
    print("一键运行所有late实验脚本")
    print("="*50)
    run_all_experiments()
