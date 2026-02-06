"""
MDL 模型实验运行脚本。

注意：该 baseline 的代码目录为 `src/dl/mdl`；文档、日志与产物命名统一使用 `mdl`。
"""

import subprocess
import threading
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


def run_all_experiments():
    """
    运行所有实验
    """
    # 定义所有实验命令
    seed = 42
    all_commands = [
        # meta + image + text
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed {seed} --image-embedding-type clip --text-embedding-type clip --use-meta --use-image --use-text --device cuda:0",
            f"MDL: clip image+text+meta (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name siglip --seed {seed} --image-embedding-type siglip --text-embedding-type siglip --use-meta --use-image --use-text --device cuda:1",
            f"MDL: siglip image+text+meta (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name bge-clip --seed {seed} --image-embedding-type clip --text-embedding-type bge --use-meta --use-image --use-text --device cuda:2",
            f"MDL: clip image + bge text + meta (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name bge-siglip --seed {seed} --image-embedding-type siglip --text-embedding-type bge --use-meta --use-image --use-text --device cuda:3",
            f"MDL: siglip image + bge text + meta (seed {seed})",
        ),
        
        # image + text
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed {seed} --image-embedding-type clip --text-embedding-type clip --use-image --use-text --device cuda:0",
            f"MDL: clip image+text (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name siglip --seed {seed} --image-embedding-type siglip --text-embedding-type siglip --use-image --use-text --device cuda:1",
            f"MDL: siglip image+text (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name bge-clip --seed {seed} --image-embedding-type clip --text-embedding-type bge --use-image --use-text --device cuda:2",
            f"MDL: clip image + bge text (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name bge-siglip --seed {seed} --image-embedding-type siglip --text-embedding-type bge --use-image --use-text --device cuda:3",
            f"MDL: siglip image + bge text (seed {seed})",
        ),
        
        # meta + image
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed {seed} --image-embedding-type clip --use-meta --use-image --device cuda:0",
            f"MDL: clip meta+image (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name siglip --seed {seed} --image-embedding-type siglip --use-meta --use-image --device cuda:1",
            f"MDL: siglip meta+image (seed {seed})",
        ),
        
        # image
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed {seed} --image-embedding-type clip --use-image --device cuda:2",
            f"MDL: clip image-only (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name siglip --seed {seed} --image-embedding-type siglip --use-image --device cuda:3",
            f"MDL: siglip image-only (seed {seed})",
        ),
        
        # meta + text
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed {seed} --text-embedding-type clip --use-meta --use-text --device cuda:0",
            f"MDL: clip meta+text (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name siglip --seed {seed} --text-embedding-type siglip --use-meta --use-text --device cuda:1",
            f"MDL: siglip meta+text (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name bge --seed {seed} --text-embedding-type bge --use-meta --use-text --device cuda:2",
            f"MDL: bge meta+text (seed {seed})",
        ),
         
        # text
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed {seed} --text-embedding-type clip --use-text --device cuda:3",
            f"MDL: clip text-only (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name siglip --seed {seed} --text-embedding-type siglip --use-text --device cuda:0",
            f"MDL: siglip text-only (seed {seed})",
        ),
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name bge --seed {seed} --text-embedding-type bge --use-text --device cuda:1",
            f"MDL: bge text-only (seed {seed})",
        ),
         
        # meta
        (
            f"conda run -n crowdfunding python src/dl/mdl/main.py --run-name meta --seed {seed} --use-meta --device cuda:2",
            f"MDL: meta-only (seed {seed})",
        ),
    ]
    
    threads = []
    
    for cmd_str, experiment_name in all_commands:
        cmd = cmd_str.split()
        print(f"启动实验: {experiment_name}")
        thread = threading.Thread(target=run_command, args=(cmd, experiment_name))
        threads.append(thread)
        thread.start()
        
        # 等待一小段时间再启动下一个实验，避免资源冲突
        import time
        time.sleep(2)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("所有实验已完成！")


if __name__ == "__main__":
    print("一键运行所有 MDL 实验脚本")
    print("="*50)
    run_all_experiments()
