import subprocess
import sys
import threading
from typing import List


def run_command(cmd: str, experiment_name: str):
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
    运行所有gate实验
    """
    # 定义所有实验命令
    all_commands = [
        # GATE模型的四种baseline-mode
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name late_concat --baseline-mode late_concat --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:0", 
         "GATE: Late Concat with Meta"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name stage1_only --baseline-mode stage1_only --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:1", 
         "GATE: Stage 1 Only with Meta"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name stage2_only --baseline-mode stage2_only --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:2", 
         "GATE: Stage 2 Only with Meta"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name two_stage --baseline-mode two_stage --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:3", 
         "GATE: Two Stage with Meta"),
         
        # 不使用meta数据的四种baseline-mode
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name late_concat --baseline-mode late_concat --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:0", 
         "GATE: Late Concat without Meta"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name stage1_only --baseline-mode stage1_only --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:1", 
         "GATE: Stage 1 Only without Meta"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name stage2_only --baseline-mode stage2_only --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:2", 
         "GATE: Stage 2 Only without Meta"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name two_stage --baseline-mode two_stage --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:3", 
         "GATE: Two Stage without Meta"),
         
        # 三种新的单分支baseline-mode (这些模式本身就已经确定了使用的分支，不需要额外指定use-meta)
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name seq_only --baseline-mode seq_only --image-embedding-type clip --text-embedding-type clip --device cuda:0", 
         "GATE: Seq Only"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name key_only --baseline-mode key_only --image-embedding-type clip --text-embedding-type clip --device cuda:1", 
         "GATE: Key Only"),
        ("conda run -n crowdfunding python src/dl/gate/main.py --run-name meta_only --baseline-mode meta_only --image-embedding-type clip --text-embedding-type clip --device cuda:2", 
         "GATE: Meta Only")
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
    print("一键运行所有gate实验脚本")
    print("="*50)
    run_all_experiments()