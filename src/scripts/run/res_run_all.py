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
    运行所有res实验
    """
    # 定义所有实验命令
    all_commands = [
        # RES模型的不同配置
        ("conda run -n crowdfunding python src/dl/res/main.py --run-name mlp_baseline --baseline-mode mlp --image-embedding-type clip --text-embedding-type clip --device cuda:2", 
         "RES: MLP Baseline"),
        ("conda run -n crowdfunding python src/dl/res/main.py --run-name res_model --baseline-mode res --image-embedding-type clip --text-embedding-type clip --device cuda:3", 
         "RES: RES Model")
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
    print("一键运行所有res实验脚本")
    print("="*50)
    run_all_experiments()