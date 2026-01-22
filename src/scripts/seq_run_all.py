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
    运行所有seq实验
    """
    # 定义所有实验命令
    all_commands = [
        # 使用meta数据的不同baseline模式
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_set_mean_with_meta --baseline-mode set_mean --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:0", 
         "Seq: CLIP Image+Text with Meta - Set Mean"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_set_attn_with_meta --baseline-mode set_attn --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:1", 
         "Seq: CLIP Image+Text with Meta - Set Attention"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_trm_no_pos_with_meta --baseline-mode trm_no_pos --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:2", 
         "Seq: CLIP Image+Text with Meta - Transformer No Position"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_trm_pos_with_meta --baseline-mode trm_pos --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:3", 
         "Seq: CLIP Image+Text with Meta - Transformer With Position"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_trm_pos_shuffled_with_meta --baseline-mode trm_pos_shuffled --image-embedding-type clip --text-embedding-type clip --use-meta --device cuda:0", 
         "Seq: CLIP Image+Text with Meta - Transformer With Shuffled Position"),
         
        # 不使用meta数据的不同baseline模式
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_set_mean --baseline-mode set_mean --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:0", 
         "Seq: CLIP Image+Text - Set Mean"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_set_attn --baseline-mode set_attn --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:1", 
         "Seq: CLIP Image+Text - Set Attention"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_trm_no_pos --baseline-mode trm_no_pos --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:2", 
         "Seq: CLIP Image+Text - Transformer No Position"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_trm_pos --baseline-mode trm_pos --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:3", 
         "Seq: CLIP Image+Text - Transformer With Position"),
        ("conda run -n crowdfunding python src/dl/seq/main.py --run-name clip_trm_pos_shuffled --baseline-mode trm_pos_shuffled --image-embedding-type clip --text-embedding-type clip --no-use-meta --device cuda:0", 
         "Seq: CLIP Image+Text - Transformer With Shuffled Position")
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
    print("一键运行所有seq实验脚本")
    print("="*50)
    run_all_experiments()