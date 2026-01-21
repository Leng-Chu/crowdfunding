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
    运行所有实验
    """
    # 定义所有实验命令
    all_commands = [
        # meta + image + text
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --text-embedding-type clip --use-meta --use-image --use-text --device cuda:0", 
         "Multi-modal: CLIP Image+Text with Meta"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name siglip --image-embedding-type siglip --text-embedding-type siglip --use-meta --use-image --use-text --device cuda:1", 
         "Multi-modal: SigLIP Image+Text with Meta"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name bge-clip --image-embedding-type clip --text-embedding-type bge --use-meta --use-image --use-text --device cuda:2", 
         "Multi-modal: CLIP Image + BGE Text with Meta"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name bge-siglip --image-embedding-type siglip --text-embedding-type bge --use-meta --use-image --use-text --device cuda:3", 
         "Multi-modal: SigLIP Image + BGE Text with Meta"),
        
        # image + text
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --text-embedding-type clip --use-image --use-text --device cuda:0", 
         "Image+Text: CLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name siglip --image-embedding-type siglip --text-embedding-type siglip --use-image --use-text --device cuda:1", 
         "Image+Text: SigLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name bge-clip --image-embedding-type clip --text-embedding-type bge --use-image --use-text --device cuda:2", 
         "Image+Text: CLIP Image + BGE Text"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name bge-siglip --image-embedding-type siglip --text-embedding-type bge --use-image --use-text --device cuda:3", 
         "Image+Text: SigLIP Image + BGE Text"),
        
        # meta + image
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --use-meta --use-image --device cuda:0", 
         "Meta+Image: CLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name siglip --image-embedding-type siglip --use-meta --use-image --device cuda:1", 
         "Meta+Image: SigLIP"),
        
        # image
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --use-image --device cuda:2", 
         "Image-only: CLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name siglip --image-embedding-type siglip --use-image --device cuda:3", 
         "Image-only: SigLIP"),
        
        # meta + text
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --text-embedding-type clip --use-meta --use-text --device cuda:0", 
         "Meta+Text: CLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name siglip --text-embedding-type siglip --use-meta --use-text --device cuda:1", 
         "Meta+Text: SigLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name bge --text-embedding-type bge --use-meta --use-text --device cuda:2", 
         "Meta+Text: BGE"),
         
        # text
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --text-embedding-type clip --use-text --device cuda:3", 
         "Text-only: CLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name siglip --text-embedding-type siglip --use-text --device cuda:0", 
         "Text-only: SigLIP"),
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name bge --text-embedding-type bge --use-text --device cuda:1", 
         "Text-only: BGE"),
         
        # meta
        ("conda run -n crowdfunding python src/dl/mlp/main.py --run-name meta --use-meta --device cuda:2", 
         "Meta-only: CUDA")
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
    print("一键运行所有实验脚本")
    print("="*50)
    run_all_experiments()