#!/usr/bin/env python3
"""
检查指定目录下所有子文件夹中是否包含特定数量的.npy文件，并根据检查结果删除不符合要求的项目
"""

import os
import sys
import pandas as pd
from pathlib import Path
import shutil


# 定义期望的.npy文件列表
EXPECTED_FULL_SET = {
    'cover_image_clip.npy',
    'cover_image_resnet.npy', 
    'cover_image_siglip.npy',
    'image_clip.npy',
    'image_resnet.npy',
    'image_siglip.npy',
    'text_bge.npy',
    'text_clip.npy',
    'text_siglip.npy',
    'title_blurb_bge.npy',
    'title_blurb_clip.npy',
    'title_blurb_siglip.npy'
}

EXPECTED_NO_IMAGE_SET = {  # 没有image_*文件的版本
    'cover_image_clip.npy',
    'cover_image_resnet.npy', 
    'cover_image_siglip.npy',
    'text_bge.npy',
    'text_clip.npy',
    'text_siglip.npy',
    'title_blurb_bge.npy',
    'title_blurb_clip.npy',
    'title_blurb_siglip.npy'
}

EXPECTED_NO_TEXT_SET = {  # 没有text_*文件的版本
    'cover_image_clip.npy',
    'cover_image_resnet.npy', 
    'cover_image_siglip.npy',
    'image_clip.npy',
    'image_resnet.npy',
    'image_siglip.npy',
    'title_blurb_bge.npy',
    'title_blurb_clip.npy',
    'title_blurb_siglip.npy'
}


def count_npy_files_in_subfolders(directory):
    """
    检查目录下所有子文件夹中.npy文件的数量和名称
    
    :param directory: 要检查的目录路径
    :return: 包含子文件夹名称和.npy文件信息的字典
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"错误: 目录 {directory} 不存在")
        return {}
    
    if not directory.is_dir():
        print(f"错误: {directory} 不是一个目录")
        return {}

    results = {}
    
    # 遍历目录下的所有子文件夹
    for subfolder in directory.iterdir():
        if subfolder.is_dir():  # 只检查子文件夹，不深入更深层
            # 获取子文件夹中所有的.npy文件
            npy_files = [f for f in subfolder.iterdir() 
                         if f.is_file() and f.suffix.lower() == '.npy']
            
            actual_files = [f.name for f in npy_files]
            actual_set = set(actual_files)
            
            # 检查是否完全匹配12个文件或两种9个文件的组合
            full_match = EXPECTED_FULL_SET == actual_set
            no_image_match = EXPECTED_NO_IMAGE_SET == actual_set
            no_text_match = EXPECTED_NO_TEXT_SET == actual_set
            
            matches_expected = full_match or no_image_match or no_text_match
            
            results[subfolder.name] = {
                'files': actual_files,
                'count': len(npy_files),
                'expected': matches_expected,
                'full_match': full_match,
                'no_image_match': no_image_match,
                'no_text_match': no_text_match,
                'path': str(subfolder)
            }
    
    return results


def print_results(results):
    """
    打印检查结果
    """
    if not results:
        print("没有找到任何子文件夹")
        return
    
    print("检查子文件夹中.npy文件是否符合要求")
    print("要求: 12个文件（完整版）或9个文件（缺少image_*或text_*）\n")
    
    all_correct = True
    
    for folder_name, info in results.items():
        status = "✓" if info['expected'] else "✗"
        result_text = "符合" if info['expected'] else "不符合"
        
        if info['full_match']:
            file_type = "完整版(12个)"
        elif info['no_image_match']:
            file_type = "无image版(9个)"
        elif info['no_text_match']:
            file_type = "无text版(9个)"
        else:
            file_type = "未知"
        
        if result_text != "符合":
            print(f"{status} [{info['path']}] - {info['count']} 个.npy文件 ({result_text})")
            
            # 确定应该使用哪个期望集进行比较
            actual_set = set(info['files'])
            if actual_set == EXPECTED_FULL_SET or (actual_set <= EXPECTED_FULL_SET and len(actual_set) > len(EXPECTED_NO_IMAGE_SET)):
                expected_set = EXPECTED_FULL_SET
            elif actual_set <= EXPECTED_NO_IMAGE_SET:
                expected_set = EXPECTED_NO_IMAGE_SET
            elif actual_set <= EXPECTED_NO_TEXT_SET:
                expected_set = EXPECTED_NO_TEXT_SET
            else:
                # 如果不确定使用哪个集合，使用包含最多匹配项的那个
                matches_full = len(actual_set.intersection(EXPECTED_FULL_SET))
                matches_no_img = len(actual_set.intersection(EXPECTED_NO_IMAGE_SET))
                matches_no_txt = len(actual_set.intersection(EXPECTED_NO_TEXT_SET))
                
                if matches_full >= matches_no_img and matches_full >= matches_no_txt:
                    expected_set = EXPECTED_FULL_SET
                elif matches_no_img >= matches_no_txt:
                    expected_set = EXPECTED_NO_IMAGE_SET
                else:
                    expected_set = EXPECTED_NO_TEXT_SET
                
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            if missing:
                print(f"    缺少文件: {list(missing)}")
            if extra:
                print(f"    多余文件: {list(extra)}")
        # else:
        #     print(f"{status} [{info['path']}] - {info['count']} 个.npy文件 ({file_type})")
        
        if not info['expected']:
            all_correct = False
    
    print("\n" + "="*60)
    print(f"总计: {len(results)} 个子文件夹")
    
    incorrect_count = sum(1 for info in results.values() if not info['expected'])
    if incorrect_count > 0:
        print(f"不正确的子文件夹数: {incorrect_count}")
        print("这些文件夹中的.npy文件不符合要求（不是完整版或两种精简版之一）")
    else:
        print("所有子文件夹中的.npy文件都符合要求!")
    
    return all_correct


def clean_projects_based_on_embedding_check(csv_path, projects_base_path, trash_folder_path=None, check_only=False):
    """
    根据embedding检查结果清理项目文件夹，将不符合要求的项目移到垃圾文件夹中
    
    :param csv_path: CSV文件路径
    :param projects_base_path: 项目基础路径
    :param trash_folder_path: 垃圾文件夹路径，用于存放要删除的项目文件夹
    :param check_only: 仅检查模式，不执行实际删除操作
    :return: 更新后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建垃圾文件夹路径（如果未提供则默认创建）
    if trash_folder_path is None:
        trash_folder_path = Path(projects_base_path) / "trash"
    else:
        trash_folder_path = Path(trash_folder_path)
    
    if not check_only:
        trash_folder_path.mkdir(exist_ok=True)
    
    # 检查项目目录中的embedding文件
    results = count_npy_files_in_subfolders(projects_base_path)
    
    # 创建备份副本以便后续操作
    original_len = len(df)
    
    # 用于存储要删除的行
    rows_to_remove = []
    
    for idx, row in df.iterrows():
        project_id = str(row.get('project_id', ''))  # 确保project_id为字符串
        
        # 检查项目是否在检查结果中，以及是否符合要求
        if project_id in results:
            if not results[project_id]['expected']:
                # 该项目的embedding文件不符合要求
                print(f"项目 {project_id} 的embedding文件不符合要求，将被加入删除列表")
                rows_to_remove.append(idx)
                
                # 如果不是仅检查模式，则移动项目文件夹到垃圾文件夹
                if not check_only:
                    project_folder = Path(projects_base_path) / project_id
                    if project_folder.exists():
                        trash_project_folder = trash_folder_path / project_id
                        print(f"移动项目文件夹 {project_folder} 到 {trash_project_folder}")
                        
                        # 移动文件夹
                        shutil.move(str(project_folder), str(trash_project_folder))
        else:
            # 如果项目文件夹不存在，也标记此行需要删除
            project_folder = Path(projects_base_path) / project_id
            if not project_folder.exists():
                print(f"项目文件夹不存在，将把对应行加入删除列表 (项目ID: {project_id})")
                rows_to_remove.append(idx)
    
    # 获取剩余的行
    remaining_df = df.drop(rows_to_remove).reset_index(drop=True)
    
    # 保存更新后的CSV文件
    if not check_only:
        csv_path_obj = Path(csv_path)
        stem = csv_path_obj.stem
        suffix = csv_path_obj.suffix
        
        # 生成新文件名，包含剩余行数
        new_stem = f"{stem}_{len(remaining_df)}"
        new_csv_path = csv_path_obj.parent / f"{new_stem}{suffix}"
        remaining_df.to_csv(new_csv_path, index=False)
        print(f"已保存更新后的CSV到 {new_csv_path} (共 {len(remaining_df)} 行)")
    
    print(f"处理完成，原始行数: {original_len}, 保留行数: {len(remaining_df)}, 删除行数: {len(rows_to_remove)}")
    
    return remaining_df


if __name__ == "__main__":
    CSV_PATH = "/home/zlc/crowdfunding/data/metadata/2025.csv"
    PROJECTS_BASE_PATH = "/home/zlc/crowdfunding/data/projects/2025"
    TRASH_FOLDER_PATH = "/home/zlc/crowdfunding/data/projects/tmp"  # 指定垃圾文件夹路径
    CHECK_ONLY = False  # 设为True则只检查不执行删除操作
    
    if CHECK_ONLY:
        # 仅执行检查，不修改任何文件
        results = count_npy_files_in_subfolders(PROJECTS_BASE_PATH)
        all_correct = print_results(results)
        sys.exit(0 if all_correct else 1)
    else:
        # 执行清理操作
        clean_projects_based_on_embedding_check(
            csv_path=CSV_PATH,
            projects_base_path=PROJECTS_BASE_PATH,
            trash_folder_path=TRASH_FOLDER_PATH,
            check_only=CHECK_ONLY
        )