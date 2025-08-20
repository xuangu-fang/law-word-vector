#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Analysis - 完整分析运行器

功能：
1. 运行所有topic分析的完整流程
2. 包含数据处理、法律功能分析、法律流程分析
3. 生成完整的分析报告和可视化结果

使用示例：
python src/topic_word_analysis/run_all_analysis.py
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def run_data_processing():
    """运行数据处理"""
    print("🔄 开始数据处理...")
    try:
        from topic_data_processor import TopicDataProcessor
        processor = TopicDataProcessor()
        results = processor.process_all_data()
        print("✅ 数据处理完成")
        return True
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return False

def run_law_function_analysis():
    """运行法律功能分析"""
    print("\n🏛️ 开始法律功能分析...")
    try:
        # 导入并运行法律功能分析
        import subprocess
        result = subprocess.run([
            sys.executable, 
            str(current_dir / "law_function_analysis.py")
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 法律功能分析完成")
            return True
        else:
            print(f"❌ 法律功能分析失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 法律功能分析失败: {e}")
        return False

def run_legal_process_analysis():
    """运行法律流程分析"""
    print("\n⚖️ 开始法律流程分析...")
    try:
        # 导入并运行法律流程分析
        import subprocess
        result = subprocess.run([
            sys.executable, 
            str(current_dir / "legal_process_analysis.py")
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 法律流程分析完成")
            return True
        else:
            print(f"❌ 法律流程分析失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 法律流程分析失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Topic Analysis - 完整分析运行器")
    print("=" * 60)
    
    success_count = 0
    total_steps = 3
    
    # 步骤1: 数据处理
    if run_data_processing():
        success_count += 1
    
    # 步骤2: 法律功能分析
    if run_law_function_analysis():
        success_count += 1
    
    # 步骤3: 法律流程分析
    if run_legal_process_analysis():
        success_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"📊 分析完成总结: {success_count}/{total_steps} 步骤成功")
    
    if success_count == total_steps:
        print("🎉 所有分析已成功完成！")
        print("📂 查看结果: output/topic_analysis/")
    else:
        print("⚠️ 部分分析未成功，请检查错误信息")
    
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)