
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Analysis - 数据处理器

功能：
1. 读取Excel文件中的专家标注数据
2. 将sheet内容转化为结构化的JSON文件
3. 支持多种topic类型（法律功能、法律流程等）
4. 自动创建输出目录结构

输出目录：output/topic_analysis/
"""

import pandas as pd
import json
import os
from pathlib import Path

class TopicDataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        self.base_dir = Path(__file__).parent.parent.parent
        self.output_dir = self.base_dir / "output" / "topic_analysis"
        
    def process_excel_to_json(self, excel_path, json_path):
        """
        读取Excel文件并转换为结构化JSON
        
        Args:
            excel_path (str): Excel文件路径
            json_path (str): JSON输出文件路径
        """
        xls = pd.ExcelFile(excel_path)
        
        output_data = {}

        sheet_names = [
            "法制era1", "法制era2", "法制era3",
            "法治era1", "法治era2", "法治era3"
        ]

        for sheet_name in sheet_names:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Extract the main key ("法制" or "法治") and the era
                if "法制" in sheet_name:
                    main_key = "法制"
                    era = sheet_name.replace("法制", "")
                elif "法治" in sheet_name:
                    main_key = "法治"
                    era = sheet_name.replace("法治", "")
                else:
                    continue

                if main_key not in output_data:
                    output_data[main_key] = {}
                
                if era not in output_data[main_key]:
                    output_data[main_key][era] = {}

                # Group words by 'class-label'
                grouped = df.groupby('class-label')['word'].apply(list)
                
                for class_label, words in grouped.items():
                    output_data[main_key][era][class_label] = words

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Save the dictionary to a JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

    def process_law_function_data(self):
        """处理法律功能数据（发展、秩序、规范、权力限制）"""
        excel_file = self.base_dir / 'expert_labeled_data' / 'topic_发展、秩序、规范、权力限制.xlsx'
        json_file = self.output_dir / 'law-function' / 'topic_word_sets_law_function.json'
        
        self.process_excel_to_json(str(excel_file), str(json_file))
        print(f"✅ 法律功能数据处理完成: {json_file}")
        return json_file
        
    def process_legal_process_data(self):
        """处理法律流程数据（立法、司法、执法、守法）"""
        excel_file = self.base_dir / 'expert_labeled_data' / 'topic_立法、司法、执法、守法.xlsx'
        json_file = self.output_dir / 'legal_process' / 'topic_word_sets_legal_process.json'
        
        self.process_excel_to_json(str(excel_file), str(json_file))
        print(f"✅ 法律流程数据处理完成: {json_file}")
        return json_file
        
    def process_all_data(self):
        """处理所有topic数据"""
        print("Topic Analysis - 数据处理器")
        print("=" * 60)
        
        results = {}
        results['law_function'] = self.process_law_function_data()
        results['legal_process'] = self.process_legal_process_data()
        
        print(f"\n🎉 所有数据处理完成！")
        return results

def main():
    """主函数"""
    processor = TopicDataProcessor()
    processor.process_all_data()

if __name__ == "__main__":
    main()
