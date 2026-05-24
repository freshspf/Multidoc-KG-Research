import pandas as pd

# 输入文件路径
input_path = r"C:\Users\zya\OneDrive\Desktop\Multidoc-KG\graph_export_onto_ancient.xlsx"
output_path = input_path.replace('.xlsx', '_converted.xlsx')

# 需要修改的列映射：sheet名 -> 列名
replace_map = {
    '关系': 'relation_type',
    '本体层次': 'relation'
}

# 替换字典
replace_dict = {
    'SUB_CLASS_OF': '子类',
    'INSTANCE_OF': '类型'
}

# 读取所有 sheet
xls = pd.ExcelFile(input_path)
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)  # 读为字符串防止类型干扰
        if sheet_name in replace_map:
            col = replace_map[sheet_name]
            if col in df.columns:
                # 对该列进行替换
                df[col] = df[col].replace(replace_dict, regex=False)
        # 写入新文件
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"处理完成，新文件已保存至：{output_path}")