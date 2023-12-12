# -*- encoding: utf-8 -*-
"""

@File    :   data_loader.py  
@Modify Time : 2023/12/11 22:46 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : Filter out the data we need from the raw data.

"""
# import package begin
import pandas as pd
# import package end

def data_loader(file_path):
    df = pd.read_excel(file_path, header=0)

    # 选择指定的列
    selected_columns = ['代码', '名称', '昨收', '日期']
    df_selected = df[selected_columns]

    # 重命名列
    new_column_names = ['code', 'name', 'price', 'date']
    df_selected.columns = new_column_names
    return df_selected