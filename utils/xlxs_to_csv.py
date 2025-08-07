import pandas as pd

# 讀取 Excel 檔案
excel_file = '../AI_output2.xlsx'
df = pd.read_excel(excel_file)
df = df.drop(columns=['Type','IndexName'])

# 將資料寫入 CSV 檔案
csv_file = '../AI_output2.csv'
df.to_csv(csv_file, index=False)

print('Excel 轉換成 CSV 完成')x