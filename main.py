# -*- encoding: utf-8 -*-
"""

@File    :   main.py
@Modify Time : 2023/12/11 19:56 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : Main and PLT show

"""

# import package begin
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import data_loader
from train import predict
# import package end

# file path begin
excel_file_path = './datasets/ChinaUnicom2023.xlsx'
model_path = './model/stock_model.pth'
# file path end

df_selected = data_loader(excel_file_path)

dates = df_selected['date']
prices = df_selected['price']
# 找到最高点和最低点的索引
max_price_index = prices.idxmax()
min_price_index = prices.idxmin()

# 添加最高点和最低点的标注
plt.annotate(f'Highest: {prices[max_price_index]} ({dates[max_price_index].strftime("%Y-%m-%d")})',
             xy=(dates[max_price_index], prices[max_price_index]),
             xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
plt.annotate(f'Lowest: {prices[min_price_index]} ({dates[min_price_index].strftime("%Y-%m-%d")})',
             xy=(dates[min_price_index], prices[min_price_index]),
             xytext=(-30, -40), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

# 获取最新日期和价格
latest_date = dates.iloc[-1]
latest_price = prices.iloc[-1]

# 添加最新日期的标注
plt.annotate(f'Latest: {latest_price} ({latest_date.strftime("%Y-%m-%d")})',
             xy=(latest_date, latest_price),
             xytext=(-50, -20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))



# 调整图表的布局
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

# 绘制股票价格趋势图
#plt.figure(figsize=(300, 500))
plt.plot(dates, prices)
plt.xlabel('Time')
plt.ylabel('Price')
#plt.title('Stock Price Trend')
#plt.savefig('./stock_price_trend.png')
# 显示图表
#plt.show()



result = predict(model_path)[:20].detach().numpy()
print(result)

# 生成预测结果对应的日期序列
last_date = pd.to_datetime(dates.iloc[-1])  # 获取最后一个日期
predicted_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=len(result), freq='D')  # 生成日期序列
print(predicted_dates)
plt.plot(predicted_dates, result, label='Predicted Price', color='red')
plt.title('Stock Price Predict Trend')


# 添加预测价格的标注
# 在最开始、中间和结束三个点进行标注
points_to_label = [0, len(predicted_dates) // 2, len(predicted_dates) - 1]
for i in points_to_label:
    plt.text(predicted_dates[i], result[i], f'{float(result[i]):.2f}', color='red', va='center')


plt.legend()

plt.show()

plt.savefig("stock_price_predict_trend.png")