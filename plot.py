import csv
import matplotlib.pyplot as plt
import argparse  # 导入 argparse 模块
import os  # 导入 os 模块，用于处理文件路径

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="绘制训练过程中胜率和不输率的折线图")
parser.add_argument("csv_file", type=str, help="输入的 CSV 文件路径")
args = parser.parse_args()

# 获取 CSV 文件路径
csv_file_path = args.csv_file

# 提取 CSV 文件的名称（不含路径和扩展名）
csv_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]

# 初始化数据存储
iterations = []
win_rates = []
no_lose_rates = []

# 读取 CSV 文件
with open(csv_file_path, "r", encoding="utf-8") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # 跳过表头
    for row in reader:
        iterations.append(int(row[0]))  # 迭代轮次
        win_rates.append(float(row[1]))  # 胜率
        no_lose_rates.append(float(row[2]))  # 不输率

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(iterations, win_rates, label="Win Rate", marker="o", color="blue")
plt.plot(iterations, no_lose_rates, label="No Lose Rate", marker="x", color="green")

# 图表标题和标签
plt.title("Win Rate and No Lose Rate vs Iterations", fontsize=16)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Rate", fontsize=14)
plt.ylim(0, 1)  # 胜率和不输率范围在 [0, 1]
plt.grid(True)
plt.legend(fontsize=12)

# 动态生成图表文件名
output_image_path = f"plots/{csv_file_name}_curve.png"

# 保存图表或显示
plt.savefig(output_image_path)  # 保存为图片文件
plt.show()  # 显示图表

print(f"图表已保存为 {output_image_path}")