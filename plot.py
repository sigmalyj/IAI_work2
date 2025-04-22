import re
import csv
import argparse
from datetime import datetime
import os
import matplotlib.pyplot as plt

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="解析日志文件并绘制训练过程中胜率和不输率的折线图")
parser.add_argument("log_file", type=str, help="输入的日志文件路径")
args = parser.parse_args()

# 获取输入文件路径
log_file_path = args.log_file

# 提取日志文件的名称（不含路径和扩展名）
log_file_name = os.path.splitext(os.path.basename(log_file_path))[0]

# 根据当前时间和日志文件名称生成输出文件名
current_time = datetime.now().strftime("%m%d_%H%M")
output_csv_path = f"csv/{log_file_name}_{current_time}.csv"

# 初始化存储结果的列表
results = []

# 正则表达式匹配每轮的迭代和训练结果
iteration_pattern = r"------ Start Self-Play Iteration (\d+) ------"
training_result_pattern = r"\[AlphaZeroParallel\] Finished .* Win: (\d+), Draw: (\d+), Lose: (\d+)"

# 读取日志文件
with open(log_file_path, "r", encoding="utf-8") as log_file:
    lines = log_file.readlines()

current_iteration = None

for line in lines:
    # 匹配迭代轮次
    iteration_match = re.search(iteration_pattern, line)
    if iteration_match:
        current_iteration = int(iteration_match.group(1))
        continue

    # 匹配训练结果
    training_result_match = re.search(training_result_pattern, line)
    if training_result_match and current_iteration is not None:
        wins = int(training_result_match.group(1))
        draws = int(training_result_match.group(2))
        losses = int(training_result_match.group(3))
        total_games = wins + draws + losses

        # 计算胜率和不输率
        win_rate = wins / total_games
        no_lose_rate = (wins + draws) / total_games

        # 格式化为两位小数
        win_rate = round(win_rate, 2)
        no_lose_rate = round(no_lose_rate, 2)

        # 保存结果
        results.append([current_iteration, win_rate, no_lose_rate])

# 将结果写入 CSV 文件
os.makedirs("csv", exist_ok=True)
with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Iteration", "Win Rate", "No Lose Rate"])
    writer.writerows(results)

print(f"结果已保存到 {output_csv_path}")

# 绘制折线图
iterations = [row[0] for row in results]
win_rates = [row[1] for row in results]
no_lose_rates = [row[2] for row in results]

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
os.makedirs("plots", exist_ok=True)
output_image_path = f"plots/{log_file_name}_curve_{current_time}.png"

# 保存图表或显示
plt.savefig(output_image_path)  # 保存为图片文件
plt.show()  # 显示图表

print(f"图表已保存为 {output_image_path}")