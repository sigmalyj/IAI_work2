import re
import csv
import argparse  # 导入 argparse 模块
from datetime import datetime  # 导入时间模块
import os  # 导入 os 模块，用于处理文件路径

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="解析日志文件并生成结果 CSV 文件")
parser.add_argument("log_file", type=str, help="输入的日志文件路径")
args = parser.parse_args()

# 获取输入文件路径
log_file_path = args.log_file

# 提取日志文件的名称（不含路径和扩展名）
log_file_name = os.path.splitext(os.path.basename(log_file_path))[0]

# 根据当前时间和日志文件名称生成输出文件名
current_time = datetime.now().strftime("%m%d_%H%M")  # 格式化时间为 YYYYMMDD_HHMMSS
output_csv_path = f"csv/{log_file_name}_{current_time}.csv"  # 动态生成文件名

# 初始化存储结果的列表
results = []

# 正则表达式匹配每轮的迭代和评估结果
iteration_pattern = r"------ Start Self-Play Iteration (\d+) ------"
baseline_evaluation_pattern = r"\[AlphaZeroParallel\] Start evaluating with baseline for 20 round"
evaluation_result_pattern = r"\[EVALUATION RESULT\]: win(\d+), lose(\d+), draw(\d+)"

# 读取日志文件
with open(log_file_path, "r", encoding="utf-8") as log_file:
    lines = log_file.readlines()

current_iteration = None
is_baseline_evaluation = False

for line in lines:
    # 匹配迭代轮次
    iteration_match = re.search(iteration_pattern, line)
    if iteration_match:
        current_iteration = int(iteration_match.group(1))
        is_baseline_evaluation = False  # 重置标志
        continue

    # 检测是否进入 baseline 评估部分
    if re.search(baseline_evaluation_pattern, line):
        is_baseline_evaluation = True
        continue

    # 匹配 baseline 评估结果
    if is_baseline_evaluation:
        evaluation_match = re.search(evaluation_result_pattern, line)
        if evaluation_match and current_iteration is not None:
            wins = int(evaluation_match.group(1))
            losses = int(evaluation_match.group(2))
            draws = int(evaluation_match.group(3))
            total_games = wins + losses + draws

            # 计算胜率和不输率
            win_rate = wins / total_games
            no_lose_rate = (wins + draws) / total_games

            # 格式化为两位小数
            win_rate = round(win_rate, 2)
            no_lose_rate = round(no_lose_rate, 2)

            # 保存结果
            results.append([current_iteration, win_rate, no_lose_rate])
            is_baseline_evaluation = False  # 处理完一轮 baseline 评估后重置标志

# 将结果写入 CSV 文件
with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Iteration", "Win Rate", "No Lose Rate"])
    writer.writerows(results)

print(f"结果已保存到 {output_csv_path}")
