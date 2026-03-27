import statistics

def calculate_stats(numbers):
    if len(numbers) < 2:
        return "提示：至少需要两个数字才能计算标准差。"

    # 计算平均数
    mean_val = statistics.mean(numbers)
    
    # 计算样本标准差
    std_dev_val = statistics.stdev(numbers)
    
    return mean_val, std_dev_val

# 在这里输入你的数字
my_numbers = [7.7076, 6.6516,  5.5440, 8.3123]

# 调用函数并获取结果
result = calculate_stats(my_numbers)

if isinstance(result, tuple):
    mean, std_dev = result
    print(f"你的数据: {my_numbers}")
    print(f"平均数 (Mean): {mean:.2f}")
    print(f"标准差 (Standard Deviation): {std_dev:.2f}")
else:
    print(result)