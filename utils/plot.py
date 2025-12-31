import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def print_metrics_table(epoch,
                        train_cls, train_km,
                        test_cls, test_km):
    """
    每个输入都是 (ACC, NMI, PUR, ARI)
    """
    header = (
        f"\nEpoch {epoch}\n"
        "+---------+-----------+--------+--------+--------+\n"
        "| Split   | Method    |  ACC   |  NMI   |  PUR   |\n"
        "+---------+-----------+--------+--------+--------+"
    )

    row_fmt = "| {:<7} | {:<9} | {:>6.4f} | {:>6.4f} | {:>6.4f} |"

    footer = "+---------+-----------+--------+--------+--------+"

    print(header)
    print(row_fmt.format("Train", "CLS",    *train_cls))
    print(row_fmt.format("Train", "KMeans", *train_km))
    print(footer)
    print(row_fmt.format("Test",  "CLS",    *test_cls))
    print(row_fmt.format("Test",  "KMeans", *test_km))
    print(footer)


# 定义绘制准确率曲线的函数，参数metric_list为各轮训练的准确率列表
def plot_metric(metric_list, dataset_name, name, imgs_path):
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    # 获取总的训练轮数
    epochs = len(metric_list)
    # 设置绘图的大小
    plt.figure(figsize=(12, 6))
    # 绘制准确率曲线，设置线型、点标记、线宽等
    plt.plot(range(1, epochs + 1), metric_list, marker='o', linestyle='-', linewidth=2, markersize=6)

    # 设置x轴和y轴的标签及其字体大小
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'{name}', fontsize=14)
    # 设置图表的标题及其字体大小
    plt.title(f'{dataset_name}[{name}]', fontsize=16)

    # 计算最大准确率及其对应的轮数
    max_metric = max(metric_list)
    max_epoch = metric_list.index(max_metric) + 1
    # 获取最后一轮的准确率
    last_metric = metric_list[-1]

    # 绘制表示最大准确率的水平线
    plt.axhline(y=max_metric, color='gray', linestyle='--', linewidth=0.5)
    # 在图表上标注最大准确率及其对应的轮数
    plt.text(epochs, max_metric, f'Max Metric: {max_metric * 100:.2f}% at Epoch {max_epoch}', ha='right', va='bottom',
             fontsize=10)
    # 在图表上标注最后一轮的准确率
    plt.text(1, 0, f'Last Metric: {last_metric * 100:.2f}%', ha='right', va='bottom', fontsize=10,
             transform=plt.gca().transAxes)

    # 设置x轴的刻度，如果训练轮数多于100轮，减少显示的刻度以避免拥挤
    if epochs > 100:
        step = epochs // 10
        plt.xticks(range(1, epochs + 1, step))
    else:
        plt.xticks(range(1, epochs + 1))

    # 设置y轴的刻度
    plt.yticks(np.arange(min(metric_list), max(metric_list) + 0.05, step=0.05))
    # 设置仅在y轴方向显示网格线
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    # 自动调整子图参数，确保图表的元素不会重叠
    plt.tight_layout()

    # 生成文件名，包含当前时间，以确保文件名唯一
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # TODO 文件名
    filename = f'{imgs_path}/{dataset_name}/{dataset_name}_{name}_ep{epochs}_{current_time}.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # 保存图表为PNG文件，指定分辨率为300dpi
    plt.savefig(filename, dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    # 窗口显示n秒后自动继续执行
    plt.pause(2)
    # 自动关闭窗口
    plt.close()
    # 打印保存的图表文件名
    print(f'Plot saved as {filename}')
