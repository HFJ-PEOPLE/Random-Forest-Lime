import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件夹路径，包含xlsx文件
folder_path = 'D:\\所有文档\\项目\\CD-数据集hzj\\12色\\测试集\\Ag\\0'

# 获取文件夹中所有xlsx文件的列表
xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 遍历每个xlsx文件
for file_name in xlsx_files:
    # 构造完整路径
    file_path = os.path.join(folder_path, file_name)

    # 读取xlsx文件到DataFrame
    df = pd.read_excel(file_path)

    # 假设x轴和y轴列名已知，可以根据实际情况修改
    x_data = df['x轴列名']
    y_data = df['y轴列名']

    # 创建新的图表
    plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.title(f'图表标题 - {file_name}')  # 使用文件名作为标题的一部分

    # 保存为图片文件（可以选择不同的格式，如png, jpg等）
    output_file_name = os.path.splitext(file_name)[0] + '.jpg'
    output_file_path = os.path.join(folder_path, output_file_name)
    plt.savefig(output_file_path)

    # 可以选择在每次循环结束时显示图表（可选）
    # plt.show()

    # 关闭当前图表，避免内存泄漏
    plt.close()

print("所有xlsx文件转换为图片完成！")
