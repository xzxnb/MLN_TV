import pandas as pd
import plotly.express as px
import numpy as np


def create_3d_scatter(file_path, value_range=(4, 5), columns=[0, 1, 4]):
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 检查列数
        if df.shape[1] < 5:
            raise ValueError("Excel文件列数不足5列")

        # 筛选第5列数值在指定范围内的数据
        filtered_df = df[(df.iloc[:, 4] > value_range[0]) &
                         (df.iloc[:, 4] < value_range[1])]

        if len(filtered_df) == 0:
            raise ValueError(f"没有数据在指定范围 {value_range} 内")

        # 选取指定列的数据
        selected_df = filtered_df.iloc[:, columns]

        # 创建3D散点图
        fig = px.scatter_3d(selected_df,
                            x=selected_df.columns[0],
                            y=selected_df.columns[1],
                            z=selected_df.columns[2],
                            title=f'3D Scatter Plot (Range: {value_range[0]}-{value_range[1]})',
                            labels={
                                selected_df.columns[0]: selected_df.columns[0],
                                selected_df.columns[1]: selected_df.columns[1],
                                selected_df.columns[2]: selected_df.columns[2]
                            })

        # 更新布局
        fig.update_layout(
            scene=dict(
                xaxis_title=selected_df.columns[0],
                yaxis_title=selected_df.columns[1],
                zaxis_title=selected_df.columns[2],
                # 添加网格线
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray'),
                zaxis=dict(gridcolor='lightgray')
            ),
            width=1000,
            height=800,
            margin=dict(r=20, b=10, l=10, t=40),
            showlegend=True
        )

        # 更新点的样式
        fig.update_traces(
            marker=dict(
                size=5,
                opacity=0.7,
                colorscale='Viridis'
            )
        )

        # 显示图形
        fig.show()

        # 打印统计信息
        print(f"原始数据行数: {len(df)}")
        print(f"筛选后的数据行数: {len(selected_df)}")
        print("\n数据统计:")
        print(selected_df.describe())

        return selected_df  # 返回筛选后的数据框

    except Exception as e:
        print(f"错误: {str(e)}")
        return None


# 使用函数
file_path = 'domain6.xlsx'  # 替换为你的文件路径
value_range = (0.4, 0.5)  # 指定范围
columns = [0, 1, 4]  # 指定要选取的列

# 调用函数
filtered_data = create_3d_scatter(file_path, value_range, columns)