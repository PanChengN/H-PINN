import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('relative_errors.csv')

# 将 'network_structure' 转换为有特定顺序的分类类型
network_order = ["3x30", "3x50", "3x70", "3x90", "5x30", "5x50", "5x70", "5x90", "7x30", "7x50", "7x70", "7x90"]
df['network_structure'] = pd.Categorical(df['network_structure'], categories=network_order, ordered=True)

# 定义每个模型类型的标记和线型
markers = {"PINN": "o", "IFNN-PINN": "s", "A-PINN": "D", "H-PINN": "^"}
linestyles = {"PINN": ":", "IFNN-PINN": "--", "A-PINN": "-.", "H-PINN": "-"}

# 设置绘图风格
sns.set(style="whitegrid")

# 创建折线图
plt.figure(figsize=(12, 8), dpi=300)  # 提高分辨率

for model_type, marker in markers.items():
    linestyle = linestyles[model_type]
    subset = df[df['model_type'] == model_type]
    sns.lineplot(
        data=subset,
        x="network_structure",
        y="relative_error",
        marker=marker,
        linestyle=linestyle,
        label=model_type,
        linewidth=2.5
    )

# 设置字体和标签
plt.xlabel("Network Structure", fontsize=14, fontname='Arial')
plt.ylabel("Relative Error", fontsize=14, fontname='Arial')
plt.yscale('log')
plt.xticks(rotation=45, fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')
plt.legend(title='Model Type', fontsize=12, title_fontsize=14, loc='upper right', frameon=False)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 调整图例位置
plt.legend(loc='upper right')

# 设置紧凑布局
plt.tight_layout()

# 保存图表
plt.savefig('Relative_Error_Comparison_for_Different_Network_Structures_and_Models.pdf')

# 显示图表
plt.show()