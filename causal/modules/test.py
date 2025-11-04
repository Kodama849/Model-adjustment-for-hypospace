import matplotlib.pyplot as plt
import numpy as np

# 数据
metrics = ["Parse Success", "Valid", "Novelty", "Recovery"]

node3_base = [1.000, 0.710, 0.921, 0.645]
node3_student_teacher = [0.992, 0.883, 0.671, 0.579]

node4_base = [0.983, 0.466, 0.777, 0.398]
node4_student_teacher = [0.961, 0.872, 0.547, 0.478]

x = np.arange(len(metrics))
width = 0.18

fig, ax = plt.subplots(figsize=(8,5))

# 绘图
rects1 = ax.bar(x - 1.5*width, node3_base, width, label='Node3 Base', color='#9ecae1')
rects2 = ax.bar(x - 0.5*width, node3_student_teacher, width, label='Node3 Student-Teacher', color='#3182bd')
rects3 = ax.bar(x + 0.5*width, node4_base, width, label='Node4 Base', color='#a1d99b')
rects4 = ax.bar(x + 1.5*width, node4_student_teacher, width, label='Node4 Student-Teacher', color='#31a354')

# 标注
ax.set_ylabel('Mean Value')
ax.set_title('Comparison of Base vs Student-Teacher across Node3 & Node4')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend(frameon=False)

for rect in rects1 + rects2 + rects3 + rects4:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
