'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

#x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# VGG_supervised = np.array([2.9749694, 3.9357018, 4.7440844, 6.482254, 8.720203, 13.687582])
# VGG_unsupervised = np.array([2.1044724, 2.9757383, 3.7754183, 5.686206, 8.367847, 14.144531])
# ourNetwork = np.array([2.0205495, 2.6509762, 3.1876223, 4.380781, 6.004548, 9.9298])

#JointECW = np.array([0.4478, 0.4627, 0.4406, 0.4278, 0.3918, 0.3681, 0.3740, 0.3730, 0.3867, 0.3715])
JointECW = np.array([0.4267575247458083, 0.43887570458642894, 0.4187372062185938, 0.4063883671215359, 0.37194957835094433, 0.34876868498601105, 0.3538856069180201, 0.35245544561900316, 0.36576505088750444, 0.3507592817749894])
JointEC = np.array([0.3140, 0.3872, 0.3701, 0.3390, 0.3272, 0.3310, 0.3145, 0.3136, 0.3129, 0.3152])

# JointECW = np.array([0.4614, 0.4617, 0.4647, 0.4649, 0.4637, 0.4686, 0.4551, 0.4659, 0.4627, 0.4665, 0.4529])
# #JointECW = np.array([0.4395361511381575, 0.4406824141935681, 0.44301458714515857, 0.4432425551167616, 0.4421565351289889, 0.44414470954082635, 0.432722199302453, 0.442511258402428, 0.43832068230776006, 0.44273597507524487, 0.42926890927527644])
# JointECW = JointECW[::-1]
# JointEC = np.array([0.3517, 0.3625, 0.3530, 0.3758, 0.3678, 0.3662, 0.3845, 0.3695, 0.3769, 0.3591, 0.3630])
# JointEC = JointEC[::-1]

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(10, 5))
#plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
#ax.spines['top'].set_visible(False)  # 去掉上边框
#ax.spines['right'].set_visible(False)  # 去掉右边框


plt.plot(x, JointECW, marker='o', color="blue", label="Joint-ECW", linewidth=1.5)
plt.plot(x, JointEC, marker='o', color="green", label="Joint-EC", linewidth=1.5)
#plt.plot(x, JointECW, marker='o', color="red", label="ShuffleNet-style Network", linewidth=1.5)

#group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
#group_labels = ['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
group_labels = ['1','2','3','4','5', '6', '7', '8', '9', '10']
plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Chunk Size", fontsize=13, fontweight='bold')
plt.ylabel("F1 Score", fontsize=13, fontweight='bold')
plt.xlim(0.9, 10.1)  # 设置x轴的范围
plt.ylim(0.25, 0.50)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

plt.savefig('./linechart.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()