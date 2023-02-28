import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.6)
import os


train_df = pd.read_csv('train.csv')
ss = pd.read_csv('sample_submission.csv')
train_bb = pd.read_csv('train_bounding_boxes.csv')
test_df = pd.read_csv('test.csv')

# print(f"train: {train_df.shape}")
# print(f"train_bb: {train_bb.shape}")
# print(f"Sample Submissoion: {ss.shape}")
# print(f"test: {test_df.shape}")


# print(f"{train_df.head(5)}")
# print(train_bb.head(8))
# print(f"ss:-{ss.head(5)}")
# print(test_df.head(3))

# print(train_df['patient_overall'])


print(train_df.info())
print(train_df.describe())

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
axis_1 = sns.countplot(data=train_df, x='patient_overall')
# values = axis_1.containers

for contrainer in axis_1.containers:
    # print(f"Containers1:- {contrainer}")
    axis_1.bar_label(contrainer)
plt.title("Fracture by patient")
plt.ylim([0, 1500])

train_melt = pd.melt(train_df, id_vars=['StudyInstanceUID'], value_vars=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
                     var_name="Vertebrae", value_name="Fractured")
# print(f"{train_melt}")

plt.subplot(1, 2, 2)
axis_2 = sns.countplot(data=train_melt, x='Vertebrae', hue="Fractured")
for contrainer in axis_2.containers:
    # print(f"Containers2:- {contrainer}")
    axis_2.bar_label(contrainer)
plt.title('Fractures by vertebrae')
plt.ylim([0, 2600])
# plt.show()

plt.figure(figsize=(10,5))
ax = sns.countplot(x = train_df[['C1','C2','C3','C4','C5','C6','C7']].sum(axis=1))
for container in ax.containers:
    ax.bar_label(container)
plt.title('Number of fracture by patient')
plt.xlabel('Number of fractures')
plt.ylim([0,1500])

# print(f"print the values:- {ax}")

plt.figure(figsize=(6,5))
sns.heatmap(train_df[['C1','C2','C3','C4','C5','C6','C7']].corr(),annot=True,cmap='bwr',vmin=-1,vmax=1)
plt.title('Correlarions')
plt.show()








for i in range(7):
    val=(train_df['StudyInstanceUID'].map(lambda x : x.split('.')[i]).unique())
    # print(val[0][1])
