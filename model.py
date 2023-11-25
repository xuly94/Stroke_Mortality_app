import lightgbm as lgb
import pandas as pd
import streamlit as st
import numpy as np
from prefunc import *
from funs_class_weight import *

from sklearn.utils import resample
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.utils import class_weight

#------------------------------------------------------------------------构建AKD预测模型model.txt
# 读取数据集
df = pd.read_csv('modified_stroke_new_mi_wei12.csv', encoding="utf_8_sig")

X = df[['ACEI_ARB','AKI_and_AKD','NEUT','Diuretic','Scr','Antibiotic','Lipoprotein_a','Na','K','Mg']]
y = df['Death'].astype(int)

X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=23,test_size=0.15)

# 使用您提供的最佳参数和随机种子重新训练模型
params = {
    'max_depth': 5,
    'num_leaves': 14,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'verbose': -1,
    'force_col_wise': True, 
    'class_weight':'balanced'
}
model = lgb.LGBMClassifier(**params)
model.fit(X, y)

# 保存模型到文件
booster = model.booster_
booster.save_model('stroke_mortality_model.txt')

