
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from torch.nn import init
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

path='D:\Pycharm\Pycharm项目文件\IML\data\data_processed.csv'
data_df = pd.read_csv(path)

x_data = data_df[[i for i in data_df.columns if i not in ['KT2','KL2']]]
y_data = data_df.KT2

feature_names = [i for i in data_df.columns if i not in ['KT2','KL2']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

model = xgb.XGBRegressor(eval_metric='rmse',
                        max_depth=10,
                        learning_rate=0.05,
                        n_estimators=200,
                        objective='reg:squarederror')
model.fit(x_train,y_train, eval_set=[(x_train, y_train), (x_test, y_test)],  verbose=10 )

# test
ansxgb = model.predict(x_test)
mse = mean_squared_error(y_test,ansxgb)
rmse = np.sqrt(mean_squared_error(y_test,ansxgb))
r2 = r2_score(y_test,ansxgb)
mae = mean_absolute_error(y_test,ansxgb)
print('mse:', mse)
print('rmse:', rmse)
print(f"R2 score: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# train
ansxgb_train = model.predict(x_train)
train_mse = mean_squared_error(y_train,ansxgb_train)
train_rmse = np.sqrt(mean_squared_error(y_train,ansxgb_train))
train_r2 = r2_score(y_train,ansxgb_train)
train_mae = mean_absolute_error(y_train,ansxgb_train)
print('mse:', train_mse)
print('rmse:', train_rmse)
print(f"R2_train score: {train_r2:.2f}")
print(f"MAE_train: {train_mae:.2f}")


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))
xgb.plot_importance(model, importance_type='gain', ax=ax)
plt.show()


# SHAP
import shap
explainer = shap.Explainer(model)
type(explainer)

shap_values = explainer.shap_values(x_data)

shap.initjs()
shap_values = explainer(x_data)
shap.plots.waterfall(shap_values[5])

shap_values = explainer(x_data)
shap.plots.beeswarm(shap_values)

shap.summary_plot(shap_values, x_train, plot_type="bar")

shap.dependence_plot("tw", shap_values.values, x_data, interaction_index='ta')
shap.dependence_plot("b", shap_values.values, x_data, interaction_index='a')
shap.dependence_plot("Ld", shap_values.values, x_data, interaction_index='Ec')
shap.dependence_plot("tc", shap_values.values, x_data, interaction_index='b')
shap.dependence_plot("d", shap_values.values, x_data, interaction_index='Sd')

# PDP
from pdpbox import pdp, get_dataset, info_plots
import matplotlib.pyplot as plt
import pandas as pd
import torch
for i, feature in enumerate(feature_names):
    pdp_goals = pdp.pdp_isolate(model=model, dataset=x_data, model_features=feature_names, feature=feature)
    pdp.pdp_plot(pdp_goals, feature, plot_lines=True, frac_to_plot=0.5)
plt.tight_layout()
plt.show()