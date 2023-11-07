import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from pdpbox import pdp, get_dataset, info_plots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
import shap


data = pd.read_csv('D:\Pycharm\Pycharm项目文件\IML\data\data_processed.csv')
X = data.drop(['KT2','KL2'], axis=1)
y = data['KT2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = X.columns

model = MLPRegressor(hidden_layer_sizes=(60), activation='relu', solver='adam', batch_size='auto',
                     learning_rate='constant', learning_rate_init=0.5, power_t=0.5,
                     alpha=0.001, random_state=42, max_iter=300)

model.fit(X_train, y_train)


# test
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print("MAE: {:.2f}".format(test_mae))
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE: {:.2f}".format(test_rmse))
test_r2 = r2_score(y_test, y_pred)
print("R2: {:.2f}".format(test_r2))

# train
y_pred_train = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
print("MAE: {:.2f}".format(train_mae))
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
print("RMSE: {:.2f}".format(train_rmse))
train_r2 = r2_score(y_train, y_pred_train)
print("R2: {:.2f}".format(train_r2))

# PDP
from pdpbox import pdp, get_dataset, info_plots

for i, feature in enumerate(feature_names):
    pdp_goals = pdp.pdp_isolate(model=model, dataset=X, model_features=feature_names, feature=feature)
    pdp.pdp_plot(pdp_goals, feature, plot_lines=True, frac_to_plot=0.5)

plt.tight_layout()
plt.show()

# SHAP
background = shap.sample(X_train, 100)
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X)

shap.initjs()
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)

shap.summary_plot(shap_values, X, feature_names=feature_names)

shap.initjs()
shap.plots.waterfall(shap.Explanation(values=shap_values[29],
                                      base_values=explainer.expected_value,
                                      feature_names=feature_names,
                                      data=X.iloc[29]
                                      ))


shap.dependence_plot("tw", shap_values.values, x_data, interaction_index='ta')
shap.dependence_plot("b", shap_values.values, x_data, interaction_index='a')
shap.dependence_plot("Ld", shap_values.values, x_data, interaction_index='Ec')
shap.dependence_plot("tc", shap_values.values, x_data, interaction_index='b')
shap.dependence_plot("d", shap_values.values, x_data, interaction_index='Sd')
