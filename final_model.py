####DATA 543 Semester Project#######

###########################################################
#Cell 1
###########################################################
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################
#Cell 2
###########################################################

###################################################################
# Import Data from the Github Repo and Clean + Format
###################################################################

base_path = 'Local Data/'
start_date = '2010-01-09'
end_date = '2025-12-31'

df_localtemps = pd.read_csv(base_path +'local_weather.csv') #daily temp observations, precipitation at Shasta Dam
df_localtemps = df_localtemps.drop(columns=['STATION', 'NAME', 'DAPR', 'MDPR', 'SNOW', 'SNWD'])
df_localtemps['DATE'] = pd.to_datetime(df_localtemps['DATE'], format='mixed')
df_localtemps.columns = ['Date', 'Precipitation', 'Max Temp (F)', 'Min Temp (F)', 'Observed Temp (F)']
df_localtemps = df_localtemps.set_index('Date').sort_index()

df_hdd_cdd = pd.read_csv(base_path + 'hdd_cdd.csv') #monthly hdd and cdd
df_hdd_cdd['Date'] = pd.to_datetime(df_hdd_cdd['Date'], format='%Y%m')
df_hdd_cdd = df_hdd_cdd.set_index('Date').sort_index()
df_hdd_cdd = df_hdd_cdd.resample('D').ffill() #fill out monthly data to days

df_ONI = pd.read_csv(base_path + 'Monthly Oceanic Nino Index (ONI) - Long.csv') #monthly
df_ONI['Date'] = pd.to_datetime(dict(year=df_ONI['Year'], month=df_ONI['MonthNum'], day=1)) #dataframe ends 12-01-2025 not 12-31 so need to manually add the extra days so that there are no NaN entries to the model
df_ONI = df_ONI.set_index('Date').sort_index()
df_ONI = df_ONI.drop(columns = ['Year', 'MonthTxt', 'MonthNum'])
last = df_ONI.index.max()
next_month = last + pd.offsets.MonthBegin(1)
df_ONI.loc[next_month] = df_ONI.loc[last]
df_ONI = df_ONI.resample('D').ffill()
df_ONI.columns = ['ONI']

df_pricedata = pd.read_csv(base_path + 'price_data.csv')
df_pricedata.columns = ['Date', 'PX_LAST', 'Lag [t-1]', 'Lag [t-7]', 'Rolling Average [7d]', 'Rolling Average [30d]', 'NatGas Lag [t-1]', 'NatGas Lag [t-7]', 'NatGas Rolling [7d]', 'NatGas Rolling [30d]', 'NatGas Rolling 2 [30d]'] #temporary we can name these whatever evenutally
df_pricedata['Date'] = pd.to_datetime(df_pricedata['Date'], format='%m/%d/%Y')
df_pricedata = df_pricedata.set_index('Date').sort_index()
df_pricedata = df_pricedata.drop(columns = ['NatGas Rolling 2 [30d]'])

df_1day_load = pd.read_csv(base_path + 'caiso_load_forecast_daily_2010_2026.csv')
df_1day_load['date'] = pd.to_datetime(df_1day_load['date'], format='mixed')
df_1day_load.columns = ['Date', 'Mean Hourly Forecast [MW]', 'Max Hourly Forecast [MW]', 'Min Hourly Forecast [MW]']
df_1day_load = df_1day_load.set_index('Date').sort_index()
#print(df_1day_load.shape[0])

df_7day_load = pd.read_csv(base_path + 'caiso_tac_7day_load_forecast_daily_2010_2026.csv')
df_7day_load['date'] = pd.to_datetime(df_7day_load['date'], format='mixed')
df_7day_load['date'] = df_7day_load['date'] - timedelta(weeks = 1)
df_7day_load = df_7day_load.drop(columns=['pge_7day_load_forecast_hours', 'first_publish_time', 'last_publish_time'])
df_7day_load.columns = ['Date', 'Week-Ahead Mean Daily Forecast [MW]', 'Week-Ahead Max Daily Forecast [MW]', 'Week-Ahead Min Daily Forecast [MW]']
df_7day_load = df_7day_load.set_index('Date').sort_index()
#print(df_7day_load.shape[0])

dfs = [df_hdd_cdd, df_ONI, df_localtemps, df_pricedata, df_1day_load, df_7day_load] #add new dataframes here

####################################################################################
# double check no duplicates, aligned indexes, etc. to prepare for splice and join
####################################################################################

cleaned = []
for df in dfs:
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    cleaned.append(df)
dfs = cleaned
dfs = [df.loc[start_date:end_date] for df in dfs]
df_hdd_cdd, df_ONI, df_localtemps, df_pricedata, df_1day_load, df_7day_load = dfs

    
dfs = [df.loc[start_date:end_date] for df in dfs]
df_hdd_cdd, df_ONI, df_localtemps, df_pricedata, df_1day_load, df_7day_load = dfs #add back in any new dataframes

####################################################################################
# Final Concatenation, Formatting to feed into ML models
####################################################################################

monthly_df_list = [df_hdd_cdd, df_ONI]
daily_df_list = [df_pricedata, df_localtemps]
load_df_list = [df_1day_load, df_7day_load]

df_monthly = pd.concat(monthly_df_list, axis=1, join='outer') #join into seperate sets for later
df_daily = pd.concat(daily_df_list, axis=1, join='outer')
df_load = pd.concat(load_df_list, axis = 1, join = 'outer')

###df_load fix for datetime duplicate error
manual_rows = {"2014-11-26": [25245.268, 29913.52, 19864.41],"2018-09-09": [28060.909, 35389.44, 22421.09],"2018-09-26": [27511.411, 33713.98, 21236.35],"2018-09-27": [29351.431, 36329.69, 22088.29]}
cols = ["Week-Ahead Mean Daily Forecast [MW]", "Week-Ahead Max Daily Forecast [MW]", "Week-Ahead Min Daily Forecast [MW]"]
for d, vals in manual_rows.items(): df_load.loc[pd.Timestamp(d), cols] = vals
df_load = df_load.sort_index()

n_vars_monthly = df_monthly.shape[1] #variable so we don't have to mess with stuff in the model as we add stuff
n_vars_daily = df_daily.shape[1]
n_vars_load = df_load.shape[1]

df_load_daily_combined = pd.concat([df_daily, df_load], axis = 1, join = 'outer')
combined_df = pd.concat([df_monthly, df_daily], axis = 1, join = 'outer') #one total dataframe

###########################################################
#Cell 3
###########################################################

###############################################################
# Settings
###############################################################

horizons = [1, 2, 7]
window = 7
horizon = 7

###############################################################
# Random Forest: 1, 2, 7 days ahead
###############################################################

rf_metrics = []
rf_predictions = {}
rf_actuals = {}

for h in horizons:

    df_rf = combined_df.copy()
    df_rf["Target"] = df_rf["PX_LAST"].shift(-h)
    df_rf = df_rf.dropna()

    X_rf = df_rf.drop(columns=["PX_LAST", "Target", "Target PX_LAST [t+1]"], errors="ignore")
    y_rf = df_rf["Target"]

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rf,
        y_rf,
        test_size=0.2,
        shuffle=False
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )

    rf_model.fit(X_train_rf, y_train_rf)

    ###############################################################
    # RF Feature Importance
    ###############################################################

    rf_importance = pd.DataFrame({
        "Feature": X_train_rf.columns,
        "Importance": rf_model.feature_importances_
    })

    rf_importance = rf_importance.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(
        rf_importance["Feature"].head(15)[::-1],
        rf_importance["Importance"].head(15)[::-1]
    )
    plt.title(f"RF Feature Importance: {h}-Day Ahead", fontweight="semibold")
    plt.xlabel("Importance")
    plt.grid(axis="x")
    plt.show()

    rf_pred = rf_model.predict(X_test_rf)

    rf_predictions[h] = pd.Series(rf_pred, index=y_test_rf.index)
    rf_actuals[h] = y_test_rf

    rf_metrics.append({
        "Model": "Random Forest",
        "Horizon": h,
        "MAE": mean_absolute_error(y_test_rf, rf_pred),
        "RMSE": mean_squared_error(y_test_rf, rf_pred) ** 0.5,
        "R-squared": r2_score(y_test_rf, rf_pred)
    })

###############################################################
# CNN: Create windows with PX_LAST excluded from input
###############################################################

def create_hybrid_windows(daily_df, monthly_df, target_col="PX_LAST", window=7, horizon=7):

    X_daily = []
    X_month = []
    y = []

    daily_feature_df = daily_df.drop(columns=[target_col])
    daily_vals = daily_feature_df.values
    monthly_vals = monthly_df.values
    target_vals = daily_df[target_col].values

    n = len(daily_df)

    for i in range(n - window - horizon):
        X_daily.append(daily_vals[i:i+window].T)
        X_month.append(monthly_vals[i+window-1])
        y.append(target_vals[i+window:i+window+horizon])

    return np.array(X_daily), np.array(X_month), np.array(y), daily_feature_df.columns


X_daily, X_month, y, daily_feature_cols = create_hybrid_windows(
    df_daily,
    df_monthly,
    target_col="PX_LAST",
    window=window,
    horizon=horizon
)

n_vars_daily = X_daily.shape[1]
n_vars_monthly = X_month.shape[1]

split = int(0.8 * len(X_daily))

X_daily_train = X_daily[:split]
X_month_train = X_month[:split]
y_train_cnn = y[:split]

X_daily_val = X_daily[split:]
X_month_val = X_month[split:]
y_val_cnn = y[split:]

###############################################################
# CNN Dataset
###############################################################

class HybridClimateDataset(Dataset):
    def __init__(self, X_daily, X_month, y):
        self.X_daily = torch.tensor(X_daily, dtype=torch.float32)
        self.X_month = torch.tensor(X_month, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_daily)

    def __getitem__(self, idx):
        return self.X_daily[idx], self.X_month[idx], self.y[idx]


train_ds = HybridClimateDataset(X_daily_train, X_month_train, y_train_cnn)
val_ds = HybridClimateDataset(X_daily_val, X_month_val, y_val_cnn)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

###############################################################
# CNN Model
###############################################################

class HybridCNN(nn.Module):
    def __init__(self, in_channels_daily, monthly_features, horizon):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels_daily, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(monthly_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x_daily, x_month):
        daily_latent = self.cnn(x_daily).squeeze(-1)
        month_latent = self.mlp(x_month)
        fused = torch.cat([daily_latent, month_latent], dim=1)
        return self.fc(fused)


cnn_model = HybridCNN(
    in_channels_daily=n_vars_daily,
    monthly_features=n_vars_monthly,
    horizon=horizon
)

###############################################################
# Train CNN
###############################################################

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

epochs = 25

for epoch in range(epochs):

    cnn_model.train()
    train_loss = 0.0

    for xb_daily, xb_month, yb in train_loader:
        optimizer.zero_grad()
        pred = cnn_model(xb_daily, xb_month)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    cnn_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb_daily, xb_month, yb in val_loader:
            pred = cnn_model(xb_daily, xb_month)
            val_loss += criterion(pred, yb).item()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss / len(train_loader):.4f} | "
        f"Val Loss: {val_loss / len(val_loader):.4f}"
    )

###############################################################
# CNN Predictions
###############################################################

cnn_model.eval()

cnn_preds = []
cnn_actuals = []

with torch.no_grad():
    for xb_daily, xb_month, yb in val_loader:
        out = cnn_model(xb_daily, xb_month)
        cnn_preds.append(out.numpy())
        cnn_actuals.append(yb.numpy())

cnn_preds = np.vstack(cnn_preds)
cnn_actuals = np.vstack(cnn_actuals)

cnn_dates = df_daily.index[window + split : window + split + len(cnn_actuals)]

###############################################################
# CNN Permutation Feature Importance
###############################################################

X_daily_val = X_daily[split:].copy()
X_month_val = X_month[split:].copy()
y_val = y[split:].copy()

daily_feature_names = list(daily_feature_cols)

cnn_model.eval()

with torch.no_grad():
    baseline_pred = cnn_model(
        torch.tensor(X_daily_val, dtype=torch.float32),
        torch.tensor(X_month_val, dtype=torch.float32)
    ).numpy()

for h in horizons:

    col = h - 1

    baseline_loss = mean_squared_error(
        y_val[:, col],
        baseline_pred[:, col]
    )

    importance_rows = []

    for j, feature_name in enumerate(daily_feature_names):

        X_daily_perm = X_daily_val.copy()

        shuffle_idx = np.random.permutation(X_daily_perm.shape[0])
        X_daily_perm[:, j, :] = X_daily_perm[shuffle_idx, j, :]

        with torch.no_grad():
            perm_pred = cnn_model(
                torch.tensor(X_daily_perm, dtype=torch.float32),
                torch.tensor(X_month_val, dtype=torch.float32)
            ).numpy()

        perm_loss = mean_squared_error(
            y_val[:, col],
            perm_pred[:, col]
        )

        importance_rows.append({
            "Feature": feature_name,
            "Importance": perm_loss - baseline_loss
        })

    cnn_importance = pd.DataFrame(importance_rows)
    cnn_importance = cnn_importance.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(
        cnn_importance["Feature"].head(15)[::-1],
        cnn_importance["Importance"].head(15)[::-1]
    )
    plt.title(f"CNN Permutation Feature Importance: {h}-Day Ahead", fontweight="semibold")
    plt.xlabel("Increase in MSE after shuffling")
    plt.grid(axis="x")
    plt.show()

###############################################################
# Metrics Comparison
###############################################################

comparison_rows = rf_metrics.copy()
for h in horizons:
    col = h - 1
    comparison_rows.append({
        "Model": "CNN",
        "Horizon": h,
        "MAE": mean_absolute_error(cnn_actuals[:, col], cnn_preds[:, col]),
        "RMSE": mean_squared_error(cnn_actuals[:, col], cnn_preds[:, col]) ** 0.5,
        "R-squared": r2_score(cnn_actuals[:, col], cnn_preds[:, col])
    })
comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values(["Horizon", "Model"]).reset_index(drop=True)


###############################################################
# Spike-Only Metrics: RF vs CNN
###############################################################

spike_rows = []

for h in horizons:

    col = h - 1

    cnn_actual_h = pd.Series(cnn_actuals[:, col], index=cnn_dates)     # CNN actual/pred
    cnn_pred_h = pd.Series(cnn_preds[:, col], index=cnn_dates)

    rf_actual_h = rf_actuals[h] # RF actual/pred
    rf_pred_h = rf_predictions[h]

    common_idx = cnn_actual_h.index.intersection(rf_actual_h.index) # Use common dates so RF and CNN are compared on the same days

    actual_common = cnn_actual_h.loc[common_idx]
    cnn_pred_common = cnn_pred_h.loc[common_idx]
    rf_pred_common = rf_pred_h.loc[common_idx]

    spike_threshold = actual_common.quantile(0.90) # Define spikes using actual prices only
    spike_mask = actual_common >= spike_threshold

    for model_name, pred_common in [
        ("Random Forest", rf_pred_common),
        ("CNN", cnn_pred_common)
    ]:

        spike_rows.append({
            "Model": model_name,
            "Horizon": h,
            "Spike Threshold": spike_threshold,
            "Spike Days": int(spike_mask.sum()),
            "Spike MAE": mean_absolute_error(actual_common[spike_mask], pred_common[spike_mask]),
            "Spike RMSE": mean_squared_error(actual_common[spike_mask], pred_common[spike_mask]) ** 0.5,
            "Normal MAE": mean_absolute_error(actual_common[~spike_mask], pred_common[~spike_mask]),
            "Normal RMSE": mean_squared_error(actual_common[~spike_mask], pred_common[~spike_mask]) ** 0.5
        })

spike_comparison_df = pd.DataFrame(spike_rows)
spike_comparison_df = spike_comparison_df.sort_values(["Horizon", "Model"]).reset_index(drop=True)

###############################################################
# Plot RF vs CNN
###############################################################

for h in horizons:

    col = h - 1
    plt.figure(figsize=(14,5))
    plt.plot(cnn_dates,cnn_actuals[:, col],label=f"Validation")
    plt.plot(cnn_dates,cnn_preds[:, col],label=f"CNN Prediction ({h}-day ahead)")
    common_idx = rf_predictions[h].index.intersection(cnn_dates)
    plt.plot(common_idx,rf_predictions[h].loc[common_idx],label=f"RF Prediction ({h}-day ahead)")

    plt.title(f"RF vs CNN: {h}-Day Ahead Forecast", fontweight="semibold")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.ylabel("Energy Price [$/MWh]", fontweight="semibold")
    plt.legend()
    plt.show()

###########################################################
#Cell 4
###########################################################

######################################################
# Random Forest: 1-day, 2-day, and 7-day Ahead Forecasts
######################################################

horizons = [1, 2, 7]
models = {}
predictions = {}
actuals = {}
metrics = []

for h in horizons:

    ######################################################
    # Dataframe for each horizon
    ######################################################

    df_rf_h=combined_df.copy()
    df_rf_h["Target"] = df_rf_h["PX_LAST"].shift(-h)  # Target = price h days ahead
    df_rf_h = df_rf_h.dropna() # Drop missing values
    x_rf_h = df_rf_h.drop(columns=["PX_LAST", "Target", "Target PX_LAST [t+1]"], errors="ignore")  # X = features only, no actual price and no target columns
    y_rf_h = df_rf_h["Target"]

    ######################################################
    # Train/test split
    ######################################################

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(x_rf_h,y_rf_h,test_size=0.2,shuffle=False)

    ######################################################
    # Random Forest model
    ######################################################

    rf_h = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )

    rf_h.fit(X_train_h, y_train_h)

    ######################################################
    # Feature Importance
    ######################################################

    feature_importance_h = pd.DataFrame({
        "Feature": X_train_h.columns,
        "Importance": rf_h.feature_importances_
    })

    feature_importance_h = feature_importance_h.sort_values(
        by="Importance",
        ascending=False
    )

    print(f"\nTop Feature Importances for {h}-day ahead forecast")

    plt.figure(figsize=(10,6))
    plt.barh(
        feature_importance_h["Feature"].head(15)[::-1],
        feature_importance_h["Importance"].head(15)[::-1]
    )

    plt.title(f"RF Feature Importance: {h}-Day Ahead", fontweight="semibold")
    plt.xlabel("Importance")
    plt.grid(axis="x")
    plt.show()

    ######################################################
    # Metrics
    ######################################################

    pred_h = rf_h.predict(X_test_h)
    mse_h = mean_squared_error(y_test_h, pred_h)
    rmse_h = mse_h ** 0.5
    mae_h = mean_absolute_error(y_test_h, pred_h)
    r2_h = r2_score(y_test_h, pred_h)

    print("\n================================================")
    print(f"Random Forest: {h}-day ahead forecast")
    print("OOB Score:", rf_h.oob_score_)
    print("MAE:", mae_h)
    print("RMSE:", rmse_h)
    print("Mean Squared Error:", mse_h)
    print("R-squared:", r2_h)

    models[h] = rf_h
    predictions[h] = pd.Series(pred_h, index=y_test_h.index)
    actuals[h] = y_test_h

    metrics.append({
        "Horizon": h,
        "OOB Score": rf_h.oob_score_,
        "MAE": mae_h,
        "RMSE": rmse_h,
        "MSE": mse_h,
        "R-squared": r2_h
    })

    ######################################################
    # Plot forecast
    ######################################################

    plt.figure(figsize=(14,5))
    plt.plot(y_test_h.index, y_test_h, label=f"Actual Test ({h}-day ahead)")
    plt.plot(y_test_h.index, pred_h, label=f"RF Prediction ({h}-day ahead)")

    plt.title(f"Random Forest {h}-Day Ahead Forecast", fontweight="semibold")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.ylabel("Energy Price [$/MWh]", fontweight="semibold")
    plt.legend()
    plt.show()

    ######################################################
    # Uncertainty band from tree disagreement
    ######################################################

    tree_predictions = []

    for tree in rf_h.estimators_:
        tree_pred = tree.predict(X_test_h)
        tree_predictions.append(tree_pred)

    tree_preds = np.array(tree_predictions)

    q10_pred = np.percentile(tree_preds, 10, axis=0)
    q90_pred = np.percentile(tree_preds, 90, axis=0)

    plt.figure(figsize=(14,5))
    plt.plot(y_test_h.index, y_test_h, label=f"Actual Test ({h}-day ahead)")
    plt.plot(y_test_h.index, pred_h, label=f"RF Mean Prediction ({h}-day ahead)")
    plt.fill_between(y_test_h.index, q10_pred, q90_pred, alpha=0.2, label="RF 10–90% Band")

    plt.title(f"Random Forest {h}-Day Ahead Forecast with Uncertainty Band", fontweight="semibold")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.ylabel("Energy Price [$/MWh]", fontweight="semibold")
    plt.legend()
    plt.show()

######################################################
# Metrics Table
######################################################

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

###########################################################
#End
###########################################################
