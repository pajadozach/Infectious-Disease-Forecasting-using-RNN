"""
Forecasting Model

Install dependencies separately with:
    python -m pip install pandas numpy scikit-learn keras matplotlib seaborn dill keras-tuner tensorflow

Run from the project root containing Data/data.csv.
"""

# Imports
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Bidirectional, Attention, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from keras_tuner.tuners import BayesianOptimization
from kerastuner.engine.hypermodel import HyperModel

from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from keras_tuner.engine import trial as trial_module

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from keras.layers import Bidirectional

from keras_tuner.engine.trial import Trial

# Data preparation

file_path = 'Data/data.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df.dropna(subset=['Date'], inplace=True)

df_raw = df.copy()

numeric_cols = ['Cases', 'tsf', 'mst', 'rh', 'rfm', 'sca']

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

horizon = 1

def Lag(data, lags):
    for lag in lags:
        data[f'lag_{lag}'] = data['Cases'].shift(lag)
    return data

lags = [1, 2, 3, 4, 5, 6] # Specify how many lagged features are used
df = Lag(df, lags=lags)
df.dropna(inplace=True)

feature_cols = [f'lag_{lag}' for lag in lags] + ['tsf','mst','rh','rfm','sca'] #Insert the lagged cases into the numeric cols
n_features = len(feature_cols)

def build_model(hp):
    window = hp.Int('window', 2, 8, step=2)

    model = Sequential()
    model.add(Input(shape=(window, n_features)))

    num_layers = hp.Int('num_layers', 1, 3, 1) #(1,4,1)
    for i in range(num_layers):
        units      = hp.Int(f'lstm_units_{i}',    64, 512, 32)
        rec_drop   = hp.Float(f'rec_dropout_{i}',  0.0, 0.3, 0.05)
        return_seq = (i < num_layers - 1)

        model.add(LSTM(units,return_sequences=return_seq, recurrent_dropout=rec_drop))

        model.add(Dropout(hp.Float(f'dropout_{i}', 0.1, 0.3, 0.05)))

    model.add(Dense(1))

    lr  = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
    opt = Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])
    return model

class Optimize(BayesianOptimization):
    '''
    Hyperparameters are formatted as (hyperparamemter, min, max, steps)
    '''
    def run_trial(self, trial: Trial, df, features, **fit_kwargs):
        window = trial.hyperparameters.Int('window', 2, 16, step=1) # Match the window parameters to the model
        batch_size = trial.hyperparameters.Int('batch_size', 32, 128, 16)
        epochs     = trial.hyperparameters.Int('epochs',     20, 150, 10)

        df_proc = df.copy() #Use the copy of the data
        df_proc['y_future'] = df_proc['Cases'].shift(-horizon) #Shifting the cases t-horizon days to line up with features in t
        df_proc.dropna(subset=['y_future'], inplace=True)
        X_loc, y_loc = [], []

        for i in range(window, len(df_proc)-horizon):
            X_loc.append(df_proc[features].iloc[i-window:i].values)
            y_loc.append(df_proc['y_future'].iloc[i])

        X_loc = np.array(X_loc)
        y_loc = np.array(y_loc)

        scaler = MinMaxScaler()

        X_train_2d = X_loc[:train_end].reshape(-1, X_loc.shape[-1])
        scaler.fit(X_train_2d)

        X_loc_scaled = scaler.transform(X_loc.reshape(-1, X_loc.shape[-1])).reshape(X_loc.shape)

        y_loc_scaled = y_scaler.transform(y_loc.reshape(-1,1)).flatten()

        sw = 1.0 + np.where(y_loc > 0, 5.0, 0.5)
        sw_train = sw[:train_end]
        sw_val   = sw[train_end:val_end]

        callbacks = [EarlyStopping(monitor='mse', patience=10, restore_best_weights=True)]

        model = build_model(trial.hyperparameters)
        model.y_scaler = y_scaler

        return super().run_trial(
            trial,
            X_loc_scaled[:train_end], y_loc_scaled[:train_end],
            validation_data=(X_loc_scaled[train_end:val_end], y_loc_scaled[train_end:val_end]),
            sample_weight=sw_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            **fit_kwargs
        )

n_features = len(feature_cols)

df_proc = df.copy()
df_proc['y_future'] = df_proc['Cases'].shift(-horizon)
df_proc.dropna(subset=['y_future'], inplace=True)

X, y = [], []
window = best_hp.get('window')

for i in range(window, len(df_proc)):
    X.append(df_proc[feature_cols].iloc[i-window:i].values)
    y.append(df_proc['y_future'].iloc[i])

X = np.array(X)
y = np.array(y)

# Split into train, val, test (80/10/10)/ This is based on indexing; make sure that data is arranged linearly.
n = len(X)
train_end = int(0.8 * n)
val_end   = int(0.9 * n)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

feature_scaler = MinMaxScaler()

X_train_2d = X_train.reshape(-1, X_train.shape[-1])
X_val_2d   = X_val.reshape(-1, X_val.shape[-1])
X_test_2d  = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = feature_scaler.fit_transform(X_train_2d).reshape(X_train.shape)

X_val_scaled   = feature_scaler.transform(X_val_2d).reshape(X_val.shape)
X_test_scaled  = feature_scaler.transform(X_test_2d).reshape(X_test.shape)

y_scaler = MinMaxScaler()

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
y_val_scaled   = y_scaler.transform(y_val.reshape(-1,1)).flatten()
y_test_scaled  = y_scaler.transform(y_test.reshape(-1,1)).flatten()

# Hyperparameter tuning

tuner = Optimize(
    hypermodel=build_model,
    objective='val_loss', #MSE in this case
    max_trials=100,
    num_initial_points=10,
    executions_per_trial=2,
    directory='Models',
    project_name=f'H{horizon}',
    #overwrite=True, #Only activate when Overwriting!
)

tuner.search(df=df, features=feature_cols)

best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

best_hp = best_trial.hyperparameters

window = (best_hp.get('window'))
best_batch   = best_hp.get('batch_size')
best_epochs  = best_hp.get('epochs')

num_layers = best_hp.get('num_layers')
lstm_units = [best_hp.get(f'lstm_units_{i}') for i in range(num_layers)]
dropouts = [best_hp.get(f'dropout_{i}') for i in range(num_layers)]
rec_dropouts = [best_hp.get(f'rec_dropout_{i}') for i in range(num_layers)]

best_lr  = best_hp.get('learning_rate')

best_model = tuner.get_best_models(1)[0]
best_model.y_scaler = y_scaler

print(f"Horizon: {horizon}")
print(f"Best window:  {window}")
print(f"Best batch:  {best_batch}")
print(f"Best epoch: {best_epochs}")
print(f"Best learning rate: {best_lr}")
print(f"Best Num Layers: {num_layers}")
print(f"LSTM Units: {lstm_units}")
print(f"Dropouts: {dropouts}")
print(f"Recurrent dropouts: {rec_dropouts}")

tuner.reload()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

provinces = ['Aurora', 'Bulacan', 'Bataan', 'Nueva Ecija', 'Tarlac', 'Pampanga', 'Zambales']
all_results = []

# Province-level predictions
for province in provinces:
    prov_df = df[df['Province'] == province].sort_values('Date').reset_index(drop=True)
    raw = df_raw[df_raw['Province'] == province].sort_values('Date').set_index('Date')
    if len(prov_df) <= window:
        continue

    Xp, dates = [], []
    for i in range(window, len(prov_df) - horizon):
        Xp.append(prov_df[feature_cols].iloc[i-window:i].values)
        dates.append(prov_df.loc[i+horizon, 'Date'])
    Xp = np.array(Xp)
    dates = pd.to_datetime(dates)
    if Xp.size == 0:
        continue

    scaled_pred = best_model.predict(Xp).flatten()
    ypr_raw    = best_model.y_scaler.inverse_transform(scaled_pred.reshape(-1,1)).flatten()
    true_series = raw['Cases'].shift(-horizon)
    ytr_raw = true_series.iloc[window:window + len(ypr_raw)].values

    t = min(len(dates), len(ypr_raw), len(ytr_raw))
    dates, ypr_raw, ytr_raw = dates[:t], ypr_raw[:t], ytr_raw[:t]

    dfp = pd.DataFrame({
        'Date': dates,
        'Province': province,
        'Actual': ytr_raw,
        'Predicted': np.clip(ypr_raw, 0, None)
    })
    all_results.append(dfp)

# One-step-ahead predictions
extras = []
for province in provinces:
    prov_df = df[df['Province'] == province].sort_values('Date').reset_index(drop=True)
    raw = df_raw[df_raw['Province'] == province].sort_values('Date').set_index('Date')
    if len(prov_df) < window:
        continue
    last_win = prov_df[feature_cols].iloc[-window:].values.reshape(1, window, len(feature_cols))

    for h in range(1, horizon + 1):
        spred = best_model.predict(last_win).flatten()[0]
        rawp = best_model.y_scaler.inverse_transform([[spred]])[0, 0]
        pred_un = max(rawp, 0)
        next_date = pd.to_datetime(raw.index.max()) + pd.Timedelta(weeks=h)
        extras.append({'Date': next_date, 'Province': province, 'Actual': np.nan, 'Predicted': pred_un})
all_results.append(pd.DataFrame(extras))

# Combine predictions
full = pd.concat(all_results, ignore_index=True)
full['Date'] = pd.to_datetime(full['Date'])
full = full.set_index(['Date','Province']).sort_index()
start = '2023-07-04'
last_actual = df_raw['Date'].max()
full_results = full.loc[start: last_actual ]#+ pd.Timedelta(weeks=horizon)]

# Weekly totals
raw_weekly = (
    df_raw
    .loc[lambda d: (d['Date'] >= start) & (d['Date'] <= last_actual)]
    .groupby('Date')['Cases']
    .sum()
)
pred_weekly = (
    full_results
    .groupby('Date')['Predicted']
    .sum()
    .reindex(raw_weekly.index.union([last_actual ]))#+ pd.Timedelta(weeks=horizon)]), fill_value=0)
)

# Evaluation
raw_metrics = raw_weekly.loc['2023-07-09':'2023-12-31']
pred_metrics = pred_weekly.loc['2023-07-09':'2023-12-31']
if not raw_metrics.empty:
    y_t = raw_metrics.values
    y_p = pred_metrics.values
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    print(f"Metrics (2023‑07‑09 to 2023‑12‑31): MAE={mae:.2f}, RMSE={rmse:.2f}")
else:
    print("No actuals in Jul–Dec 2023 for metric calc.")

# Plot and export results
plt.figure(figsize=(8,3))
plt.plot(raw_weekly.index, raw_weekly, '--', linewidth=1, alpha=0.7, color='k', label='Actual')
plt.plot(pred_weekly.index, pred_weekly, '-', linewidth=2, color='k', label='Predicted')
plt.title(f'(a) Central Luzon (horizon: {horizon})')
plt.xlabel('Date'); plt.ylabel('Total Case Count')
plt.legend()
ax = plt.gca()
loc = mdates.WeekdayLocator(byweekday=mdates.SU, interval=2)
fmt = mdates.DateFormatter('%b%d')
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(fmt)
ticks = ax.get_xticks()
j1 = mdates.date2num(pd.to_datetime('2023-07-16'))
if ticks.size and ticks[0] > j1:
    ticks[0] = j1
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(fmt)
plt.grid(False)
plt.tight_layout()
plt.savefig(f"Results/Plots/central_luzon({horizon}).tiff", dpi=600, format='tiff', bbox_inches='tight')
plt.show()

export_idx = raw_weekly.index.union([last_actual])#+ pd.Timedelta(weeks=horizon)])
df_export = pd.DataFrame({
    'Date': export_idx,
    'Actual': raw_weekly.reindex(export_idx).values,
    'Predicted': pred_weekly.reindex(export_idx).values
})
df_export['Province'] = 'Central Luzon'
df_export.to_csv(f'Results/CSVs/central_luzon({horizon}).csv', index=False)
print(f"Wrote {len(df_export)} rows (through {last_actual.date() + pd.Timedelta(weeks=1)}) to CSV")
