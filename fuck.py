import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl  # using lightning.pytorch for consistency
from pytorch_forecasting import TimeSeriesDataSet, NBeats

# ---------------------------------
# Toggle: Set to True to apply log transform and detrending,
#         or False to use raw values.
# ---------------------------------
apply_transform = True

# -------------------------
# 1. Load and Preprocess Data
# -------------------------
data = pd.read_csv("Monthly.csv")
data['sasdate'] = pd.to_datetime(data['sasdate'], format='%m/%d/%Y', errors='coerce')
data = data.dropna(subset=['sasdate'])
data = data.sort_values('sasdate').reset_index(drop=True)
data = data.set_index('sasdate')

# -------------------------
# 2. Prepare Data: Add Time Index and Group
# -------------------------
data = data.copy()
data["time_idx"] = np.arange(len(data))
data["group"] = "group1"  # single series

# -------------------------
# 3. Process Data: Optionally apply Log-transform and Detrend
# -------------------------
if apply_transform:
    # Apply log transform to the raw target values
    data["log_target"] = np.log(data["DPCERA3M086SBEA"])
    # Compute a linear trend on the log-transformed data
    slope, intercept = np.polyfit(data["time_idx"], data["log_target"], 1)
    data["trend"] = intercept + slope * data["time_idx"]
    # Detrend the log-transformed series
    data["detrended"] = data["log_target"] - data["trend"]
else:
    # Use the raw target values without transformation
    data["detrended"] = data["DPCERA3M086SBEA"]

# -------------------------
# 4. Define Training Cutoff and Forecast Horizon Based on Date
#    (Use only data through 2019 for training; forecast from 2020 onward)
# -------------------------
max_training_year = 2019

# Select training data (all observations in 2019 or earlier)
training_data = data.loc[data.index.year <= max_training_year].copy()

# Use the maximum time index from the training data as cutoff.
# (Remember that "time_idx" is a numeric column from 0 to len(data)-1.)
cutoff = training_data["time_idx"].max()

# Forecast horizon: number of observations after the training cutoff in the full dataset.
# (For example, if your full dataset extends several years into the 2020s.)
forecast_horizon = len(data.loc[data.index.year > max_training_year])
# Alternatively, to forecast a fixed number of steps (e.g., 12 months), you could use:
# forecast_horizon = 12

# Adjust maximum encoder length based on training data length (or keep it fixed)
max_encoder_length = min(36, len(training_data))

# -------------------------
# 5. Create Training and Prediction Datasets
# -------------------------
# Build the training dataset from the training data only.
training_dataset = TimeSeriesDataSet(
    training_data,
    time_idx="time_idx",
    target="detrended",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=forecast_horizon,
    time_varying_unknown_reals=["detrended"],
    target_normalizer=None,
)

# Build the prediction (forecast) dataset from the full data.
# We set min_prediction_idx to cutoff+1 so that forecasts start after the training period.
predict_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    data,
    min_prediction_idx=cutoff + 1,  # forecasts start from the first index after training cutoff
    predict_mode=True,
)

train_dataloader = training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)
predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

# -------------------------
# 6. Train the N‑BEATS Model with Increased Epochs (e.g., 50 epochs)
# -------------------------
trainer = pl.Trainer(
    max_epochs=50,   # increased epochs for improved training
    accelerator="cpu",  # change to "gpu" if available
    devices=1,
    logger=False
)

net = NBeats.from_dataset(
    training_dataset,
    learning_rate=1e-3,
    log_interval=10,
    weight_decay=1e-2,
    widths=[512, 512],
    backcast_loss_ratio=0.1,
)

trainer.fit(net, train_dataloaders=train_dataloader)

# -------------------------
# 7. Forecast Future Values (After 2019)
# -------------------------
predicted_series = net.predict(predict_dataloader)
forecast_output = predicted_series.detach().cpu().numpy().squeeze()

if apply_transform:
    # If using the log transform, reconstruct the original scale.
    forecast_indices = np.arange(cutoff + 1, len(data))
    forecast_trend = intercept + slope * forecast_indices
    predicted_log = forecast_output + forecast_trend
    predicted_values = np.exp(predicted_log)
else:
    # If no transform was applied, the forecast output is already on the original scale.
    predicted_values = forecast_output

# -------------------------
# 8. Plot Actual vs Predicted Values
# -------------------------
# Select actual values for dates after 2019.
actual_series = data.loc[data.index.year > max_training_year, "DPCERA3M086SBEA"]

# Create a date range for the forecasted period.
# Assuming monthly frequency and that training data ends with the last date in 2019.
last_training_date = training_data.index[-1]
forecast_index = pd.date_range(
    start=last_training_date + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq='M'
)

plt.figure(figsize=(12, 6))
plt.plot(actual_series.index.to_numpy(), actual_series.values, label="Actual", marker="o")
plt.plot(forecast_index.to_numpy(), predicted_values, label="Forecast (Predicted)", marker="x")
plt.title("Actual vs Predicted GDP Values (Forecasting After 2019)")
plt.xlabel("Date")
plt.ylabel("GDP (DPCERA3M086SBEA)")
plt.legend()
plt.grid(True)
plt.show()
