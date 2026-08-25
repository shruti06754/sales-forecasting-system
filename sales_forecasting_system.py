

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
pd.set_option('display.max_columns', None)
print("Libraries loaded successfully.")

USE_UPLOADED_CSV = False
if USE_UPLOADED_CSV:
  from google.colab import files
  uploaded =files.upload()
  filename = next(iter(uploaded))
  df_raw = pd.read_csv(filename)
  print("Uploaded:", filename)
else:

   np.random.seed(42)
   dates =pd.date_range("2022-01-01", "2024-12-31", freq="D")
   n = len(dates)

trend = np.linspace(0, 1800, n)
yearly = 1800 * np.sin(2*np.pi*dates.dayofyear.to_numpy()/365.25)
weekly = 700 * np.sin(2*np.pi*dates.dayofweek.to_numpy()/7)
noise = np.random.normal(0, 450, n)


month = dates.month.to_numpy()
holiday_boost = np.where(np.isin(month, [11, 12]), 1800, 0)

sales = 12000 + trend + yearly + weekly + holiday_boost + noise
sales = np.maximum(sales, 1000)

df_raw = pd.DataFrame({"Date": dates, "Sales": sales})

df_raw.head()

df = df_raw.copy()



rename_map = {}
for c in df.columns:
  cl = c.strip().lower().replace("","_")
if cl =="date":
     rename_map[c] = "Date"
elif cl in ["sales", "weekly_sales", "sales_amount", "sales_amount_quantity"]:
    rename_map[c]= "Sales"

df = df.rename(columns=rename_map)

if "Sales" not in df.columns:
  raise ValueError("Your CSV must contain a Sales or Weekly_Sales column.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df = df.dropna(subset=["Date", "Sales"])
df = df.drop_duplicates()
df = df.groupby("Date", as_index=False)["Sales"].sum()
df = df.sort_values("Date").reset_index(drop=True)

print("Rows after cleaning:", len(df))
print("Missing values:")
print(df.isna().sum())
df.head()

plt.figure(figsize=(14,5))
plt.plot(df["Date"], df["Sales"])
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(alpha=0.2)
plt.show()

monthly = df.set_index("Date")["Sales"].resample("ME").sum()

plt.figure(figsize=(14,5))
plt.plot(monthly.index, monthly.values)
plt.title("monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(alpha=0.2)
plt.show()

print("Highest sales date:", df.loc[df["Sales"].idxmax(), "Date"])
print("“Highest sales value:", round(df["Sales"].max(), 2))
print("Lowest sales date:", df.loc[df["Sales"].idxmin(), "Date"])
print("Lowest sales value:", round(df["Sales"].min(), 2))


data = df.copy()


data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["DayOfYear"] = data["Date"].dt.dayofyear
data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)


for lag in [1, 7, 14, 28]:
    data[f"Lag_{lag}"] = data["Sales"].shift(lag)

data["RollingMean_7"] = (
    data["Sales"].shift(1).rolling(7).mean()
)

data["RollingMean_28"] = (
    data["Sales"].shift(1).rolling(28).mean()
)
data = data.dropna().reset_index(drop=True)

features = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "WeekOfYear",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "RollingMean_7",
    "RollingMean_28"
]

X = data[features]

y = data["Sales"]

print("Features:", features)

data.head()

from sklearn.ensemble import RandomForestRegressor


split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


model = RandomForestRegressor(
    n_estimators=250,
    random_state=42,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2
)


model.fit(X_train, y_train)

print("Model trained successfully.")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))



test_pred = model.predict(X_test)



comparison = pd.DataFrame({
    "Date": data.loc[X_test.index, "Date"].values,
    "Actual": y_test.values,
    "Predicted": test_pred
})


comparison.head(10)



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.plot(
    comparison["Date"],
    comparison["Actual"],
    label="Actual"
)

plt.plot(
    comparison["Date"],
    comparison["Predicted"],
    label="Predicted"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(alpha=0.2)

plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, test_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, test_pred)
)

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")



history = (
    df[["Date", "Sales"]]
    .copy()
    .sort_values("Date")
    .reset_index(drop=True)
)

future_rows = []


for _ in range(30):

    
    next_date = history["Date"].iloc[-1] + pd.Timedelta(days=1)


    s = history["Sales"]

    
    row = {
        "Year": next_date.year,
        "Month": next_date.month,
        "Day": next_date.day,
        "DayOfWeek": next_date.dayofweek,
        "DayOfYear": next_date.dayofyear,
        "WeekOfYear": int(next_date.isocalendar().week),

        "Lag_1": s.iloc[-1],
        "Lag_7": s.iloc[-7],
        "Lag_14": s.iloc[-14],
        "Lag_28": s.iloc[-28],

        "RollingMean_7": s.iloc[-7:].mean(),
        "RollingMean_28": s.iloc[-28:].mean()
    }

    pred = model.predict(
        pd.DataFrame([row])[features]
    )[0]

    
    future_rows.append({
        "Date": next_date,
        "Predicted_Sales": pred
    })

    
    history = pd.concat(
        [
            history,
            pd.DataFrame({
                "Date": [next_date],
                "Sales": [pred]
            })
        ],
        ignore_index=True
    )


future_forecast = pd.DataFrame(future_rows)

future_forecast.head(30)



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))


plt.plot(
    df["Date"].tail(120),
    df["Sales"].tail(120),
    label="Historical Sales"
)

plt.plot(
    future_forecast["Date"],
    future_forecast["Predicted_Sales"],
    marker="o",
    label="Next 30 Days Forecast"
)

plt.title("Future Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(alpha=0.2)

plt.show()



average_sales = future_forecast["Predicted_Sales"].mean()

print(
    "Average predicted sales for next 30 days:",
    round(average_sales, 2)
)



future_forecast.to_csv(
    "sales_forecast_30_days.csv",
    index=False
)

comparison.to_csv(
    "actual_vs_predicted.csv",
    index=False
)

print("Files saved:")
print("- sales_forecast_30_days.csv")
print("- actual_vs_predicted.csv")
