import loaders as L
import pandas as pd

info = L.load_event_info("A")                        # event metadata
ids  = L.list_event_ids("A", label="anomaly")        # 12 anomaly, 10 normal in Farm A
df   = L.load_event("A", 0)                          # adds is_anomaly and is_normal_status columns
train, test = L.split_train_test(df)
X = df[L.sensor_columns(df)]

print(info.head())
print(ids)
print(df.head())