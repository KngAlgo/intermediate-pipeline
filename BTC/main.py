from BTC.csv_json_process import DataProcess, split_data, split_ds

data = DataProcess.data_processor("train.csv", 'csv', True, 0.25)

x, y = split_ds(data, "price_range")

print(x.head())
