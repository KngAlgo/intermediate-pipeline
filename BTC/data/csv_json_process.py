import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcess:

    def csv_importer(name):
        csv_file = pd.read_csv(name)
        return csv_file

    def json_importer(name):
        json_file = pd.read_json(name)
        return json_file

    def data_processor(file, file_type, fix_na=True, drop_threshold=0.25):
        if file_type == 'csv':
            data = DataProcess.csv_importer(file)
        elif file_type == 'json':
            data = DataProcess.json_importer(file)

        if fix_na == True:
            for col in data.columns:
                n_unique = data[col].nunique()
                na_count = data[col].isna().sum()
                na_percent = na_count / len(data[col])
                if na_percent > drop_threshold:
                    data = data.drop(columns=[col])
                else:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        if n_unique > 20:
                            mean = data[col].mean()
                            data[col].fillna(mean, inplace=True)

                            scaler = StandardScaler()
                            data[col] = scaler.fit_transform(data[[col]]).flatten()
                        else:
                            mode = data[col].mode()[0]
                            data[col].fillna(mode, inplace=True)

                    elif pd.api.types.is_object_dtype(data[col]):
                        ratio = data[col].nunique() / len(data[col].dropna())
                        if ratio < 0.5:
                            mode = data[col].mode()[0]
                            data[col].fillna(mode, inplace=True)
                        else:
                            data[col].fillna("[UNK]", inplace=True)

        else:
            data = data.dropna()
        
        return data
            
def split_data(data, target_column, test_size=0.2):
    x = data.drop(columns=(target_column))
    y = data[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    return x_train, x_test, y_train, y_test

def split_ds(data, target_column):
    x = data.drop(columns=(target_column))
    y = data[target_column]
    return x, y


# Manual Preprocessing (for specifics)
