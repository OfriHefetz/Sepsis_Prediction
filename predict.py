import xgboost as xgb
import pickle
import sys
import os
import pandas as pd
import numpy as np

OPTIMAL_THRESHOLD = 0.35553555355535554

features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
            'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
            'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
            'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
            'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'HR_missing', 'O2Sat_missing',
            'Temp_missing', 'SBP_missing', 'MAP_missing', 'DBP_missing', 'Resp_missing', 'EtCO2_missing',
            'BaseExcess_missing', 'HCO3_missing', 'FiO2_missing', 'pH_missing', 'PaCO2_missing',
            'SaO2_missing', 'AST_missing', 'BUN_missing', 'Alkalinephos_missing', 'Calcium_missing',
            'Chloride_missing', 'Creatinine_missing', 'Bilirubin_direct_missing', 'Glucose_missing',
            'Lactate_missing', 'Magnesium_missing', 'Phosphate_missing', 'Potassium_missing', 'Bilirubin_total_missing',
            'TroponinI_missing', 'Hct_missing', 'Hgb_missing', 'PTT_missing', 'WBC_missing', 'Fibrinogen_missing',
            'Platelets_missing',
            'Age_missing', 'Gender_missing', 'Unit1_missing', 'Unit2_missing', 'HospAdmTime_missing', 'ICULOS_missing']


def read_data(path):
    dfs = []
    for filename in os.listdir(path):
        if filename.endswith(".psv"):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, sep='|')
            df['SepsisPatient'] = 0
            # Filter out rows after the row where SepsisLabel is 1
            if df['SepsisLabel'].eq(1).any():
                index = df.index[df['SepsisLabel'] == 1].min()
                df = df.iloc[:index + 1, :]
                df['SepsisPatient'] = 1
            # Add the filename/Patient name to the dataframe
            df['filename'] = filename.removesuffix('.psv')
            dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df


def data_prep(data):
    #  add a new column for each feature that indicates if the value is missing 0/1 values
    for col in data.columns:
        data[col + '_missing'] = data[col].isna().astype(int)
    return data


if __name__ == '__main__':
    input_dir_path = sys.argv[1]
    # input_dir_path = '/Users/ofrihefetz/PycharmProjects/lab2_hw1/data/test'
    print('Reading dataframes')
    data = read_data(input_dir_path)

    print('Loading model')
    pickled_model = pickle.load(open('XGB_model.pkl', 'rb'))

    print('Preprocessing')
    df = data_prep(data)
    df_ = df.groupby(by=['filename']).mean().reset_index()
    features_new = df.columns.tolist()
    save_features  = [i for i in features_new if i in features]

    final_df = df_[save_features]

    print('Predicting')
    y_predict = pickled_model.predict_proba(final_df)
    y_predicted_label = 1 * (y_predict[:, 1] > OPTIMAL_THRESHOLD)
    df_['predicted_label'] = y_predicted_label

    print('Saving to CSV')
    df_[['filename', 'predicted_label']].to_csv('prediction.csv')
