import os
import pandas as pd
import csv
import numpy as np
import pickle
import json


def create_validation_df(validation_path, terms, headers):
    '''
    validation_path (str): path pointing to all the .csv validation files
    term (str): word inside filename that wants to be pulled (ex. card, mesh)
    headers (list): list of strings containing headers that will be converted based on the TERMS dictionary
    '''
    validation_files = os.listdir(validation_path)

    df = []
    for file in validation_files:
        check = [term in file for term in terms]
        if all(check):
            vali_df = pd.read_csv(os.path.join(validation_path, file))
            df.append(vali_df)

    df = pd.concat(df, axis=0, sort=False)

    for item in headers:
        df[item] = [TERMS.get(str(item), 2) for item in df[item]]
    return df


def add_predictions_to_df(path_to_prediction, df):
    with open(path_to_prediction, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    results = {}

    for row in rows:
        results[row['pmid']] = row['prediction']

    df['Prediction'] = [TERMS.get(str(results[str(item)]), 2)
                        for item in df['pmid']]

    return df


def add_multiple_predictions_to_df(path_to_prediction, df):
    for path in path_to_prediction:
        if '.json' in path:
            with open(path, 'r') as file:
                data = json.load(file)
        elif '.pkl' in path:
            with open(path, 'rb') as file:
                data = pickle.load(file)

        for ml_type in data:
            data[ml_type] = {
                int(pmid): prediction for pmid, prediction in data[ml_type]}

        pmid_list = list(df['pmid'])
        # print(data[ml_type])
        for ml_type in data:
            predictions = []
            for pmid in pmid_list:
                predictions.append(data[ml_type][pmid])

            df[ml_type] = predictions

    return df


def add_final_result_to_df(df, headers, df_type):
    header_results = {item: list(df[item]) for item in headers[:2]}

    for ml_type in df:
        if any([x in ml_type for x in ('lr', 'rf', 'nb', 'xgb', 'svm', 'random', 'shark')]):
            predictions = list(df[ml_type])
            final_results = []

            for i, prediction in enumerate(predictions):
                header_values = [header_results[x][i] for x in header_results]

                if header_values[0] == header_values[1]:
                    actual = header_values[0]
                else:
                    actual = 0

                if prediction == actual:
                    if prediction == 0:
                        value = 'TN'
                    if prediction == 1:
                        value = 'TP'
                else:
                    if prediction == 0:
                        value = 'FN'
                    if prediction == 1:
                        value = 'FP'

                # print(prediction, actual, value)
                final_results.append(value)

            df['Result'+ml_type] = final_results

    return df


def remove_different_results(df, headers, df_type, filters):
    '''
    If two rows with the same PMID have a different set of results, remove both rows.
    '''
    pmids = set(df['pmid'])
    count = 0
    for pmid in pmids:
        temp_df = df.loc[df['pmid'] == pmid]
        indexes = list(temp_df.index)

        try:
            row1 = temp_df.iloc[0]
            row2 = temp_df.iloc[1]
        except:
            print(temp_df)
            print('There is only a single validated paper. Needs >=2')
            continue

        unique_col = row1 != row2
        cols = [colname for colname, unique_column in zip(
            df.columns, unique_col) if unique_column]

        if 'abstract' in cols and len(cols) == 1:
            continue

        elif filters == 'main' and any(unique_col):
            if headers[0] in cols or (headers[1] in cols and row1[headers[0]] == 1):
                df.drop([indexes[0], indexes[1]], inplace=True)
                count += 1

    print('%s rows dropped due to conflicts' % (count*2))
    return df


def correct_weird_results(df, df_type, headers, CORRECTIONS={}):
    '''
    Based on rows have a '2' in their results columns--meaning that the contents
    of that cell has an unknown value. These rows have been inspected manually to
    identify what that cell's value should have been and updated in the CORRECTIONS dictionary.

    CORRECTIONS = {
        df_type: {
            row_#: {
                header_#: value
            }
        }
    }
    '''

    for row in CORRECTIONS.get(df_type[0]):
        for header_index in CORRECTIONS[df_type[0]].get(row):
            header = headers[int(header_index)]
            value = CORRECTIONS[df_type[0]][row].get(header_index)

            df.at[int(row), header] = value

    return df


TERMS = {
    'F': 0,
    'T': 1,
    'FALSE': 0,
    'False': 0,
    'True': 1,
    'TRUE': 1,
    'No': 0,
    'No ': 0,
    ' No': 0,
    'Yes': 1,
    't': 1,
    'f': 0
}


def main(config, prediction_paths):
    with open(config, 'r') as file:
        info = json.load(file)

    final = {}
    all_df = {}

    print('Running using %s settings' % (config))
    for title in info:
        headers = info[title].get('headers')
        validation_path = info[title].get('validations')
        terms = info[title].get('terms')
        corrections = info[title].get('corrections')

        validation_df = create_validation_df(
            validation_path, terms, headers).reset_index(drop=True)
        if corrections:
            validation_df = correct_weird_results(
                validation_df, terms, headers, corrections)
        validation_df = remove_different_results(
            validation_df, headers, terms, 'main')

        # DEBUGGING:
        # for item in headers:
        #     print(validation_df[item].value_counts())

        # Adds the 'Prediction' column
        updated_df = add_multiple_predictions_to_df(
            prediction_paths, validation_df.copy())

        # Adds the 'Result' column
        final_df = add_final_result_to_df(
            updated_df.copy(), headers, terms)

        acc = {}
        for ml_type in final_df:
            if 'Result' in ml_type:
                temp = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                for value in final_df[ml_type]:
                    temp[value] += 1

                acc[ml_type] = temp

        final[title] = {'acc': acc}
        all_df[title] = final_df

    return final, all_df


# if __name__ == "__main__":
#     config = os.path.join('resources', 'config.json')
#     final_acc, final_all_df = main(config)

#     with open(os.path.join('out', 'saved_model_results', 'final_model_results.json'), 'w') as file:
#         json.dump(final_acc, file)
