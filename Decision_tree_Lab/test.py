from typing import no_type_check
import pandas as pd
import numpy as np

df = pd.read_csv("play_tennis.csv")
# print(df.head())

# df_new = df[df['play'] == 'No']
# print(df_new)

# print(df['play'])

def find_unique_values_of_df_column(column:df.columns):    
    unique = {}
    for item in column.values:
        if item in unique:
            unique[item] +=1
        else:
            unique[item] = 1
    return unique

def filter_df_only_containing_value_of_column(df, column, value):
    return df[df[column] == value]

myDict = find_unique_values_of_df_column(df, 'play')
# print(myDict)
filteredDict = filter_df_only_containing_value_of_column(df, 'play', 'No')
# print(filteredDict)
# print(list(df))

def create_values_map(df:pd.DataFrame):
    valuesMap = {}
    for column in list(df):
        valuesMap[column] = []
        data = df[column]
        for item in data.values:
            if not item in valuesMap[column]:
                valuesMap[column].append(item)
    print(valuesMap)

create_values_map(df)





