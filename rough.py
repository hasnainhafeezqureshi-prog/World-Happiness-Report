import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD & STANDARDIZE

df_2015 = pd.read_csv('2015.csv')
df_2016 = pd.read_csv('2016.csv')
df_2017 = pd.read_csv('2017.csv')
df_2018 = pd.read_csv('2018.csv')
df_2019 = pd.read_csv('2019.csv')


df_2015 = df_2015.rename(
    columns={
        'Country': 'Country',
        'Region': 'Region',
        'Happiness Score': 'Happiness_Score',
        'Economy (GDP per Capita)': 'GDP',
        'Family': 'Social_Support',      
        'Health (Life Expectancy)': 'Health',
        'Freedom': 'Freedom',
        'Trust (Government Corruption)': 'Corruption',
    }
)

df_2016 = df_2016.rename(
    columns={
        'Country': 'Country',
        'Region': 'Region',
        'Happiness Score': 'Happiness_Score',
        'Economy (GDP per Capita)': 'GDP',
        'Family': 'Social_Support',       
        'Health (Life Expectancy)': 'Health',
        'Freedom': 'Freedom',
        'Trust (Government Corruption)': 'Corruption',
    }
)

df_2017 = df_2017.rename(
    columns={
        'Country': 'Country',
        'Happiness.Score': 'Happiness_Score',
        'Economy..GDP.per.Capita.': 'GDP',
        'Family': 'Social_Support',      
        'Health..Life.Expectancy.': 'Health',
        'Freedom': 'Freedom',
        'Trust..Government.Corruption.': 'Corruption',
    }
)

df_2018 = df_2018.rename(
    columns={
        'Country or region': 'Country',
        'Score': 'Happiness_Score',
        'GDP per capita': 'GDP',
        'Social support': 'Social_Support',
        'Healthy life expectancy': 'Health',
        'Freedom to make life choices': 'Freedom',
        'Perceptions of corruption': 'Corruption',
    }
)

df_2019 = df_2019.rename(
    columns={
        'Country or region': 'Country',
        'Score': 'Happiness_Score',
        'GDP per capita': 'GDP',
        'Social support': 'Social_Support',
        'Healthy life expectancy': 'Health',
        'Freedom to make life choices': 'Freedom',
        'Perceptions of corruption': 'Corruption',
    }
)

df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017
df_2018['Year'] = 2018
df_2019['Year'] = 2019

# CONCATENATE VERTICALLY

concat_df = pd.concat([df_2015, df_2016, df_2017, df_2018, df_2019], ignore_index=True)
target_columns = ['Country', 'Region', 'Happiness_Score', 'GDP', 'Social_Support', 'Health', 'Freedom', 'Corruption', 'Year']
concat_df = concat_df[target_columns]

# FILLING NANs IN REGION COLUMN WITH POSSIBLE VALUES

initial_mask = (concat_df['Year'] == 2015) | (concat_df['Year'] == 2016)
two_cols = concat_df[initial_mask][['Country', 'Region']]
two_cols = two_cols.drop_duplicates()
key_values = two_cols.set_index('Country')['Region'].to_dict()

final_mask = (concat_df['Year'] == 2017) | (concat_df['Year'] == 2018) | (concat_df['Year'] == 2019)
concat_df.loc[final_mask, 'Region'] = concat_df.loc[final_mask, 'Country'].map(key_values)

concat_df = concat_df.dropna()                          # DROPS FEW ROWS VALUE IN KEY COLUMN WAS EMPTY

melted_df = concat_df.melt(
    id_vars = ['Country', 'Region', 'Happiness_Score', 'Year'],
    value_vars = ['GDP', 'Social_Support', 'Health', 'Freedom', 'Corruption'],
    var_name = 'Feature',
    value_name = 'Value'
)

melted_df['Region'] = melted_df['Region'].astype('category')
print(melted_df.isnull().sum())

# NUMPY ANALYSIS
















