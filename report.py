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

concat_df = concat_df.dropna(subset=['Happiness_Score', 'GDP', 'Social_Support', 'Health', 'Freedom', 'Corruption'])                          # DROPS FEW ROWS VALUE IN KEY COLUMN WAS EMPTY

melted_df = concat_df.melt(
    id_vars = ['Country', 'Region', 'Happiness_Score', 'Year'],
    value_vars = ['GDP', 'Social_Support', 'Health', 'Freedom', 'Corruption'],
    var_name = 'Feature',
    value_name = 'Value'
)

melted_df['Region'] = melted_df['Region'].astype('category')

# NUMPY ANALYSIS
operations_df = melted_df.groupby('Region')['Happiness_Score'].agg(['mean', 'std', 'var'])
print(operations_df)

concat_df['Happiness_Score_Normalized'] = (concat_df['Happiness_Score'] - concat_df['Happiness_Score'].mean())/concat_df['Happiness_Score'].std()

# USED concat_df BECUASE melted_df CONTAINS DUPLICATE COUNTRIES

recent_year_df = concat_df[concat_df['Year'] == 2019]
score_arr = recent_year_df['Happiness_Score'].to_numpy()
sorting_indices = score_arr.argsort()[::-1]
country_arr = recent_year_df['Country'].to_numpy()
sorted_countries = country_arr[sorting_indices]

# VISUALIZATION

# Group by Region, getting the mean, sorting them, and grabbing the top 5 names
top_5_region_names = concat_df.groupby('Region')['Happiness_Score'].mean().nlargest(5).index.tolist()
filtered_df = concat_df[concat_df['Region'].isin(top_5_region_names)]
yearly_regional_means = filtered_df.groupby(['Region', 'Year'])['Happiness_Score'].mean().unstack(level=0)

years = yearly_regional_means.index.tolist()
y_data = [yearly_regional_means[col].tolist() for col in yearly_regional_means.columns]

plt.figure(figsize=(10,6))
plt.stackplot(years, y_data, labels=yearly_regional_means.columns)
plt.xticks([2015, 2016, 2017, 2018, 2019])
plt.xlabel('Years')
plt.ylabel('Happiness Score')
plt.title('Happiness Score For Five years Across Five Regions')
plt.legend(loc='upper left')
plt.show()


plt.figure(figsize=(8,4))
plt.scatter(concat_df['GDP'], concat_df['Happiness_Score'], c=concat_df['Happiness_Score'], cmap='viridis')
plt.colorbar()
plt.xlabel('GDP')
plt.ylabel('Happiness Score')
plt.title('GDP vs Happiness Score')
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(data=concat_df, x='Region', y='Happiness_Score', color='grey')
plt.xticks(rotation = 45, ha = 'right')
plt.title('Region vs Happiness Score')
plt.tight_layout()
plt.show()


custom_colors = {
    'GDP': '#4E220F',
    'Freedom': '#112E81',
    'Health': '#1A312C',
    'Corruption': '#077A7D',
    'Social_Support': '#8A7650'
}

plt.figure(figsize=(10,6))
sns.barplot(data=melted_df, x='Region', y='Value', hue='Feature', palette=custom_colors)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper right')
plt.title('Average Factor Contribution')
plt.tight_layout()
plt.show()