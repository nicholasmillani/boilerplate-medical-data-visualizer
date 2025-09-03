import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Import data
df = pd.read_csv('medical_examination.csv')

# 2 - Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3 - Normalize cholesterol and gluc values
df['cholesterol'] = df['cholesterol'].replace({1: 0, 2: 1, 3: 1})
df['gluc'] = df['gluc'].replace({1: 0, 2: 1, 3: 1})

# 4 - Categorical Plot
def draw_cat_plot():
    # 5 & 6 - Melt DataFrame
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 7 - Group and reformat
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 8 - Draw the catplot
    fig = sns.catplot(
        x="variable", y="total", hue="value", col="cardio",
        data=df_cat, kind="bar"
    ).fig

    # 9 - Save
    fig.savefig('catplot.png')
    return fig

# 10 - Heat Map
def draw_heat_map():
    # 11 - Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 - Correlation matrix
    corr = df_heat.corr()

    # 13 - Generate mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15 - Draw the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )

    # 16 - Save
    fig.savefig('heatmap.png')
    return fig
