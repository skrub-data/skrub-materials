
## Data cleaning with pandas/polars: setup {.smaller auto-animate="true"}

::: {.panel-tabset}

### Pandas
```{python}
# | echo: true
import pandas as pd
import numpy as np

data = {
    "Int": [2, 3, 2],  # Multiple unique values
    "Const str": ["x", "x", "x"],  # Single unique value
    "Str": ["foo", "bar", "baz"],  # Multiple unique values
    "All nan": [np.nan, np.nan, np.nan],  # All missing values
    "All empty": ["", "", ""],  # All empty strings
    "Date": ["01 Jan 2023", "02 Jan 2023", "03 Jan 2023"],
}

df_pd = pd.DataFrame(data)
display(df_pd)
```

### Polars
```{python}
#| echo: true
import polars as pl
import numpy as np
data = {
    "Int": [2, 3, 2],  # Multiple unique values
    "Const str": ["x", "x", "x"],  # Single unique value
    "Str": ["foo", "bar", "baz"],  # Multiple unique values
    "All nan": [np.nan, np.nan, np.nan],  # All missing values
    "All empty": ["", "", ""],  # All empty strings
    "Date": ["01 Jan 2023", "02 Jan 2023", "03 Jan 2023"],
}

df_pl = pl.DataFrame(data)
display(df_pl)
```

:::


## Nulls, datetimes, constant columns with pandas/polars {.smaller auto-animate="true"}

:::{.panel-tabset}
### Pandas
```{python}
#| echo: true
# Parse the datetime strings with a specific format
df_pd['Date'] = pd.to_datetime(df_pd['Date'], format='%d %b %Y')

# Drop columns with only a single unique value
df_pd_cleaned = df_pd.loc[:, df_pd.nunique(dropna=True) > 1]

# Function to drop columns with only missing values or empty strings
def drop_empty_columns(df):
    # Drop columns with only missing values
    df_cleaned = df.dropna(axis=1, how='all')
    # Drop columns with only empty strings
    empty_string_cols = df_cleaned.columns[df_cleaned.eq('').all()]
    df_cleaned = df_cleaned.drop(columns=empty_string_cols)
    return df_cleaned

# Apply the function to the DataFrame
df_pd_cleaned = drop_empty_columns(df_pd_cleaned)
```

### Polars
```{python}
#| echo: true 
# Parse the datetime strings with a specific format
df_pl = df_pl.with_columns([
    pl.col("Date").str.strptime(pl.Date, "%d %b %Y", strict=False).alias("Date")
])

# Drop columns with only a single unique value
df_pl_cleaned = df_pl.select([
    col for col in df_pl.columns if df_pl[col].n_unique() > 1
])

# Import selectors for dtype selection
import polars.selectors as cs

# Drop columns with only missing values or only empty strings
def drop_empty_columns(df):
    all_nan = df.select(
        [
            col for col in df.select(cs.numeric()).columns if 
            df [col].is_nan().all()
        ]
    ).columns
    
    all_empty = df.select(
        [
            col for col in df.select(cs.string()).columns if 
            (df[col].str.strip_chars().str.len_chars()==0).all()
        ]
    ).columns

    to_drop = all_nan + all_empty

    return df.drop(to_drop)

df_pl_cleaned = drop_empty_columns(df_pl_cleaned)
``` 

:::
