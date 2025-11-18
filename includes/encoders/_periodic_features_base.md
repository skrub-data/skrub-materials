
## Adding periodic features with pandas/polars{.smaller}
:::{.panel-tabset}

### Pandas
```{python}
import pandas as pd
data = {
    'date': ['2023-01-01 12:34:56', '2023-02-15 08:45:23', '2023-03-20 18:12:45'],
    'value': [10, 20, 30]
}
```
```{python}
#| echo: true
df_pd['hour_sin'] = np.sin(2 * np.pi * df_pd['hour'] / 24)
df_pd['hour_cos'] = np.cos(2 * np.pi * df_pd['hour'] / 24)

df_pd['month_sin'] = np.sin(2 * np.pi * df_pd['month'] / 12)
df_pd['month_cos'] = np.cos(2 * np.pi * df_pd['month'] / 12)
```

### Polars
```{python}
#| echo: true
df_pl = df_pl.with_columns(
    hour_sin = np.sin(2 * np.pi * pl.col("hour") / 24),
    hour_cos = np.cos(2 * np.pi * pl.col("hour") / 24),
    
    month_sin = np.sin(2 * np.pi * pl.col("month") / 12),
    month_cos = np.cos(2 * np.pi * pl.col("month") / 12),
)
```

:::
