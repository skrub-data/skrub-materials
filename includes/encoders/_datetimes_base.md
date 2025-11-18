## Encoding datetime features with pandas/polars {.smaller}
:::{.panel-tabset}

### Pandas
```{python}
#|echo: true
import pandas as pd
data = {
    'date': ['2023-01-01 12:34:56', '2023-02-15 08:45:23', '2023-03-20 18:12:45'],
    'value': [10, 20, 30]
}
df_pd = pd.DataFrame(data)
datetime_column = "date"
df_pd[datetime_column] = pd.to_datetime(df_pd[datetime_column], errors='coerce')

df_pd['year'] = df_pd[datetime_column].dt.year
df_pd['month'] = df_pd[datetime_column].dt.month
df_pd['day'] = df_pd[datetime_column].dt.day
df_pd['hour'] = df_pd[datetime_column].dt.hour
df_pd['minute'] = df_pd[datetime_column].dt.minute
df_pd['second'] = df_pd[datetime_column].dt.second
```

### Polars
```{python}
#|echo: true
import polars as pl
data = {
    'date': ['2023-01-01 12:34:56', '2023-02-15 08:45:23', '2023-03-20 18:12:45'],
    'value': [10, 20, 30]
}
df_pl = pl.DataFrame(data)
df_pl = df_pl.with_columns(date=pl.col("date").str.to_datetime())

df_pl = df_pl.with_columns(
    year=pl.col("date").dt.year(),
    month=pl.col("date").dt.month(),
    day=pl.col("date").dt.day(),
    hour=pl.col("date").dt.hour(),
    minute=pl.col("date").dt.minute(),
    second=pl.col("date").dt.second(),
)
```

:::
