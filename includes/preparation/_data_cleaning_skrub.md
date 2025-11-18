## Data cleaning with `skrub.Cleaner` {.smaller auto-animate="true"}

:::{.panel-tabset}
### Pandas
```{python}
#| echo: true
from skrub import Cleaner
cleaner = Cleaner(drop_if_constant=True, datetime_format='%d %b %Y')
df_cleaned = cleaner.fit_transform(df_pd)
display(df_cleaned)
```

### Polars
```{python}
#| echo: true
from skrub import Cleaner
cleaner = Cleaner(drop_if_constant=True, datetime_format='%d %b %Y')
df_cleaned = cleaner.fit_transform(df_pl)
display(df_cleaned)
```

:::