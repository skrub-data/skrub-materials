
## Tuning with `DataOps` is not limited to estimators
::: {.panel-tabset}
### Pandas
```{python}
import pandas as pd
import skrub
```
```{python}
#| echo: true
df = pd.DataFrame(
    {"subject": ["math", "math", "art", "history"], "grade": [10, 8, 4, 6]}
)

df_do = skrub.var("grades", df)

agg_grades = df_do.groupby("subject").agg(skrub.choose_from(["count", "mean"]))
agg_grades.skb.describe_param_grid()
```

### Polars
```{python}
import polars as pl
import skrub
```

```{python}
#| echo: true
df = pl.DataFrame(
    {"subject": ["math", "math", "art", "history"], "grade": [10, 8, 4, 6]}
)

df_do = skrub.var("grades", df)

agg_grades = df_do.group_by("subject").agg(
    skrub.choose_from([pl.mean("grade"), pl.count("grade")])
)
agg_grades.skb.describe_param_grid()
```

:::
