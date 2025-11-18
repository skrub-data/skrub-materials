
## Encoding datetime features with `skrub.DatetimeEncoder` {auto-animate="true" visibility="uncounted" .smaller}
```{python}
import polars as pl
data = {
    'date': ['2023-01-01 12:34:56', '2023-02-15 08:45:23', '2023-03-20 18:12:45'],
    'value': [10, 20, 30]
}
df = pl.DataFrame(data)
```
```{python}
#| echo: true
from skrub import DatetimeEncoder, ToDatetime

X_date = ToDatetime().fit_transform(df["date"])
de = DatetimeEncoder(resolution="second")
# de = DatetimeEncoder(periodic_encoding="spline")
X_enc = de.fit_transform(X_date)
print(X_enc)
```