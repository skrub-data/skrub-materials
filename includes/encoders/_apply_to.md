## Column transformations with `ApplyToCols` and `ApplyToFrame` {.smaller auto-animate="true"}

![](images/ApplyToCols.png){fig-align="center"}

## Column transformations with `ApplyToCols` and `ApplyToFrame` {.smaller auto-animate="true"}

![](images/ApplyToFrame.png){fig-align="center"}

## Replacing `ColumnTransformer` with `ApplyToCols` {.smaller auto-animate="true"}

```{python}
#| echo: true
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.DataFrame({"text": ["foo", "bar", "baz"], "number": [1, 2, 3]})

categorical_columns = selector(dtype_include=object)(df)
numerical_columns = selector(dtype_exclude=object)(df)

ct = make_column_transformer(
      (StandardScaler(),
       numerical_columns),
      (OneHotEncoder(handle_unknown="ignore"),
       categorical_columns))
transformed = ct.fit_transform(df)
transformed
```

## Replacing `ColumnTransformer` with `ApplyToCols` {.smaller auto-animate="true"}
```{python}
#| echo: true
import skrub.selectors as s
from sklearn.pipeline import make_pipeline
from skrub import ApplyToCols

numeric = ApplyToCols(StandardScaler(), cols=s.numeric())
string = ApplyToCols(OneHotEncoder(sparse_output=False), cols=s.string())

transformed = make_pipeline(numeric, string).fit_transform(df)
transformed
```

