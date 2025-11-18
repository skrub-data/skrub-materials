## Build a predictive pipeline {auto-animate="true"}
```{.python}
from sklearn.linear_model import Ridge
model = Ridge()
```
## Build a predictive pipeline {auto-animate="true" visibility="uncounted"}
```{.python}
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer

categorical_columns = selector(dtype_include=object)(employees)
numerical_columns = selector(dtype_exclude=object)(employees)

ct = make_column_transformer(
      (StandardScaler(),
       numerical_columns),
      (OneHotEncoder(handle_unknown="ignore"),
       categorical_columns))

model = make_pipeline(ct, SimpleImputer(), Ridge())
```
## Build a predictive pipeline with `tabular_pipeline` {auto-animate="true" .smaller}
```{.python}
import skrub
from sklearn.linear_model import Ridge
model = skrub.tabular_pipeline(Ridge())
```

![](images/skrub-tabular-pipeline-linear-model.png){fig-align="center"}

## 
![](/images/drakeno.png){fig-align="center"}
