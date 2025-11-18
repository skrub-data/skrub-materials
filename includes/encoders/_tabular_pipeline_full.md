## Let's build a predictive pipeline {auto-animate="true"}
```{.python}
from sklearn.linear_model import Ridge
model = Ridge()
```

## Let's build a predictive pipeline {auto-animate="true" visibility="uncounted"}
```{.python code-line-numbers="3-6"}
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

model = make_pipeline(StandardScaler(), SimpleImputer(), Ridge())
```


## Let's build a predictive pipeline {auto-animate="true" visibility="uncounted"}
```{.python code-line-numbers="3,5,6-17|"}
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
## Now with `tabular_pipeline` {auto-animate="true" .smaller}
```{python}
#| echo: true
import skrub
from sklearn.linear_model import Ridge
model = skrub.tabular_pipeline(Ridge())
```

![](images/skrub-tabular-pipeline-linear-model.png){fig-align="center"}

## 
![](images/drakeno.png){fig-align="center"}
