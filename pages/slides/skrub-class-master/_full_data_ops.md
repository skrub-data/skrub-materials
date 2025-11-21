
# What if this is not enough?? 

## What if...

- Your data is spread over multiple tables? 
- You want to avoid data leakage? 
- You want to tune more than just the hyperparameters of your model? 
- You want to guarantee that your pipeline is replayed exactly on new data? 

## 
When a normal pipe is not enough...

::: {.fragment style="font-size:2em;"}
... the `skrub` DataOps come to the rescue 🚒
:::


## DataOps...
- Extend the `scikit-learn` machinery to complex multi-table operations, and take care of data leakage
- Track all operations with a computational graph (a *Data Ops plan*)
- Allow tuning any operation in the data plan
- Can be persisted and shared easily 

## How do DataOps work, though?  {.smaller}
DataOps **wrap** around *user operations*, where user operations are:

- any dataframe operation (e.g., merge, group by, aggregate etc.)
- scikit-learn estimators (a Random Forest, RidgeCV etc.)
- custom user code (load data from a path, fetch from an URL etc.)

::: {.fragment}

::: {.callout-important}
DataOps _record_ user operations, so that they can later be _replayed_ in the same
order and with the same arguments on unseen data. 
:::
::: 

## DataOps, Plans, `learner`s: oh my!  
- A `DataOp` (singular) wraps a single operation, and can be combined and concatenated with other `DataOps`. 

- The **Data Ops** Plan is a collective name for the directed computational graph
that tracks a sequence and combination of `DataOps`. 

- The plan can be exported as a standalone object called `learner`. The `learner` 
works like a scikit-learn estimator that takes a dictionary of values rather 
than just `X` and `y`. 

## Starting with the `DataOps`

```{python code-line-numbers=5,6|8-}
#| echo: true
import skrub
data = skrub.datasets.fetch_credit_fraud()

baskets = skrub.var("baskets", data.baskets)
products = skrub.var("products", data.products) # add a new variable

X = baskets[["ID"]].skb.mark_as_X()
y = baskets["fraud_flag"].skb.mark_as_y()
```

:::{.incremental}
- `X`, `y`, `products` represent inputs to the pipeline.
- `skrub` splits `X` and `y` when training. 
:::

##  Building a full data plan
```{python}
#| echo: true
from skrub import selectors as s

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.StringEncoder()
)
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
```

##  Building a full data plan {auto-animate="true"}
```{python}
#| echo: true
aggregated_products = vectorized_products.groupby(
    "basket_ID"
).agg("mean").reset_index()

features = X.merge(aggregated_products, left_on="ID", right_on="basket_ID")
features = features.drop(columns=["ID", "basket_ID"])
```

##  Building a full data plan {auto-animate="true"}
```{python}
#| echo: true
from sklearn.ensemble import ExtraTreesClassifier  
predictions = features.skb.apply(
    ExtraTreesClassifier(n_jobs=-1), y=y
)
```

## Inspecting the data plan
```{.python}
predictions.skb.full_report()
```
<br/>

<a href="dataop_report/index.html" target="_blank">Execution report</a>

Each node:

- Shows a preview of the data resulting from the operation
- Reports the location in the code where the code is defined
- Shows the run time of the node (in the next release)

## Exporting the plan in a `learner` {.smaller}
The data plan can be exported as a `learner`:
```{python}
#| echo: true
# anywhere
learner = predictions.skb.make_learner(fitted=True)
```

::: {.fragment}
Then, the `learner` can be pickled ...

```{python}
import pickle 

learner_bytes = pickle.dumps(learner)
```

```{.python}
import pickle

with open("learner.bin", "wb") as fp:
    pickle.dump(learner, fp)
```
:::

::: {.fragment}

... loaded ...

```{python}
loaded_learner = pickle.loads(learner_bytes)
```

```{.python}
with open("learner.bin", "rb") as fp:
    loaded_learner = pickle.load(fp)
```
:::

::: {.fragment}
... and applied to new data:
```{python}
#| echo: true
data = skrub.datasets.fetch_credit_fraud(split="test")
new_baskets = data.baskets
new_products = data.products
loaded_learner.predict({"baskets": new_baskets, "products": new_products})
```
:::


## Hyperparameter tuning in a Data Plan 
`skrub` implements four `choose_*` functions:

- `choose_from`: select from the given list of options
- `choose_int`: select an integer within a range
- `choose_float`: select a float within a range
- `choose_bool`: select a bool 
- `optional`: chooses whether to execute the given operation


## Tuning in `scikit-learn` can be complex {.smaller auto-animate="true"}

```{.python}
pipe = Pipeline([("dim_reduction", PCA()), ("regressor", Ridge())])
grid = [
    {
        "dim_reduction": [PCA()],
        "dim_reduction__n_components": [10, 20, 30],
        "regressor": [Ridge()],
        "regressor__alpha": loguniform(0.1, 10.0),
    },
    {
        "dim_reduction": [SelectKBest()],
        "dim_reduction__k": [10, 20, 30],
        "regressor": [Ridge()],
        "regressor__alpha": loguniform(0.1, 10.0),
    },
    {
        "dim_reduction": [PCA()],
        "dim_reduction__n_components": [10, 20, 30],
        "regressor": [RandomForestClassifier()],
        "regressor__n_estimators": loguniform(20, 200),
    },
    {
        "dim_reduction": [SelectKBest()],
        "dim_reduction__k": [10, 20, 30],
        "regressor": [RandomForestClassifier()],
        "regressor__n_estimators": loguniform(20, 200),
    },
]
model = RandomizedSearchCV(pipe, grid)
```
## Tuning with `DataOps` is simple! {.smaller} 

```python
dim_reduction = X.skb.apply(
    skrub.choose_from(
        {
            "PCA": PCA(n_components=skrub.choose_int(10, 30)),
            "SelectKBest": SelectKBest(k=skrub.choose_int(10, 30))
        }, name="dim_reduction"
    )
)
regressor = dim_reduction.skb.apply(
    skrub.choose_from(
        {
            "Ridge": Ridge(alpha=skrub.choose_float(0.1, 10.0, log=True)),
            "RandomForest": RandomForestClassifier(
                n_estimators=skrub.choose_int(20, 200, log=True)
            )
        }, name="regressor"
    )
)
search = regressor.skb.make_randomized_search(scoring="roc_auc", fitted=True)
```

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

## Run hyperparameter search
```{.python}
# fit the search 
search = regressor.skb.make_randomized_search(scoring="roc_auc", fitted=True, cv=5)

# save the best learner
best_learner = search.best_learner_
```

## Observe the impact of the hyperparameters {auto-animate="true" .smaller} 
Data Ops provide a built-in parallel coordinate plot. 

```{.python}
search = pred.skb.get_randomized_search(fitted=True)
search.plot_parallel_coord()
```
```{python}
from plotly.io import read_json

fig = read_json("parallel_coordinates_hgbr.json")
fig.update_layout(margin=dict(l=200))
```


[source](https://skrub-data.org/EuroSciPy2025/content/notebooks/single_horizon_prediction.html)
