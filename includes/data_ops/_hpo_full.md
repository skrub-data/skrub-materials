## Hyperparameter tuning in a Data Ops plan 
Skrub implements four `choose_*` functions:

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
        "regressor": [RandomForestRegressor()],
        "regressor__n_estimators": loguniform(20, 200),
    },
    {
        "dim_reduction": [SelectKBest()],
        "dim_reduction__k": [10, 20, 30],
        "regressor": [RandomForestRegressor()],
        "regressor__n_estimators": loguniform(20, 200),
    },
]
```
## Tuning with Data Ops is simple! {.smaller} 

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
            "RandomForest": RandomForestRegressor(
                n_estimators=skrub.choose_int(20, 200, log=True)
            )
        }, name="regressor"
    )
)
```

## Run hyperparameter search
```{.python}
# fit the search 
search = regressor.skb.make_randomized_search(
    scoring="roc_auc", fitted=True, cv=5
)

# save the best learner
best_learner = search.best_learner_
```