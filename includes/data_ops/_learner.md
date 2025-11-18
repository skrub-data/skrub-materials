## Exporting the plan in a `learner` {.smaller}
The **Learner** is a stand-alone object that works like
a scikit-learn estimator that takes a dictionary as input rather
than just `X` and `y`. 


::: {.fragment}

```{python}
#| echo: true
learner = predictions.skb.make_learner(fitted=True)
```

:::


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
```{python}
loaded_learner = pickle.loads(learner_bytes)
```

:::

::: {.fragment}
... loaded and applied to new data:

```{.python}
with open("learner.bin", "rb") as fp:
    loaded_learner = pickle.load(fp)
data = skrub.datasets.fetch_credit_fraud(split="test")
new_baskets = data.baskets
new_products = data.products
loaded_learner.predict({"baskets": new_baskets, "products": new_products})
```
```{python}
data = skrub.datasets.fetch_credit_fraud(split="test")
new_baskets = data.baskets
new_products = data.products
loaded_learner.predict(
    {"baskets": new_baskets, "products": new_products}
)
```
:::

