## Starting with the `DataOps` {auto-animate="true"} 

```{python}
#| echo: true
import skrub
data = skrub.datasets.fetch_credit_fraud()

baskets = skrub.var("baskets", data.baskets)
products = skrub.var("products", data.products) # add a new variable

X = baskets[["ID"]].skb.mark_as_X()
y = baskets["fraud_flag"].skb.mark_as_y()
```

- `baskets` and `products` represent inputs to the pipeline.
- Skrub tracks `X` and `y` so that training and test splits are never mixed. 

## Applying a transformer {auto-animate="true"}
```{python code-line-numbers="10-18|"}
# | echo: true
from skrub import selectors as s

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.StringEncoder()
)
vectorized_products = products.skb.apply(
    vectorizer, cols=s.all() - "basket_ID"
)
```

##  Executing dataframe operations {auto-animate="true"}
```{python}
#| echo: true
aggregated_products = vectorized_products.groupby(
    "basket_ID"
).agg("mean").reset_index()

features = X.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
)
features = features.drop(columns=["ID", "basket_ID"])
```

##  Applying a ML model {auto-animate="true"}
```{python}
#| echo: true
from sklearn.ensemble import ExtraTreesClassifier  
predictions = features.skb.apply(
    ExtraTreesClassifier(n_jobs=-1), y=y
)
```
