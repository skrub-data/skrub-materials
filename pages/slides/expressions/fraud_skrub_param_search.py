from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import skrub
from skrub import selectors as s
from skrub import datasets

data = datasets.fetch_credit_fraud()

X = skrub.X(data.baskets[["ID"]])
y = skrub.y(data.baskets["fraud_flag"])
products = skrub.var("products", data.products)

products = products[products["basket_ID"].isin(X["ID"])]

n_components = skrub.choose_int(2, 50, name="n_components")
encoder = skrub.choose_from(
    {
        "lse": skrub.StringEncoder(n_components=n_components),
        "minhash": skrub.MinHashEncoder(n_components=n_components),
    },
    name="encoder",
)
product_vectorizer = skrub.TableVectorizer(high_cardinality=encoder)
vectorized_products = products.skb.apply(
    product_vectorizer, cols=s.all() - "basket_ID"
)

aggregated_products = (
    vectorized_products.groupby("basket_ID")
    .agg(skrub.choose_from(["min", "mean", "sum"], name="agg function"))
    .reset_index()
)
X = X.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
    columns=["ID", "basket_ID"]
)

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.001, 0.8, log=True, name="learning_rate"),
    max_leaf_nodes=skrub.choose_int(2, 255, log=True, name="max_leaf_nodes"),
)
logistic = LogisticRegression(C=skrub.choose_float(0.01, 100, log=True, name="C"))
pred = X.skb.apply(
    skrub.choose_from({"hgb": hgb, "logistic": logistic}, name="classifier"), y=y
)

search = pred.skb.get_randomized_search(
    scoring="roc_auc", n_iter=64, n_jobs=16, cv=4, verbose=3, fitted=True
)


fig = search.plot_parallel_coord()
fig.show()
fig.write_html("parallel_coord.html")
