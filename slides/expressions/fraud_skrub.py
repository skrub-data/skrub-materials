from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
from skrub import selectors as s
import skrub.datasets

data = skrub.datasets.fetch_credit_fraud()

X = skrub.X(data.baskets[["ID"]])
y = skrub.y(data.baskets["fraud_flag"])
products = skrub.var("products", data.products)

products = products[products["basket_ID"].isin(X["ID"])]

product_vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.StringEncoder(n_components=5)
)
vectorized_products = products.skb.apply(
    product_vectorizer, cols=s.all() - "basket_ID"
)

aggregated_products = (
    vectorized_products.groupby("basket_ID").agg("mean").reset_index()
)
X = X.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
    columns=["ID", "basket_ID"]
)

classifier = HistGradientBoostingClassifier()
pred = X.skb.apply(classifier, y=y)

pred.skb.cross_validate(scoring="roc_auc", n_jobs=5)

pred.skb.full_report(output_dir="expression_report", overwrite=True)


# %%
import pickle

estimator = pred.skb.get_estimator(fitted=True)
stored = pickle.dumps(estimator)
loaded = pickle.loads(stored)

print(
    loaded.predict(
        {"X": data.baskets.drop(columns="fraud_flag"), "products": data.products}
    )
)
