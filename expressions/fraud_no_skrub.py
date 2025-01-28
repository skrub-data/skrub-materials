from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import skrub
import skrub.datasets

data = skrub.datasets.fetch_credit_fraud()

X = data.baskets[["ID"]]
y = data.baskets["fraud_flag"]
products = data.products

product_vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.StringEncoder(n_components=5)
)
vectorized_products = product_vectorizer.fit_transform(products)

aggregated_products = (
    vectorized_products.groupby("basket_ID").agg("mean").reset_index()
)
X = X.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
    columns=["ID", "basket_ID"]
)

classifier = HistGradientBoostingClassifier()

cross_val_score(classifier, X, y, scoring="roc_auc", n_jobs=5)
