import skrub
from skrub import selectors as s
import skrub.datasets

data = skrub.datasets.fetch_credit_fraud()

X = skrub.X(data.baskets[["ID"]])
y = skrub.y(data.baskets["fraud_flag"])
products = skrub.var("products", data.products)
