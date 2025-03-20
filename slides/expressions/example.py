import skrub
from skrub import datasets
from skrub import selectors as s

data = datasets.fetch_credit_fraud()
baskets_df = data.baskets
products_df = data.products

# %%
products = skrub.var("products", placeholder=products_df)

# %%
total_price = products["Nbr_of_prod_purchas"] * products["cash_price"]
products = products.assign(total_price=total_price)

min_hash = skrub.MinHashEncoder(n_components=2)

product_strings = (
    products.skb.select("basket_ID" | s.string())
    .skb.apply(min_hash, cols=s.all() - "basket_ID")
    .groupby(by="basket_ID")
    .agg("min")
    .reset_index()
)
product_strings.skb.get_report().open()

# %%
product_numbers = (
    products.skb.select("basket_ID" | s.numeric())
    .groupby(by="basket_ID")
    .agg("sum")
    .reset_index()
)
product_numbers.skb.get_report().open()

# %%
baskets = skrub.var("baskets", placeholder=baskets_df)
fraud_flag = baskets["fraud_flag"].skb.mark_as_y()
baskets = baskets.drop("fraud_flag", axis=1, errors='ignore').skb.mark_as_x()
baskets = (
    baskets.merge(product_strings, left_on="ID", right_on="basket_ID")
    .drop("basket_ID", axis=1)
    .merge(product_numbers, left_on="ID", right_on="basket_ID")
    .drop(["ID", "basket_ID"], axis=1)
)
baskets.skb.get_report().open()

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

classifier = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.001, 2.0, log=True, name="learning rate")
)

prediction = baskets.skb.apply(classifier, y=fraud_flag)

# %%
prediction.skb.full_report()

# %%
cv_result = prediction.skb.cross_validate(scoring="roc_auc", n_jobs=8, verbose=3)
print(cv_result)
