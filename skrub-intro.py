# %% [markdown]
# ---
# title: "Skrub"
# format:
#   html:
#     code-fold: false
# jupyter: python3
# ---

# %% [markdown]
#
# [skrub-data.org](https://skrub-data.org/)
#
# **Less wrangling, more machine learning**
#
# # Machine learning and tabular data
#
# - ML expects numeric arrays
# - Real data is more complex:
#   - Multiple tables
#   - Dates, categories, text, locations, …
#
# **Skrub: bridge the gap between dataframes and scikit-learn.**
#
# ## Skrub helps at several stages of a tabular learning project
#
# 1. What's in the data? (EDA)
# 1. Can we learn anything? (baselines)
# 1. How do I represent the data? (feature extraction)
# 1. How do I bring it all together? (building a pipeline)
#
# # 1. What's in the data?

# %%
import skrub
from skrub import datasets

employees = datasets.fetch_employee_salaries().X
employees.iloc[0]

# %% [markdown]
#
# ## `TableReport`: interactive display of a dataframe

# %%
skrub.TableReport(employees, verbose=0)

# %% [markdown]
# We can tell skrub to patch the default display of polars and pandas dataframes.

# %%
skrub.patch_display(verbose=0)

# %% [markdown]
# # 2. Can we learn anything?

# %%
employee_salaries = datasets.fetch_employee_salaries()
X, y = employee_salaries.X, employee_salaries.y

# %% [markdown]
# ## `tabular_learner`: a pre-made robust baseline

# %%
learner = skrub.tabular_learner("regressor")
learner

# %%
from sklearn.model_selection import cross_val_score

cross_val_score(learner, X, y, scoring="r2")

# %% [markdown]
# The `tabular_learner` adapts to the supervised estimator we choose

# %%
from sklearn.linear_model import Ridge

learner = skrub.tabular_learner(Ridge())
learner

# %%
cross_val_score(learner, X, y, scoring="r2")

# %% [markdown]
# # 3. How do I represent the data?
#
# Skrub helps extract informative features from tabular data.
#
# ## `TableVectorizer`: apply an appropriate transformer to each column

# %%
vectorizer = skrub.TableVectorizer()
transformed = vectorizer.fit_transform(X)

# %% [markdown]
# The `TableVectorizer` identifies several kinds of columns:
#
# - categorical, low cardinality
# - categorical, high cardinality
# - datetime
# - numeric
# - ... we may add more

# %%
from pprint import pprint

pprint(vectorizer.column_to_kind_)

# %% [markdown]
# For each kind, it applies an appropriate transformer

# %%
vectorizer.transformers_["department"]  # low-cardinality categorical

# %%
vectorizer.transformers_["employee_position_title"]  # high-cardinality categorical

# %%
vectorizer.transformers_["date_first_hired"]  # datetime

# %% [markdown]
# ... and those transformers turn the input into numeric features that can be used for ML

# %%
transformed[vectorizer.input_to_outputs_["date_first_hired"]]

# %% [markdown]
# For high-cardinality categorical columns the default `GapEncoder` identifies
# sparse topics (more later).

# %%
transformed[vectorizer.input_to_outputs_["employee_position_title"]]

# %% [markdown]
# The transformer used for each column kind can be easily configured.

# %% [markdown]
# ## Preprocessing in the `TableVectorizer`
#
# The `TableVectorizer` actually performs a lot of preprocessing before
# applying the final transformers, such as:
#
# - ensuring consistent column names
# - detecting missing values such as `"N/A"`
# - dropping empty columns
# - handling pandas dtypes -- `float64`, `nan` vs `Float64`, `NA`
# - parsing numbers
# - parsing dates, ensuring consistent dtype and timezone
# - converting numbers to float32 for faster computation & less memory downstream
# - ...

# %%
pprint(vectorizer.all_processing_steps_["date_first_hired"])

# %% [markdown]
# ## Extracting good features
#
# Skrub offers several encoders to extract features from different columns.
# In particular from categorical columns.
#
# ### `GapEncoder`
#
# Categories are somewhere between text and an enumeration...
# The `GapEncoder` is somewhere between a topic model and a one-hot encoder!

# %%
import seaborn as sns
from matplotlib import pyplot as plt

gap = skrub.GapEncoder()
pos_title = X["employee_position_title"]
loadings = gap.fit_transform(pos_title).set_index(pos_title.values).head()

loadings.columns = [c.split(": ")[1] for c in loadings.columns]
sns.heatmap(loadings)
_ = plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")


# %% [markdown]
# ### `TextEncoder`
#
# Extract embeddings from a text column using any model from the HuggingFace Hub.

# %%
import pandas as pd

X = pd.Series(["airport", "flight", "plane", "pineapple", "fruit"])
encoder = skrub.TextEncoder(model_name="all-MiniLM-L6-v2", n_components=None)
embeddings = encoder.fit_transform(X).set_index(X.values)

sns.heatmap(embeddings @ embeddings.T)

# %% [markdown]
# ### `MinHashEncoder`
#
# A fast, stateless way of encoding strings that works especially well with
# models based on decision trees (gradient boosting, random forest).

# %% [markdown]
# # 4. How do I bring it all together?
#
# Skrub has several transformers that allow peforming typical dataframe
# operations such as projections, joins and aggregations _inside a scikit-learn pipeline_.
#
# Performing these operations in the machine-learning pipeline has several advantages:
#
# - Choices / hyperparameters can be optimized
# - Relevant state can be stored to ensure consistent transformations
# - All transformations are packaged together in an estimator
#
# There are several transformers such as `SelectCols`, `Joiner` (fuzzy joining),
# `InterpolationJoiner`, `AggJoiner`, ...
#
# A toy example using the `AggJoiner`:

# %%
from skrub import AggJoiner

airports = pd.DataFrame(
    {
        "airport_id": [1, 2],
        "airport_name": ["Charles de Gaulle", "Aeroporto Leonardo da Vinci"],
        "city": ["Paris", "Roma"],
    }
)
airports

# %%
flights = pd.DataFrame(
    {
        "flight_id": range(1, 7),
        "from_airport": [1, 1, 1, 2, 2, 2],
        "total_passengers": [90, 120, 100, 70, 80, 90],
        "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
    }
)
flights

# %%
agg_joiner = AggJoiner(
    aux_table=flights,
    main_key="airport_id",
    aux_key="from_airport",
    cols=["total_passengers"],
    operations=["mean", "std"],
)
agg_joiner.fit_transform(airports)

# %% [markdown]
# ## More interactive and expressive pipelines
#
# To go further than what can be done with scikit-learn Pipelines and the skrub
# transformers shown above, we are developing new utilities to easily define
# and inspect flexible pipelines that can process several dataframes.
#
# A prototype will be shown in a separate notebook.
