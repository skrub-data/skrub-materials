
## Encoding _all the features_: `TableVectorizer` { auto-animate="true"}

```{python}
#| echo: true
from skrub import TableVectorizer

table_vec = TableVectorizer()
df_encoded = table_vec.fit_transform(df)
```

::: {.fragment}
- Apply the `Cleaner` to all columns
- Split columns by dtype and # of unique values
- Encode each column separately
:::


## Encoding _all the features_: `TableVectorizer` {.smaller auto-animate="true"}

![](/images/skrub-table-vectorizer.png)
