
## Encoding categorical (string/text) features
Categorical features have a "**cardinality**": the number of unique values

::: {.incremental}

- Low cardinality: `OneHotEncoder`
- High cardinality (>40 unique values): `skrub.StringEncoder`
- Text: `skrub.TextEncoder` and pretrained models from HuggingFace Hub

:::