## DataOps...

::: {.incremental}
- Extend the `scikit-learn` machinery to complex multi-table operations 
- Take care of data leakage
- Track all operations with a computational graph (a *Data Ops plan*)
- Allow tuning any operation in the Data Ops plan
- Can be persisted and shared easily 
:::

## How do DataOps work, though?  {.smaller}
DataOps **wrap** around *user operations*, where user operations are:

- any dataframe operation (e.g., merge, group by, aggregate etc.)
- scikit-learn estimators (a Random Forest, RidgeCV etc.)
- custom user code (load data from a path, fetch from an URL etc.)

::: {.fragment}

::: {.callout-important}
DataOps _record_ user operations, so that they can later be _replayed_ in the same
order and with the same arguments on unseen data. 
:::
::: 
