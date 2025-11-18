## Encoding numerical features with `skrub.SquashingScaler`
```{python}
#| fig-align: center
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # for reproducibility

values = np.random.rand(100, 1)
n_outliers = 15
outlier_indices = np.random.choice(values.shape[0], size=n_outliers, replace=False)
values[outlier_indices] = np.random.rand(n_outliers, 1) * 100 - 50

x = np.arange(values.shape[0])
fig, axs = plt.subplots(1, layout="constrained", figsize=(6, 4))

axs.plot(x,values)
_ = axs.set(
    title="Feature with outliers",
    ylabel="value",
    xlabel="Sample ID"
    )
axs.axhspan(-2, 2, color="gray", alpha=0.15)

x_data, y_data = [30, 2]
desc = "Data is mostly\nin [-2, 2]"
axs.annotate(
    desc,
    xy=(x_data, y_data),
    xytext=(0.15, 0.8),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="red"),

)

x_outlier, y_outlier = np.argmax(values), np.max(values)
desc = "There are large\noutliers throughout."
_ = axs.annotate(
    desc,
    xy=(x_outlier, y_outlier),
    xytext=(0.6, 0.85),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="red"),

)
```

::: {.notes}
Skrub wants to solve ML problems based partly on solid engineering and partly on
statistical notions. The SquashingScaler is based on the second part, and is taken
from a recent paper that evaluates different techniques for improving the performance
of NNs. 
:::



## Encoding numerical features with `skrub.SquashingScaler`
```{python}
#| fig-align: center
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from skrub import SquashingScaler

squash_scaler = SquashingScaler()
squash_scaled = squash_scaler.fit_transform(values)

robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(values)

standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(values)

quantile_transformer = QuantileTransformer(n_quantiles=100)
quantile_scaled = quantile_transformer.fit_transform(values)
# %%
import matplotlib.pyplot as plt

x = np.arange(values.shape[0])
fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(8, 5))

ax = axs[0]
ax.plot(x, sorted(values), label="Original Values", linewidth=2.5)
ax.plot(x, sorted(squash_scaled), label="SquashingScaler")
ax.plot(x, sorted(robust_scaled), label="RobustScaler", linestyle="--")
ax.plot(x, sorted(standard_scaled), label="StandardScaler")
ax.plot(x, sorted(quantile_scaled), label="QuantileTransformer")

# Add a horizontal band in [-4, +4]
ax.axhspan(-4, 4, color="gray", alpha=0.15)
ax.set(title="Original data", xlim=[0, values.shape[0]], xlabel="Percentile")
ax.legend()

ax = axs[1]
ax.plot(x, sorted(values), label="Original Values", linewidth=2.5)
ax.plot(x, sorted(squash_scaled), label="SquashingScaler")
ax.plot(x, sorted(robust_scaled), label="RobustScaler", linestyle="--")
ax.plot(x, sorted(standard_scaled), label="StandardScaler")
ax.plot(x, sorted(quantile_scaled), label="QuantileTransformer")

ax.set(ylim=[-4, 4])
ax.set(title="In range [-4, 4]", xlim=[0, values.shape[0]], xlabel="Percentile")

# Highlight the bounds of the SquashingScaler
ax.axhline(y=3, alpha=0.2)
ax.axhline(y=-3, alpha=0.2)

fig.suptitle(
    "Comparison of different scalers on sorted data with outliers", fontsize=20
)
fig.supylabel("Value")

desc = "The RobustScaler is\naffected by outliers"
axs[0].annotate(
    desc,
    xy=(0, -70),
    xytext=(0.4, 0.2),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="red"),
)

desc = "The SquashingScaler is\nclipped to a finite value"
_ = axs[1].annotate(
    desc,
    xy=(0, -3),
    xytext=(0.4, 0.2),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="red"),
)

```
