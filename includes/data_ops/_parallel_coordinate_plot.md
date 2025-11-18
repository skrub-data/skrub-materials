## A parallel coordinate plot to explore hyperparameters{auto-animate="true" .smaller} 

```{.python}
search = pred.skb.get_randomized_search(fitted=True)
search.plot_parallel_coord()
```
```{python}
import os
from plotly.io import read_json

project_root = os.environ.get("QUARTO_PROJECT_DIR", ".")
json_path = os.path.join(project_root, "resources", "parallel_coordinates_hgbr.json")

fig = read_json(json_path)
fig.update_layout(margin=dict(l=200))
```
