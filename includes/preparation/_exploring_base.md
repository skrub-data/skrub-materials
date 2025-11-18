
## Exploring the data {.smaller auto-animate="true"}
```{.python}
import pandas as pd
import matplotlib.pyplot as plt
import skrub

dataset = skrub.datasets.fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y

df = pd.DataFrame(employees)

# Plot the distribution of the numerical values using a histogram
fig, axs = plt.subplots(2,1, figsize=(10, 6))
ax1, ax2 = axs

ax1.hist(df['year_first_hired'], bins=30, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Year first hired')
ax1.set_ylabel('Frequency')
ax1.grid(True, linestyle='--', alpha=0.5)

# Count the frequency of each category
category_counts = df['department'].value_counts()

# Create a bar plot
category_counts.plot(kind='bar', edgecolor='black', ax=ax2)

# Add labels and title
ax2.set_xlabel('Department')
ax2.set_ylabel('Frequency')
ax2.grid(True, linestyle='--', axis='y', alpha=0.5)  # Add grid lines for y-axis

fig.suptitle("Distribution of values")

# Show the plot
plt.show()
```
## Exploring the data {.smaller auto-animate="true"}

```{python}
#| echo: false
import pandas as pd
import matplotlib.pyplot as plt
import skrub
from skrub.datasets import fetch_employee_salaries
from pprint import pprint

dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y

df = pd.DataFrame(employees)

# Plot the distribution of the numerical values using a histogram
fig, axs = plt.subplots(2,1, figsize=(10, 6))
ax1, ax2 = axs

ax1.hist(df['year_first_hired'], bins=30, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Year first hired')
ax1.set_ylabel('Frequency')
ax1.grid(True, linestyle='--', alpha=0.5)

# Count the frequency of each category
category_counts = df['department'].value_counts()

# Create a bar plot
category_counts.plot(kind='bar', edgecolor='black', ax=ax2)

# Add labels and title
ax2.set_xlabel('Department')
ax2.set_ylabel('Frequency')
ax2.grid(True, linestyle='--', axis='y', alpha=0.5)  # Add grid lines for y-axis

fig.suptitle("Distribution of values")

# Show the plot
plt.show()
```
