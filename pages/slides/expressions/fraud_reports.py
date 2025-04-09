import skrub
from skrub import datasets

data = datasets.fetch_credit_fraud()
baskets = skrub.TableReport(data.baskets).html_snippet()
products = skrub.TableReport(data.products).html_snippet()

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Credit fraud dataset</title>
<style>
:root {{
    --fontStack-sansSerif: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
}}

body {{
    font-family: var(--fontStack-sansSerif);
}}

</style>
</head>
<body>
<h2>Baskets</h2>
{baskets}

<h2>Products</h2>
{products}
</body>
</html>
"""

with open("dataset-overview.html", "w", encoding="utf-8") as f:
    f.write(html)
