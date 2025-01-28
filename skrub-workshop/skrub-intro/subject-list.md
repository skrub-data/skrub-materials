Introduction
=== 
1. Example use case.
2. Detailed explanation of the features.
3. How to get skrub, reaching out and contributing

## Before we begin
- Open issues if you think there's something missing
- And don't forget to mention the suggestions on slido! 

## Example use case
I have a table I want to train a machine learning model on. How do I proceed? 

- Do some exploratory data analysis
- Pre-process the data 
- Build a scikit-learn estimator 
- ???
- Profit

## Exploratory data analysis 
With `pd.describe()`: 

With `skrub.TableReport()`: 

## Examples of `TableReport`

link etc

## Pre-process the data
With plain scikit-learn preprocessors:

With `tabular_learner`:

## Prepare the estimator
With plain scikit-learn:

With `tabular_learner`:

## Unmasking the `tabular_learner`:
silly meems
`tabular_learner` under the hood is `TableVectorizer`

## What is `TableVectorizer`? 
- Preprocessing
    - check nulls
    - check and convert dtypes
- Create transformers
    - numerical
    - categorical, high and low cardinality
    - datetime

## Example: encoding categorical variables
- Sentiment analysis
- Toxicity in tweets (mention that `StringEncoder` will be released soon)

## Example: `DatetimeEncoder`
Ask about circular vs spline encoding

## Joining tables
- Aggjoiner
- Spatial joins
- Fuzzy joins
- Interpolation joins

## Installing skrub
Code on how to install skrub

## How to contribute to skrub
Link to the contribute page

## How to reach out? 
- Git repo
- Discord server
- Bluesky
- Website