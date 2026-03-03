# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.11.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()

X = dataset.X
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?

# %%
# Display the number of rows and columns
print(f"Rows (examples): {X.shape[0]}")
print(f"Columns (features): {X.shape[1]}")

# %%
# You can also look at the first few rows of the dataset
X.head()

# %% [markdown]
# ## Question 2: What is the distribution of the target?

# %%
# Count how many respondents belong to each region
print(y.value_counts())

# %%
# Visualize the target distribution with a bar plot
import matplotlib.pyplot as plt

y.value_counts().plot(kind="barh")
plt.xlabel("Number of respondents")
plt.ylabel("Census Region")
plt.title("Distribution of Census Regions")
plt.tight_layout()
plt.show()

# %% [markdown]
# La cible est **déséquilibrée** : certaines régions ont beaucoup plus de répondants que d'autres.

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?

# %%
# List all column names
print(X.columns.tolist())

# %%
# Show data types for each column
print(X.dtypes)

# %% [markdown]
# La plupart des colonnes sont de type **object** (catégoriel/texte).
# Peu ou pas de colonnes numériques.

# %%
# Compter les types
print("Numerical:", (X.dtypes != "object").sum())
print("Categorical:", (X.dtypes == "object").sum())

# %%
from skrub import TableReport
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?

# %%
# Check for NaN missing values
print(X.isna().sum())
print(f"\nTotal NaN: {X.isna().sum().sum()}")

# %% [markdown]
# Les NaN classiques sont peu nombreux, mais certaines colonnes encodent les valeurs manquantes autrement.

# %%
# Look at unique values for the Household_Income column
print(X["Household_Income"].unique())

# %%
# Look at unique values for the Education column
print(X["Education"].unique())

# %% [markdown]
# On voit la valeur **"Prefer not to answer"** qui représente des données manquantes "cachées".

# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?

# %%
# Display the value counts for the column
col = "How_much_do_you_personally_identify_as_a_Midwesterner"
print(X[col].value_counts())

# %%
# Make a bar plot of the results
X[col].value_counts().plot(kind="barh")
plt.xlabel("Number of respondents")
plt.title("How much do you identify as a Midwesterner?")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bonus: Explore another feature — Age

# %%
# Distribution de l'âge des répondants
X["Age"].value_counts().plot(kind="barh")
plt.xlabel("Number of respondents")
plt.title("Age distribution")
plt.tight_layout()
plt.show()
