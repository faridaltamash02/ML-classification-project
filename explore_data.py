import pandas as pd
import matplotlib.pyplot as plt
# load the bank dataset dimensions
df = pd.read_csv("data/bank-full.csv", sep=";")


# # Number of records and columns
# print("Dataset shape:", df.shape)

# # Class distribution
# print("\nTarget counts:")
# print(df["y"].value_counts())

# # Class distribution as percentages
# print("\nTarget percentages:")
# print(df["y"].value_counts(normalize=True) * 100)

# # Missing value counts
# print("\nMissing values:")
# print(df.isnull().sum())

# # Unique values in categorical columns
# print("\nCategorical column values:")
# for col in df.select_dtypes(include=["object"]).columns:
#     print(f"{col}: {df[col].unique()}")

# categorical_cols = df.select_dtypes(
#     include=["object"]
# ).columns

# for col in categorical_cols:
#    unknown_count = (df[col] == "unknown").sum()

#    if unknown_count > 0 :
#         percentage_unknown = (unknown_count / len(df)) * 100
#         print(f"{col}: {unknown_count} unknown values ({percentage_unknown:.2f}%)")

# numerical_cols = df.select_dtypes(
#     include=["int64", "float64"]
# ).columns

# print(df[numerical_cols].describe().T)


# print("Customers not previously contacted:")
# print((df["pdays"] == -1).sum())

# print("\nCustomers previously contacted:")
# print((df["pdays"] != -1).sum())




# numerical_cols = df.select_dtypes(
#     include=["int64", "float64"]
# ).columns

# df[numerical_cols].hist(
#     figsize=(14, 10),
#     bins=30,
#     edgecolor="black"
# )

# plt.suptitle(
#     "Numerical Feature Distributions",
#     fontsize=16
# )

# plt.tight_layout()
# plt.show()




# Subscription rate by education
education_rate = (
    df.groupby("education")["y"]
      .apply(lambda x: (x == "yes").mean() * 100)
      .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    education_rate.index,
    education_rate.values,
    color="steelblue",
    edgecolor="black"
)

ax.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3
)

ax.set_title("Subscription Rate by Education")
ax.set_xlabel("Education Level")
ax.set_ylabel("Subscription Rate (%)")
ax.set_ylim(0, education_rate.max() * 1.2)

plt.tight_layout()
plt.show()

print(education_rate.round(2))
