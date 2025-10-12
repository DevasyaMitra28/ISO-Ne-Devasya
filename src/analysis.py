#!/usr/bin/env python
# coding: utf-8

# # Analysing Bidding Behaviour
# ---

# ## Slicing out unwanted columns
# We will restrict our analysis to the segment 1 bids. For this purpose, it is suitable to use `master_data_wide.csv`as all undesirable columns can be sliced out.

# In[1]:


#Importing the necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# In[2]:


#Importing master_data_wide.csv from /data/processed; file directory is elsewhere

project_root = os.path.dirname(os.getcwd()) #Gets the project root: "..\ISO-Ne-Devasya"
path = os.path.join(project_root, "data", "processed", "master_data_wide.csv") #Stores the path "..\ISO-Ne-Devasya\data\processed\master_data_wide.csv"
data = pd.read_csv(path)

#Checking whether the data has been corretly imported into the notebook
data.head()     #Prints the first 5 rows of master_data_wide.csv, now saved as `data`


# Since analysis on segment 1 bid is required, we need to slice out the data for other segments.

# In[5]:


#Getting the column names
print(data.columns)

#Deriving the list of unwanted columns
unwanted_columns = [col for i in range(2, 11) for col in (f"seg{i}_price", f"seg{i}_mw")]
date_ts = ['date', '_ts']        #repeating columns; safe to omit
unwanted_columns = unwanted_columns + date_ts
print("Unwanted columns list:",unwanted_columns)


# In[6]:


#Dropping the unwanted ones
data = data.drop(columns = unwanted_columns)
data.head(8)


# In this context, using segment 1 price is helpful because it is the real one; all the assets for both the five days for both the markets have declared the price and the unit in the first segment. However, many of them did not go beyond the first one. So I had used extrapolation techniques to best predict the units and their prices in the subsequent segments.
# 
# Now that we have cleaned the data, we can proceed towards analysis.

# ## Average Price Time Series

# A rudimentary summary of the bidding prices and the bidding units will form the base for further analysis.

# In[25]:


# Ensuring that the column 'int_start' is in datetime format
data["int_start"] = pd.to_datetime(data["int_start"])
data = data.sort_values("int_start")

# Computing simple average price per timestamp and market_type
avg_df = (
    data.groupby(["int_start", "market_type"], as_index=False)["seg1_price"]
        .mean()
)

# Computing MW-weighted average price per timestamp and market_type 
def weighted_price_calc(g):
    total_mw = g["seg1_mw"].sum()
    if total_mw == 0:
        return np.nan
    return (g["seg1_price"] * g["seg1_mw"]).sum() / total_mw

weighted_df = (
    data.groupby(["int_start", "market_type"], as_index=False)
        .apply(lambda g: pd.Series({"weighted_price": weighted_price_calc(g)}), include_groups=False)
)

# Pivoting both
avg_pivot = avg_df.pivot(index="int_start", columns="market_type", values="seg1_price").sort_index()
weighted_pivot = weighted_df.pivot(index="int_start", columns="market_type", values="weighted_price").sort_index()

# Defining function to set x-axis tick locators smartly
def format_time_axis(ax, pivot):
    span = pivot.index.max() - pivot.index.min()
    if span <= pd.Timedelta(days=2):
        locator = mdates.HourLocator(interval=1)
        fmt = mdates.DateFormatter("%b %d\n%H:%M")
    elif span <= pd.Timedelta(days=90):
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        fmt = mdates.ConciseDateFormatter(locator)
    else:
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        fmt = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis='x', rotation=45)

# Plot side-by-side horizontally
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# Color palette
colors = {"da": "#3C8208", "rt": "#BF092F"}

# Left: simple average
ax = axes[0]
for col in avg_pivot.columns:
    ax.plot(avg_pivot.index, avg_pivot[col], label=col.upper(), color=colors[col], linewidth=1.5)
ax.set_xlabel("Time")
ax.set_ylabel("Average Price")
ax.set_title("Average Prices Over Time")
format_time_axis(ax, avg_pivot)

# Right: MW-weighted average
ax = axes[1]
for col in weighted_pivot.columns:
    ax.plot(weighted_pivot.index, weighted_pivot[col], label=col.upper(), color=colors[col], linewidth=1.5)
ax.set_xlabel("Time")
ax.set_ylabel("MW-weighted Average Price")
ax.set_title("MW-weighted Average Prices Over Time")
format_time_axis(ax, weighted_pivot)

# Shared legend (right side)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, title="Market Type", loc="center right")

# Adjust layout so legend fits nicely
plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.show()


# Two broad inferences can be made:
# 
# 1. The simple average price line graph for both the markets do not show any significant seasonal trend as against the unit weighted avergae price line graph; and
# 2. Average price bids in the day-ahead market are higher than the real-time markets, although the same cannot be said for MW-weighed ones.

# ## Analysing negative price bids

# In[7]:


#Summary statistics of seg1_mw and seg1_price
data[['seg1_price','seg1_mw']].describe()


# Why are there negative prices? Oversupply could be a reason.

# In[30]:


#Getting the number of negative price bids by hour and market_type
neg_price_by_hour_market = (
    data[data["seg1_price"] < 0]
    .groupby(["asset_id", "market_type", "hour"])["seg1_price"]
    .count()
    .reset_index(name="neg_price_count")
)

#Checking
neg_price_by_hour_market


# In[34]:


# Select the key columns from neg_price_by_hour_market
keys = neg_price_by_hour_market[["asset_id", "market_type", "hour"]]

# Slice 'data' to include only rows with matching keys
negative_data_slice = data.merge(keys, on=["asset_id", "market_type", "hour"], how="inner")

# Check result
print(negative_data_slice.shape)
negative_data_slice.head()


# In[37]:


#Simple hourly average
time_series_avg = (
    negative_data_slice
    .groupby(["int_start", "market_type"])["seg1_price"]
    .mean()
    .reset_index()
)


# In[38]:


plt.figure(figsize=(12,6))

for market in time_series_avg["market_type"].unique():
    subset = time_series_avg[time_series_avg["market_type"] == market]
    plt.plot(subset["int_start"], subset["seg1_price"], label=market.upper())

plt.title("Average Negative Prices Over Time")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(title="Market Type")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# From this graph, we can see that the days-ahead prices are also at least higer than real-time markets. It is more interesting to know that these curves are parallele and have the same gradient (see the graph after 27 July). 

# ## Bid price distribution

# In[40]:


for market in data["market_type"].unique():
    s = data.loc[data["market_type"]==market, "seg1_price"].dropna()
    plt.figure(figsize=(7,4))
    plt.hist(s, bins=40, density=True, alpha=0.5, edgecolor="k")
    try:
        kde = gaussian_kde(s)
        xs = np.linspace(s.min(), s.max(), 300)
        plt.plot(xs, kde(xs), lw=2)
    except Exception:
        pass
    plt.title(f"seg1_price distribution — market {market}")
    plt.xlabel("seg1_price")
    plt.ylabel("Density")
    plt.show()


# This is a right skewed distribution for both the markets.
