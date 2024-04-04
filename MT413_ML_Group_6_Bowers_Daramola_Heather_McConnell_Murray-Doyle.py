# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:33:19 2023

@author: cian3
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Import and view dataset
file_path = r"C:\Users\cian3\Downloads"
file_name = r"\Onlineretail.csv"
online_sales = pd.read_csv(file_path + file_name,
                           sep=",", encoding="ISO-8859-1", header=0,
                           index_col=None)
print(online_sales.head())

#Check data for missing values and remove if any
print("Number of missing values: \n",online_sales.isna().sum(axis=0))
online_sales = online_sales.dropna(axis=0, how="any")
print("Number of missing values: \n",online_sales.isna().sum(axis=0))

#Remove Duplicates
#Duplicate removal not applicable here

#Change Formatting
print("Original data: \n", online_sales.describe(),"\n")
online_sales["CustomerID"] = online_sales["CustomerID"].astype(str)
online_sales["InvoiceDate"] = pd.to_datetime(online_sales["InvoiceDate"],format='%d-%m-%Y %H:%M')

#Detect and remove outliers
print("Original data: \n", online_sales.describe(),"\n")

#Negative Quantity and Price are not possible we will remove these
online_sales.drop(online_sales[online_sales["UnitPrice"] <=0].index, inplace=True)
online_sales.drop(online_sales[online_sales["Quantity"] <=1].index, inplace=True)
print("Original data: \n", online_sales.describe(),"\n")

#Z-score method
print("Original data: \n", online_sales.describe(),"\n")
z_upper = online_sales["UnitPrice"].mean() + 3*online_sales["UnitPrice"].std()
z_lower = online_sales["UnitPrice"].mean() - 3*online_sales["UnitPrice"].std()
online_sales_z = online_sales[(online_sales["UnitPrice"] < z_upper) & (
    online_sales["UnitPrice"] > z_lower)]

z_upper = online_sales["Quantity"].mean() + 3*online_sales["Quantity"].std()
z_lower = online_sales["Quantity"].mean() - 3*online_sales["Quantity"].std()
online_sales_z = online_sales_z[(online_sales_z["Quantity"] < z_upper) & (
    online_sales_z["Quantity"] > z_lower)]

print("Outliers removed using z-score: \n", online_sales_z.describe(),"\n")

#IQR Method
print("Original data: \n", online_sales.describe(),"\n")
q1 = np.percentile(online_sales["UnitPrice"],25)
q3 = np.percentile(online_sales["UnitPrice"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_iqr = online_sales[(online_sales["UnitPrice"] < upperiqr) & (
    online_sales["UnitPrice"] > lowiqr)]

q1 = np.percentile(online_sales_iqr["Quantity"],25)
q3 = np.percentile(online_sales_iqr["Quantity"],75)
iqr = q3 - q1
lowiqr1 = q1 - 1.5 * iqr
upperiqr1 = q3 + 1.5* iqr
online_sales_iqr = online_sales_iqr[(online_sales_iqr["Quantity"] < upperiqr1) & (
    online_sales_iqr["Quantity"] > lowiqr1)]
print("Outliers removed using IQR: \n", online_sales_iqr.describe(),"\n")
#Will need to create new features to detect outliers


#Feature Engineering - TotalCost of each transaction provides more info
online_sales["TotalCost"] = online_sales["Quantity"]*online_sales["UnitPrice"]

#Irrelevant Features - Heatmap to show which features have a high correlation with one another
correlation_matrix = online_sales[["Quantity","UnitPrice","TotalCost"]].corr()
sns.heatmap(correlation_matrix, vmin=0.0, vmax=1.0, annot=True)
plt.title("Correlation Matrix",fontweight="bold")
plt.show()
#We can use new TotalCost feature rather than Quantity and UnitPrice

#Feature Engineering
#RFM analysis - Recency
most_recent_purchase = max(online_sales["InvoiceDate"])
online_sales["TimeDiff"] = most_recent_purchase - online_sales["InvoiceDate"]
online_sales_rfm_r = online_sales.groupby("CustomerID")["TimeDiff"].min()
online_sales_rfm_r.head()
#Above is using customer ID as index - needs to be reset with below
online_sales_rfm_r = online_sales_rfm_r.reset_index()
#Time difference is showing hours and miutes, this much detail is not needed
online_sales_rfm_r["TimeDiff"] = online_sales_rfm_r["TimeDiff"].dt.days 
online_sales_rfm_r.head()

#RFM analysis - Frequency
online_sales_rfm_f = online_sales.groupby("CustomerID")["InvoiceNo"].count()
online_sales_rfm_f.head()
#Above is using customer ID as index - needs to be reset with below
online_sales_rfm_f = online_sales_rfm_f.reset_index()
online_sales_rfm_f.head()

#RFM analysis - Monetary - Grouping customers by their total spend
online_sales_rfm_m = online_sales.groupby("CustomerID")["TotalCost"].sum()
online_sales_rfm_m.head()
#Above is using customer ID as index - needs to be reset with below
online_sales_rfm_m = online_sales_rfm_m.reset_index()
online_sales_rfm_m.head()

#RFM - Creating dataframe for rfm analysis
online_sales_rfm = pd.merge(online_sales_rfm_r,
                            online_sales_rfm_f,
                            on="CustomerID",
                            how="inner")
online_sales_rfm = pd.merge(online_sales_rfm,
                            online_sales_rfm_m, 
                            on="CustomerID",
                            how="inner")
online_sales_rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
print("RFM Table: \n",online_sales_rfm.head())

#Visulaisation of outlier distribution
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
axes[0].boxplot(online_sales_rfm["Recency"], labels=["Recency"])
axes[0].set_title("Recency")
axes[1].boxplot(online_sales_rfm["Frequency"], labels=["Frequency"])
axes[1].set_title("Frequency")
axes[2].boxplot(online_sales_rfm["Monetary"], labels=["Monetary"])
axes[2].set_title("Monetary")
for ax in axes:
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Range", fontweight="bold")
fig.suptitle("Outlier Distribution", fontweight="bold")
plt.show()

#Removing outliers using IQR
original_len = len(online_sales_rfm)
#Rfm Recency
q1 = np.percentile(online_sales_rfm["Recency"],25)
q3 = np.percentile(online_sales_rfm["Recency"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_rfm = online_sales_rfm[(online_sales_rfm["Recency"] < upperiqr) & (
    online_sales_rfm["Recency"] > lowiqr)]


q1 = np.percentile(online_sales_rfm["Frequency"],25)
q3 = np.percentile(online_sales_rfm["Frequency"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_rfm = online_sales_rfm[(online_sales_rfm["Frequency"] < upperiqr) & (
    online_sales_rfm["Frequency"] > lowiqr)]

q1 = np.percentile(online_sales_rfm["Monetary"],25)
q3 = np.percentile(online_sales_rfm["Monetary"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_rfm = online_sales_rfm[(online_sales_rfm["Monetary"] < upperiqr) & (
    online_sales_rfm["Monetary"] > lowiqr)]
online_sales_rfm["CustomerID"] = online_sales_rfm["CustomerID"].astype(str)
print("Outliers removed from RFM table using IQR: ", original_len - len(online_sales_rfm))
#Resetting and removing current index
online_sales_rfm = online_sales_rfm.reset_index(drop=True)

#Using z-score to standardise data
online_sales_rfm_df = online_sales_rfm[["Recency","Frequency","Monetary"]]
scaler = StandardScaler()
online_sales_rfm_scaled = scaler.fit_transform(online_sales_rfm_df)
online_sales_rfm_scaled = pd.DataFrame(online_sales_rfm_scaled)
online_sales_rfm_scaled.columns = ["Recency","Frequency","Monetary"]
print("Standardised RFM Table: \n",online_sales_rfm_scaled.head())

#Creating K-Means Clustering algorithm
#Finding n clusters - Elbow graph
cluster_range = [*range(2,11)]
lst = []
for i in cluster_range:
    kmeans = KMeans(n_clusters=i,max_iter=50)
    kmeans.fit(online_sales_rfm_scaled)
    lst.append(kmeans.inertia_)

plt.plot(lst,marker="o",linestyle=":",c="black")
plt.title("Elbow Method",fontweight="bold")
plt.xlabel("Number of Clusters",fontweight="bold")
plt.ylabel("Inertia", fontweight="bold")
plt.show()

#K-Means algorithm
kmeans = KMeans(n_clusters=3,max_iter=50)
kmeans.fit(online_sales_rfm_scaled) 

online_sales_rfm["ClusterID"] = kmeans.labels_
print("RFM Table with Cluster ID: \n", online_sales_rfm.head())
online_sales_rfm_scaled["ClusterID"] = kmeans.labels_

#3d Graph showing clustering
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(online_sales_rfm_scaled.iloc[:, 0],
                     online_sales_rfm_scaled.iloc[:, 1],
                     online_sales_rfm_scaled.iloc[:, 2],
                     c=kmeans.labels_, cmap='viridis', s=50)

ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           kmeans.cluster_centers_[:, 2],
           s=200, marker='X', c='red')

ax.set_xlabel('Recency',fontweight="bold")
ax.set_ylabel('Frequency',fontweight="bold")
ax.set_zlabel('Monetary',fontweight="bold")
plt.colorbar(scatter, ax=ax, label='Clusters')
plt.title('K-Means Clustering Results in 3D',fontweight="bold")
plt.show()

#Understanding clusters with boxplots
sns.boxplot(data=online_sales_rfm, x="ClusterID", y="Recency")
plt.title("Clusters by Recency",fontweight="bold")
plt.xlabel("Clusters",fontweight="bold")
plt.ylabel("Recency (days)",fontweight="bold")
plt.show()

sns.boxplot(data=online_sales_rfm, x="ClusterID", y="Frequency")
plt.title("Clusters by Frequency",fontweight="bold")
plt.xlabel("Clusters",fontweight="bold")
plt.ylabel("Frequency",fontweight="bold")
plt.show()

sns.boxplot(data=online_sales_rfm, x="ClusterID", y="Monetary")
plt.title("Clusters by Monetary",fontweight="bold")
plt.xlabel("Clusters",fontweight="bold")
plt.ylabel("Monetary ($)",fontweight="bold")
plt.show()