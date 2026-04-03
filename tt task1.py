import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"C:\Users\yoeshwar\OneDrive\Pictures\Desktop\internships\cleaned_superstore.csv", encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
customer_df = df.groupby('Customer ID').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Discount': 'mean',
    'Order ID': 'count',
    'Order Date': lambda x: (df['Order Date'].max() - x.max()).days
}).reset_index()
customer_df.columns = [
    'CustomerID',
    'TotalSales',
    'TotalProfit',
    'TotalQuantity',
    'AvgDiscount',
    'TotalOrders',
    'Recency'
]
X = customer_df[['TotalSales', 'TotalProfit', 'TotalQuantity', 'TotalOrders', 'Recency']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)
numeric_cols = ['TotalSales', 'TotalProfit', 'TotalQuantity', 'TotalOrders', 'Recency']
cluster_summary = customer_df.groupby('Cluster')[numeric_cols].mean()
print(cluster_summary)
plt.figure()
sns.scatterplot(x=customer_df['TotalSales'], y=customer_df['TotalProfit'], hue=customer_df['Cluster'])
plt.show()
plt.figure()
sns.countplot(x='Cluster', data=customer_df)
plt.show()
customer_df.to_csv(r"C:\Users\yoeshwar\OneDrive\Pictures\Desktop\internships\Customer_Segmentation_Output.csv", index=False)