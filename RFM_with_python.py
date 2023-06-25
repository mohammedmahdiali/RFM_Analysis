import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


DATA_PATH = "../Data/flo_data_20k.csv"
data = pd.read_csv(DATA_PATH)

MAIN_DATA_COLS = data.columns


############ DATA UNDERSTANDING ############

# data.info()

# 0   master_id                          19945 non-null  object 
# 1   order_channel                      19945 non-null  object 
# 2   last_order_channel                 19945 non-null  object 
# 3   first_order_date                   19945 non-null  object 
# 4   last_order_date                    19945 non-null  object 
# 5   last_order_date_online             19945 non-null  object 
# 6   last_order_date_offline            19945 non-null  object 
# 7   order_num_total_ever_online        19945 non-null  float64
# 8   order_num_total_ever_offline       19945 non-null  float64
# 9   customer_value_total_ever_offline  19945 non-null  float64
# 10  customer_value_total_ever_online   19945 non-null  float64
# 11  interested_in_categories_12        19945 non-null  object 

TO_DATETIME_COLS = [
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline"
    ]

for col in TO_DATETIME_COLS:
    data[col] = pd.to_datetime(data[col])

# data.info()

# 0   master_id                          19945 non-null  object        
# 1   order_channel                      19945 non-null  object        
# 2   last_order_channel                 19945 non-null  object        
# 3   first_order_date                   19945 non-null  datetime64[ns]
# 4   last_order_date                    19945 non-null  datetime64[ns]
# 5   last_order_date_online             19945 non-null  datetime64[ns]
# 6   last_order_date_offline            19945 non-null  datetime64[ns]
# 7   order_num_total_ever_online        19945 non-null  float64       
# 8   order_num_total_ever_offline       19945 non-null  float64       
# 9   customer_value_total_ever_offline  19945 non-null  float64       
# 10  customer_value_total_ever_online   19945 non-null  float64       
# 11  interested_in_categories_12        19945 non-null  object

ORDER_CHANNELS = data.order_channel.unique()

# ORDER_CHANNELS : ['Android App', 'Desktop', 'Mobile', 'Ios App']

# Android App    6783
# Offline        6608
# Mobile         3172
# Ios App        1696
# Desktop        1686

TOTAL_ORDERS_OFFLINE = data.order_num_total_ever_offline.sum()
TOTAL_ORDERS_ONLINE = data.order_num_total_ever_online.sum()

# TOTAL_ORDERS_OFFLINE : 38173
# TOTAL_ORDERS_ONLINE : 62046

MEAN_VALUE_OFFLINE = round(data.customer_value_total_ever_offline.mean())
MEAN_VALUE_ONLINE = round(data.customer_value_total_ever_online.mean())

# MEAN_VALUE_OFFLINE : 254
# MEAN_VALUE ONLINE : 497

INTERESTED_CATEGORIES = []

for interested in data.interested_in_categories_12.unique():
        for category in interested.split("[")[1].split("]")[:-1]:
            if "," in category:
                for nested_cat in category.split(","):
                    if nested_cat.strip() not in INTERESTED_CATEGORIES:
                        INTERESTED_CATEGORIES.append(nested_cat.strip())
                    
# INTERESTED_CATS : ['ERKEK', 'COCUK', 'KADIN', 'AKTIFSPOR', 'AKTIFCOCUK']
    
data["total_orders"] = data.order_num_total_ever_offline + data.order_num_total_ever_online
data["total_value"] = data.customer_value_total_ever_offline + data.customer_value_total_ever_online


data[["master_id", "total_value"]].sort_values(by="total_value", ascending=False).head(5)

#                                  master_id  total_value
# 5d1c466a-9cfd-11e9-9897-000d3a38a36f     45905.10
# d5ef8058-a5c6-11e9-a2fc-000d3a38a36f     36818.29
# 73fd19aa-9e37-11e9-9897-000d3a38a36f     33918.10
# 7137a5c0-7aad-11ea-8f20-000d3a38a36f     31227.41
# 47a642fe-975b-11eb-8c2a-000d3a38a36f     20706.34


############ RFM ANALYSIS ############

today_date = dt.datetime(2021, 6, 1)
rfm = data.groupby('master_id').agg(
        {
                "last_order_date": lambda last_order_date: (today_date - last_order_date).astype('timedelta64[D]').astype(int),
                "total_orders": sum,
                "total_value": sum
            }
    )

rfm.columns = ['recency', 'frequency', 'monetary']

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

rfm_analysis = rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(
        {
                "recency": "mean",
                "frequency": "mean",
                "monetary": "mean",
                "segment": "count"
            }
    )

rfm_analysis.columns = ["recency(mean)", "frequency(mean)", "monetary(mean)", "segment(count)"]

#                         recency(mean)  frequency(mean)     monetary(mean)  segment(count)
# segment                                                         
# about_to_sleep       114.031649   2.406573   361.649373     1643
# at_Risk              242.328997   4.470178   648.325038     3152
# cant_loose           235.159129  10.716918  1481.652446     1194
# champions             17.142187   8.965104  1410.708938     1920
# hibernating          247.426303   2.391474   362.583299     3589
# loyal_customers       82.557926   8.356444  1216.257224     3375
# need_attention       113.037221   3.739454   553.436638      806
# new_customers         17.976226   2.000000   344.049495      673
# potential_loyalists   36.869744   3.310769   533.741344     2925
# promising             58.694611   2.000000   334.153338      668


############ Functionalization ############

def rfm_analysis(dataframe, export_to_csv=False):
    
    TO_DATETIME_COLS = [
            "first_order_date",
            "last_order_date",
            "last_order_date_online",
            "last_order_date_offline"
        ]
    
    for col in TO_DATETIME_COLS:
        dataframe[col] = pd.to_datetime(dataframe[col])
        
    dataframe["total_orders"] = dataframe.order_num_total_ever_offline + dataframe.order_num_total_ever_online
    dataframe["total_value"] = dataframe.customer_value_total_ever_offline + dataframe.customer_value_total_ever_online
    
    today_date = dt.datetime(2021, 6, 1)
    
    rfm = dataframe.groupby('master_id').agg(
            {
                    "last_order_date": lambda last_order_date: (today_date - last_order_date).astype('timedelta64[D]').astype(int),
                    "total_orders": sum,
                    "total_value": sum
                }
        )

    rfm.columns = ['recency', 'frequency', 'monetary']

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])

    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])

    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
                        
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
    
    rfm_analysis = rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(
            {
                    "recency": "mean",
                    "frequency": "mean",
                    "monetary": "mean",
                    "segment": "count"
                }
        )

    rfm_analysis.columns = ["recency(mean)", "frequency(mean)", "monetary(mean)", "segment(count)"]
    
    if export_to_csv:
        rfm.to_csv("rfm.csv")
        rfm_analysis.to_csv("rfm_analysis.csv")
        
    return rfm


rfm = rfm_analysis(data, export_to_csv=True)