## Fraud Detection with Isolation Forest
Determine the transaction is fraud or not with anomaly detection - Isolated Forest. Dataset contain Credit Card Number, date transaction, transaction ($), Longitude and latitude position transaction, Zipcode, State, and credit card limit.

dataset source :
https://www.kaggle.com/datasets/iabhishekofficial/creditcard-fraud-detection/data

Read Dataset
```python
df_fraud
```

# Check null value
```python
df_fraud.isnull().sum()
```
credit_card                  0
date                         0
transaction_dollar_amount    0
Long                         0
Lat                          0
city                         0
state                        0
zipcode                      0
credit_card_limit            0
dtype: int64

# Check Duplicated
```python
df_fraud.duplicated().sum()
```

# Count Total CC
```python
df_fraud['credit_card'].value_counts()
```

# Count Total City
```python
df_fraud['city'].value_counts()
```

# Count Total State
```python
df_fraud['state'].value_counts()
```

# Extract date
```python
df_fraud['year'] = df_fraud['date'].dt.year
df_fraud['month'] = df_fraud['date'].dt.month
df_fraud['day'] = df_fraud['date'].dt.day
df_fraud['hour'] = df_fraud['date'].dt.hour
df_fraud.drop(['date','city', 'state', 'zipcode'], axis=1, inplace=True)
df_fraud
```

# Transaction ($)

# Heatmap Correlation Matrix

![Heatmap](heatmap.png)
