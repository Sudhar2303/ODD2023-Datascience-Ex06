# ODD2023-Datascience-Ex06
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
## STEP 1:
Read the given Data

## STEP 2:
Clean the Data Set using Data Cleaning Process

## STEP 3:
Apply Feature Transformation techniques to all the features of the data set

## STEP 4:
Print the transformed features

# PROGRAM:
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
```
```
from google.colab import files
uploaded = files.upload()
```
```
df = pd.read_csv('Data_to_Transform(1).csv')
df
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/ad6da1d4-8987-4c52-9325-2e73cc96bf40)
```
df.skew()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/147b5981-c16b-42b6-a2bb-7c76ca6348b5)
```
df1 = df.copy()
sm.qqplot(df1['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/b0515db5-c0f8-4269-80f6-a11acf34981f)
```
sm.qqplot(df1['HighlyNegativeSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/110ff986-38ce-4e6a-9051-4ee539014274)
```
sm.qqplot(df1['HighlyNegativeSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/e9b7d4f2-364e-456e-bca2-4b2fb1e30e79)
```
sm.qqplot(df1['ModerateNegativeSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/fa0fe316-88f1-4ecf-b54a-93286375e701)
```
df1['HighlyPositiveSkew'] = np.log(df1['HighlyPositiveSkew'])
sm.qqplot(df1['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/fe482c47-0477-4821-8527-f6e14b3d78d9)
```
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2['HighlyPositiveSkew']
sm.qqplot(df2['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/a20632e0-91d4-476a-b462-18549d32c05d)
```
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3['HighlyPositiveSkew']**(1/1.2)
sm.qqplot(df2['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/cb84b880-9724-4390-bc01-b98f2879b534)
```
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4['ModeratePositiveSkew'])
sm.qqplot(df4['ModeratePositiveSkew_1'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/489030e3-28d9-43c4-b4d2-952707d597e6)
```
from sklearn.preprocessing import PowerTransformer
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
```
![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/513b1961-3c47-440c-aae5-11b3aab71727)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
```

![image](https://github.com/Sudhar2303/ODD2023-Datascience-Ex06/assets/133684710/6d27f57f-b299-497c-9919-5c7144beae9d)

# RESULT:
  Thus feature transformation is done for the given set
