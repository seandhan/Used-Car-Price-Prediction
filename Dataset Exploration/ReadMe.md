# Importing Libraries


```python
import pandas as pd
import numpy as np
import scipy as spy
import copy
import math
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```


```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

---

# Data ingestion 


```python
data = pd.read_csv("used_cars_data.csv")
```

---

## Preview the dataset


```python
# Preview the dataset
# View the first 5, last 5 and random 10 rows
print('First five rows', '--'*55)
display(data.head())

print('Last five rows', '--'*55)
display(data.tail())

print('Random ten rows', '--'*55)
np.random.seed(1)
display(data.sample(n=10))
```

    First five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.No.</th>
      <th>Name</th>
      <th>Location</th>
      <th>Year</th>
      <th>Kilometers_Driven</th>
      <th>Fuel_Type</th>
      <th>Transmission</th>
      <th>Owner_Type</th>
      <th>Mileage</th>
      <th>Engine</th>
      <th>Power</th>
      <th>Seats</th>
      <th>New_Price</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Maruti Wagon R LXI CNG</td>
      <td>Mumbai</td>
      <td>2010</td>
      <td>72000</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>26.6 km/kg</td>
      <td>998 CC</td>
      <td>58.16 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Hyundai Creta 1.6 CRDi SX Option</td>
      <td>Pune</td>
      <td>2015</td>
      <td>41000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>19.67 kmpl</td>
      <td>1582 CC</td>
      <td>126.2 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Honda Jazz V</td>
      <td>Chennai</td>
      <td>2011</td>
      <td>46000</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>18.2 kmpl</td>
      <td>1199 CC</td>
      <td>88.7 bhp</td>
      <td>5.0</td>
      <td>8.61 Lakh</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Maruti Ertiga VDI</td>
      <td>Chennai</td>
      <td>2012</td>
      <td>87000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>20.77 kmpl</td>
      <td>1248 CC</td>
      <td>88.76 bhp</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Audi A4 New 2.0 TDI Multitronic</td>
      <td>Coimbatore</td>
      <td>2013</td>
      <td>40670</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>Second</td>
      <td>15.2 kmpl</td>
      <td>1968 CC</td>
      <td>140.8 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>17.74</td>
    </tr>
  </tbody>
</table>
</div>


    Last five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.No.</th>
      <th>Name</th>
      <th>Location</th>
      <th>Year</th>
      <th>Kilometers_Driven</th>
      <th>Fuel_Type</th>
      <th>Transmission</th>
      <th>Owner_Type</th>
      <th>Mileage</th>
      <th>Engine</th>
      <th>Power</th>
      <th>Seats</th>
      <th>New_Price</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7248</th>
      <td>7248</td>
      <td>Volkswagen Vento Diesel Trendline</td>
      <td>Hyderabad</td>
      <td>2011</td>
      <td>89411</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>20.54 kmpl</td>
      <td>1598 CC</td>
      <td>103.6 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7249</th>
      <td>7249</td>
      <td>Volkswagen Polo GT TSI</td>
      <td>Mumbai</td>
      <td>2015</td>
      <td>59000</td>
      <td>Petrol</td>
      <td>Automatic</td>
      <td>First</td>
      <td>17.21 kmpl</td>
      <td>1197 CC</td>
      <td>103.6 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7250</th>
      <td>7250</td>
      <td>Nissan Micra Diesel XV</td>
      <td>Kolkata</td>
      <td>2012</td>
      <td>28000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>23.08 kmpl</td>
      <td>1461 CC</td>
      <td>63.1 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7251</th>
      <td>7251</td>
      <td>Volkswagen Polo GT TSI</td>
      <td>Pune</td>
      <td>2013</td>
      <td>52262</td>
      <td>Petrol</td>
      <td>Automatic</td>
      <td>Third</td>
      <td>17.2 kmpl</td>
      <td>1197 CC</td>
      <td>103.6 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7252</th>
      <td>7252</td>
      <td>Mercedes-Benz E-Class 2009-2013 E 220 CDI Avan...</td>
      <td>Kochi</td>
      <td>2014</td>
      <td>72443</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>10.0 kmpl</td>
      <td>2148 CC</td>
      <td>170 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Random ten rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.No.</th>
      <th>Name</th>
      <th>Location</th>
      <th>Year</th>
      <th>Kilometers_Driven</th>
      <th>Fuel_Type</th>
      <th>Transmission</th>
      <th>Owner_Type</th>
      <th>Mileage</th>
      <th>Engine</th>
      <th>Power</th>
      <th>Seats</th>
      <th>New_Price</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2397</th>
      <td>2397</td>
      <td>Ford EcoSport 1.5 Petrol Trend</td>
      <td>Kolkata</td>
      <td>2016</td>
      <td>21460</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>17.0 kmpl</td>
      <td>1497 CC</td>
      <td>121.36 bhp</td>
      <td>5.0</td>
      <td>9.47 Lakh</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>3777</th>
      <td>3777</td>
      <td>Maruti Wagon R VXI 1.2</td>
      <td>Kochi</td>
      <td>2015</td>
      <td>49818</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>21.5 kmpl</td>
      <td>1197 CC</td>
      <td>81.80 bhp</td>
      <td>5.0</td>
      <td>5.44 Lakh</td>
      <td>4.11</td>
    </tr>
    <tr>
      <th>4425</th>
      <td>4425</td>
      <td>Ford Endeavour 4x2 XLT</td>
      <td>Hyderabad</td>
      <td>2007</td>
      <td>130000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>13.1 kmpl</td>
      <td>2499 CC</td>
      <td>141 bhp</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>3661</th>
      <td>3661</td>
      <td>Mercedes-Benz E-Class E250 CDI Avantgrade</td>
      <td>Coimbatore</td>
      <td>2016</td>
      <td>39753</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>13.0 kmpl</td>
      <td>2143 CC</td>
      <td>201.1 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>35.28</td>
    </tr>
    <tr>
      <th>4514</th>
      <td>4514</td>
      <td>Hyundai Xcent 1.2 Kappa AT SX Option</td>
      <td>Kochi</td>
      <td>2016</td>
      <td>45560</td>
      <td>Petrol</td>
      <td>Automatic</td>
      <td>First</td>
      <td>16.9 kmpl</td>
      <td>1197 CC</td>
      <td>82 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>6.34</td>
    </tr>
    <tr>
      <th>599</th>
      <td>599</td>
      <td>Toyota Innova Crysta 2.8 ZX AT</td>
      <td>Coimbatore</td>
      <td>2019</td>
      <td>40674</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>11.36 kmpl</td>
      <td>2755 CC</td>
      <td>171.5 bhp</td>
      <td>7.0</td>
      <td>28.05 Lakh</td>
      <td>24.82</td>
    </tr>
    <tr>
      <th>186</th>
      <td>186</td>
      <td>Mercedes-Benz E-Class E250 CDI Avantgrade</td>
      <td>Bangalore</td>
      <td>2014</td>
      <td>37382</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>13.0 kmpl</td>
      <td>2143 CC</td>
      <td>201.1 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>32.00</td>
    </tr>
    <tr>
      <th>305</th>
      <td>305</td>
      <td>Audi A6 2011-2015 2.0 TDI Premium Plus</td>
      <td>Kochi</td>
      <td>2014</td>
      <td>61726</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>17.68 kmpl</td>
      <td>1968 CC</td>
      <td>174.33 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>20.77</td>
    </tr>
    <tr>
      <th>4582</th>
      <td>4582</td>
      <td>Hyundai i20 1.2 Magna</td>
      <td>Kolkata</td>
      <td>2011</td>
      <td>36000</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>18.5 kmpl</td>
      <td>1197 CC</td>
      <td>80 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>5434</th>
      <td>5434</td>
      <td>Honda WR-V Edge Edition i-VTEC S</td>
      <td>Kochi</td>
      <td>2019</td>
      <td>13913</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>17.5 kmpl</td>
      <td>1199 CC</td>
      <td>88.7 bhp</td>
      <td>5.0</td>
      <td>9.36 Lakh</td>
      <td>8.20</td>
    </tr>
  </tbody>
</table>
</div>


**Observations:**
* The **S.No.** is simply a row identifier which can be removed later.
* The **Name** is long string comprising the Brand, Model and addition specification. These can be split to further analyse by make and model.
* The **Location** is a geographical location. We can possibly create a heat map of car make by location.
* The **Fuel_Type**, **Transmission** and **Owner_Type** are categorical variable which can be possibly be onehot encoded.
* **Mileage** is a string leading with number and unit. This shall be split into the mileage number and mileage unit. Further analysis will determine if the mileage number to be converted based on the units.
* **Engine** and **Power** both are strings leading with number and unit. This shall be split and numerical portion will be used in analysis.
* **Seats** is a discrete numerical variable.
* **New_Price** has many missing values. Where it contains a value the currency term shall be removed by splitting.
* **Price** has missing values

For ease of programming analysis the dataframe column names shall be converted to lower case.


```python
data.columns = [col.lower() for col in data.columns]
```

Let's check the first row in the dataframe to verify the change


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>s.no.</th>
      <th>name</th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>mileage</th>
      <th>engine</th>
      <th>power</th>
      <th>seats</th>
      <th>new_price</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Maruti Wagon R LXI CNG</td>
      <td>Mumbai</td>
      <td>2010</td>
      <td>72000</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>26.6 km/kg</td>
      <td>998 CC</td>
      <td>58.16 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.75</td>
    </tr>
  </tbody>
</table>
</div>



Columns names are in lower case

---

## Variable List


```python
# Display list of variables in dataset
variable_list = data.columns.tolist()
print(variable_list)
```

    ['s.no.', 'name', 'location', 'year', 'kilometers_driven', 'fuel_type', 'transmission', 'owner_type', 'mileage', 'engine', 'power', 'seats', 'new_price', 'price']
    

---

## Get the shape of the dataset


```python
shape = data.shape
n_rows = shape[0]
n_cols = shape[1]
print(f"The Dataframe consists of '{n_rows}' rows and '{n_cols}' columns")
```

    The Dataframe consists of '7253' rows and '14' columns
    

---

## Data info


```python
# Get info of the dataframe columns
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7253 entries, 0 to 7252
    Data columns (total 14 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   s.no.              7253 non-null   int64  
     1   name               7253 non-null   object 
     2   location           7253 non-null   object 
     3   year               7253 non-null   int64  
     4   kilometers_driven  7253 non-null   int64  
     5   fuel_type          7253 non-null   object 
     6   transmission       7253 non-null   object 
     7   owner_type         7253 non-null   object 
     8   mileage            7251 non-null   object 
     9   engine             7207 non-null   object 
     10  power              7207 non-null   object 
     11  seats              7200 non-null   float64
     12  new_price          1006 non-null   object 
     13  price              6019 non-null   float64
    dtypes: float64(2), int64(3), object(9)
    memory usage: 793.4+ KB
    

**Observations:**
* There seems to be some missing values across 6 columns.

---

**Panda Object Variable states**


```python
# Panda Object Variable states function

def pandas_object_states(data):
    """
    This function checks if the variable type is pandas Object and
    displays the states and counts of each
    """
    # Loop through all variables
    for var in data.columns:
        # Check for pandas Object type
        if data[var].dtypes == "object":
            print('Unique values in', var, 'are :')
            print(data[var].value_counts().sort_index())
            print('--'*55)
```


```python
# Check the states of all pandas Object variables
pandas_object_states(data)
```

    Unique values in name are :
    name
    Ambassador Classic Nova Diesel    1
    Audi A3 35 TDI Attraction         2
    Audi A3 35 TDI Premium            1
    Audi A3 35 TDI Premium Plus       2
    Audi A3 35 TDI Technology         1
                                     ..
    Volvo XC60 D4 Summum              1
    Volvo XC60 D5                     3
    Volvo XC60 D5 Inscription         1
    Volvo XC90 2007-2015 D5 AT AWD    1
    Volvo XC90 2007-2015 D5 AWD       3
    Name: count, Length: 2041, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in location are :
    location
    Ahmedabad     275
    Bangalore     440
    Chennai       591
    Coimbatore    772
    Delhi         660
    Hyderabad     876
    Jaipur        499
    Kochi         772
    Kolkata       654
    Mumbai        949
    Pune          765
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in fuel_type are :
    fuel_type
    CNG           62
    Diesel      3852
    Electric       2
    LPG           12
    Petrol      3325
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in transmission are :
    transmission
    Automatic    2049
    Manual       5204
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in owner_type are :
    owner_type
    First             5952
    Fourth & Above      12
    Second            1152
    Third              137
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in mileage are :
    mileage
    0.0 kmpl      81
    10.0 kmpl     13
    10.1 kmpl     10
    10.13 kmpl     4
    10.2 kmpl      9
                  ..
    9.52 kmpl      2
    9.7 kmpl       1
    9.74 kmpl      4
    9.8 kmpl       4
    9.9 kmpl       3
    Name: count, Length: 450, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in engine are :
    engine
    1047 CC      5
    1061 CC     32
    1086 CC    129
    1120 CC     60
    1150 CC      8
              ... 
    970 CC       1
    993 CC      14
    995 CC      15
    998 CC     309
    999 CC      36
    Name: count, Length: 150, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in power are :
    power
    100 bhp       69
    100.6 bhp     50
    101 bhp        4
    102 bhp       71
    102.5 bhp      9
                ... 
    98.82 bhp      3
    98.96 bhp     11
    99 bhp        18
    99.6 bhp       8
    null bhp     129
    Name: count, Length: 386, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in new_price are :
    new_price
    1 Cr          1
    1.02 Cr       1
    1.04 Cr       2
    1.06 Cr       2
    1.13 Cr       1
                 ..
    92.79 Lakh    1
    95.04 Lakh    1
    95.13 Lakh    6
    95.38 Lakh    1
    99.92 Lakh    1
    Name: count, Length: 625, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    

---

**Missing value summary function**


```python
def missing_val_chk(data):
    """
    This function to checks for missing values 
    and generates a summary.
    """
    if data.isnull().sum().any() == True:
        # Number of missing in each column
        missing_vals = pd.DataFrame(data.isnull().sum().sort_values(
            ascending=False)).rename(columns={0: '# missing'})

        # Create a percentage missing
        missing_vals['percent'] = ((missing_vals['# missing'] / len(data)) *
                                   100).round(decimals=3)

        # Remove rows with 0
        missing_vals = missing_vals[missing_vals['# missing'] != 0].dropna()

        # display missing value dataframe
        print("The missing values summary")
        display(missing_vals)
    else:
        print("There are NO missing values in the dataset")
```


```python
#Applying the missing value summary function
missing_val_chk(data)
```

    The missing values summary
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># missing</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>new_price</th>
      <td>6247</td>
      <td>86.130</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1234</td>
      <td>17.014</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>53</td>
      <td>0.731</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>46</td>
      <td>0.634</td>
    </tr>
    <tr>
      <th>power</th>
      <td>46</td>
      <td>0.634</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>2</td>
      <td>0.028</td>
    </tr>
  </tbody>
</table>
</div>


**There a lot of missing values that need to be dealt with.**

---

## 5 Point Summary

**Numerical type Summary**


```python
# Five point summary of all numerical type variables in the dataset
data.describe().T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>s.no.</th>
      <td>7253.0</td>
      <td>3626.000000</td>
      <td>2093.905084</td>
      <td>0.00</td>
      <td>1813.0</td>
      <td>3626.00</td>
      <td>5439.00</td>
      <td>7252.0</td>
    </tr>
    <tr>
      <th>year</th>
      <td>7253.0</td>
      <td>2013.365366</td>
      <td>3.254421</td>
      <td>1996.00</td>
      <td>2011.0</td>
      <td>2014.00</td>
      <td>2016.00</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>kilometers_driven</th>
      <td>7253.0</td>
      <td>58699.063146</td>
      <td>84427.720583</td>
      <td>171.00</td>
      <td>34000.0</td>
      <td>53416.00</td>
      <td>73000.00</td>
      <td>6500000.0</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>7200.0</td>
      <td>5.279722</td>
      <td>0.811660</td>
      <td>0.00</td>
      <td>5.0</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>6019.0</td>
      <td>9.479468</td>
      <td>11.187917</td>
      <td>0.44</td>
      <td>3.5</td>
      <td>5.64</td>
      <td>9.95</td>
      <td>160.0</td>
    </tr>
  </tbody>
</table>
</div>



**Categorical type Summary**


```python
data.describe(include=['object']).T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>name</th>
      <td>7253</td>
      <td>2041</td>
      <td>Mahindra XUV500 W8 2WD</td>
      <td>55</td>
    </tr>
    <tr>
      <th>location</th>
      <td>7253</td>
      <td>11</td>
      <td>Mumbai</td>
      <td>949</td>
    </tr>
    <tr>
      <th>fuel_type</th>
      <td>7253</td>
      <td>5</td>
      <td>Diesel</td>
      <td>3852</td>
    </tr>
    <tr>
      <th>transmission</th>
      <td>7253</td>
      <td>2</td>
      <td>Manual</td>
      <td>5204</td>
    </tr>
    <tr>
      <th>owner_type</th>
      <td>7253</td>
      <td>4</td>
      <td>First</td>
      <td>5952</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>7251</td>
      <td>450</td>
      <td>17.0 kmpl</td>
      <td>207</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>7207</td>
      <td>150</td>
      <td>1197 CC</td>
      <td>732</td>
    </tr>
    <tr>
      <th>power</th>
      <td>7207</td>
      <td>386</td>
      <td>74 bhp</td>
      <td>280</td>
    </tr>
    <tr>
      <th>new_price</th>
      <td>1006</td>
      <td>625</td>
      <td>63.71 Lakh</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



**Observation:**
* The `Mileage`, `Engine`, `Power` and `New_Price` are strings with numerical values. We need to extract the numerical values for further analysis.
* **Seats** - There minimum no of seats definitely cant be 0. This may be an error would be address later on.
* **There is a significant amount of data pre-processing required before we can explore the dataset.**

---

---
