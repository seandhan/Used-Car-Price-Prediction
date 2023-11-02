# Feature Engineering

## `s.no.` column

**Check if column `s.no.` simliar to index**

Above we have identified that the `s.no.` is simply a row identifier.  
Let's confirm this before we drop it later.


```python
# Extract the index values and convert to Pandas Series for comparison
idx = pd.Series(data.index.values)

# Extract the first column
first_col = data['s.no.']
```

**Compare Series idx and first_col**


```python
first_col.equals(idx)
```




    True



Therefore, let's drop the `s.no.` column


```python
data.drop(columns="s.no.",inplace=True)
```

Let's do a check to see if **s.no.** is in the dropped dataframe


```python
"s.no." in data.columns
```




    False



Lets preview the dataframe to see if the column has been dropped


```python
data.head()
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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



**s.no.** column dropped!!

---

## `name` column

The `name` is long string comprising the car make, car model and additional specification.   
Let's split and uppercase the name column into `car_make`, `car_model` and `car_spec`.
The conversion to uppercase will clear any string case inconsistencies when the dataset was being built.
This will enable further analysis in car data.


```python
# The name will be split by the " " separator
# The first term will be assigned to the car_make
data["car_make"] = data.name.str.split(' ').str[0].str.upper()

# The second term will be assigned to the car_model
data["car_model"] = data.name.str.split(' ').str[1].str.upper()

# The third and remaining term will be joined using " " and assigned to the car_spec
data["car_spec"] = data.name.apply(lambda x: str.upper(" ".join(x.split(" ")[2:])))
```

Let's check the first row in the dataframe to verify the additional of the car columns


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
    </tr>
  </tbody>
</table>
</div>



Let's drop the name column


```python
data.drop(columns = 'name', inplace=True)
```

Now let's see the number of unique car_make (brands)


```python
data.car_make.nunique()
```




    32



Now let's see the counts by car make


```python
data.car_make.value_counts()
```




    car_make
    MARUTI           1444
    HYUNDAI          1340
    HONDA             743
    TOYOTA            507
    MERCEDES-BENZ     380
    VOLKSWAGEN        374
    FORD              351
    MAHINDRA          331
    BMW               312
    AUDI              285
    TATA              228
    SKODA             202
    RENAULT           170
    CHEVROLET         151
    NISSAN            117
    LAND               67
    JAGUAR             48
    FIAT               38
    MITSUBISHI         36
    MINI               31
    VOLVO              28
    PORSCHE            19
    JEEP               19
    DATSUN             17
    ISUZU               5
    FORCE               3
    BENTLEY             2
    SMART               1
    AMBASSADOR          1
    LAMBORGHINI         1
    HINDUSTAN           1
    OPELCORSA           1
    Name: count, dtype: int64



Now let's see the number of unique car_model (models)


```python
data.car_model.nunique()
```




    218



Now let's see the counts by car model


```python
data.car_model.value_counts()
```




    car_model
    SWIFT      418
    CITY       318
    I20        303
    INNOVA     203
    VERNA      200
              ... 
    FORTWO       1
    REDI         1
    E            1
    MUSTANG      1
    1.4GSI       1
    Name: count, Length: 218, dtype: int64



**Observation:**
* There are 32 different car brands of which MARUTI is the most popular
* The dataset has 218 different car models while SWIFT is the most popular

Now let's see the number of unique car_spec (specifications)


```python
data.car_spec.nunique()
```




    1860



---

## `location` column

**Now let's see the number of cities**


```python
data.location.nunique()
```




    11



**Now let's see the counts by city**


```python
data.location.value_counts()
```




    location
    Mumbai        949
    Hyderabad     876
    Coimbatore    772
    Kochi         772
    Pune          765
    Delhi         660
    Kolkata       654
    Chennai       591
    Jaipur        499
    Bangalore     440
    Ahmedabad     275
    Name: count, dtype: int64



**Observation:**
* There are 11 different cities of which `Mumbai` is the most popular and `Ahmedabad` is the least popular.

**Group `location` by region**


```python
# Define a dictionary to map locations to regions
region_mapping = {
    'Chennai': 'South',
    'Jaipur': 'North',
    'Pune': 'West',
    'Mumbai': 'West',
    'Kolkata': 'East',
    'Delhi': 'North',
    'Ahmedabad': 'West',
    'Hyderabad': 'South',
    'Coimbatore': 'South',
    'Bangalore': 'South',
    'Kochi': 'South'
}

# Use the dictionary to map locations to regions
data['region'] = data['location'].map(region_mapping)
```

---

## `year` column

**Now let's see the number of years**


```python
data.year.nunique()
```




    23



**Now let's see the counts by year**


```python
data.year.value_counts()
```




    year
    2015    929
    2014    925
    2016    886
    2013    791
    2017    709
    2012    690
    2011    579
    2010    407
    2018    361
    2009    252
    2008    207
    2007    148
    2019    119
    2006     89
    2005     68
    2004     35
    2003     20
    2002     18
    2001      8
    2000      5
    1998      4
    1999      2
    1996      1
    Name: count, dtype: int64



**Observation:**
* There are 23 different years of manufacture of which **2015** is the most popular and **1996** is the least popular.

Since the 'year' column represents the year of manufacture, let's get the age of the car by subtracting it from the year the survey was done.  
We shall use **2019** for the 2018-2019 period

Since we have seen that `year` datatype is **int64** we will need to converted to *datetime*.


```python
data["car_age"]=2019-data.year
```

Let's check the first row in the dataframe to verify the additional of the `car_age` column


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



**Now let's see the car age counts**


```python
data.car_age.value_counts()
```




    car_age
    4     929
    5     925
    3     886
    6     791
    2     709
    7     690
    8     579
    9     407
    1     361
    10    252
    11    207
    12    148
    0     119
    13     89
    14     68
    15     35
    16     20
    17     18
    18      8
    19      5
    21      4
    20      2
    23      1
    Name: count, dtype: int64



**Observation:**
* In the dataset the most popular used car age is `4` years old while the oldest and least popular used car age is `23`.


---

## `fuel_type` column

**Now let's see the number of fuel types**


```python
data.fuel_type.nunique()
```




    5



**Now let's see the counts by fuel type**


```python
data.fuel_type.value_counts()
```




    fuel_type
    Diesel      3852
    Petrol      3325
    CNG           62
    LPG           12
    Electric       2
    Name: count, dtype: int64



**Observation:**
* Diesel and Petrol are the most popular fuel types.

---

## `transmission` column

**Now let's see the number of different transmissions**


```python
data.transmission.nunique()
```




    2



**Now let's see the counts by transmissions**


```python
data.transmission.value_counts()
```




    transmission
    Manual       5204
    Automatic    2049
    Name: count, dtype: int64



**Observation:**
* `Manual transmission` is the most popular type in the dataset.

---

## `owner_type` column

**Now let's see the number of different car owner types**


```python
data.owner_type.nunique()
```




    4



**Now let's see the counts by car owner type**


```python
data.owner_type.value_counts()
```




    owner_type
    First             5952
    Second            1152
    Third              137
    Fourth & Above      12
    Name: count, dtype: int64



**Observation:**
* `First owner` is the most popular type in the dataset.

---

## `mileage` column

`mileage` contains string leading with number and unit. This shall be split into the mileage number and mileage unit. 


Further analysis will determine the following:
* If the there is only one unit in the `mileage_unit` then the `mileage_unit` column can be dropped.
* If the there is more than one unit in the `mileage_unit` then the `mileage_no` can be converted to normalise all values for analysis (based on a conversion factor).

Let's split the `mileage` column into `mileage_no` and `mileage_unit`
This will enable further analysis in car data.


```python
# The mileage will be split by the " " separator
# The first term will be assigned to the mileage_no
data["mileage_no"] = data.mileage.str.split(' ').str[0]
# # The second term will be assigned to the mileage_unit
data["mileage_unit"] = data.mileage.str.split(' ').str[1]
```

Let's check the first row in the dataframe to verify the additional of the car columns


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage_no</th>
      <th>mileage_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>9</td>
      <td>26.6</td>
      <td>km/kg</td>
    </tr>
  </tbody>
</table>
</div>



Now let's see the number of unique `mileage_unit`


```python
data.mileage_unit.nunique()
```




    2



Now let's see the counts by `mileage_unit`


```python
data.mileage_unit.value_counts()
```




    mileage_unit
    kmpl     7177
    km/kg      74
    Name: count, dtype: int64



Let's explore the **km/kg** mileage unit


```python
data[data.mileage_unit=='km/kg']
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage_no</th>
      <th>mileage_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>9</td>
      <td>26.6</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hyderabad</td>
      <td>2012</td>
      <td>75000</td>
      <td>LPG</td>
      <td>Manual</td>
      <td>First</td>
      <td>21.1 km/kg</td>
      <td>814 CC</td>
      <td>55.2 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.35</td>
      <td>HYUNDAI</td>
      <td>EON</td>
      <td>LPG ERA PLUS OPTION</td>
      <td>South</td>
      <td>7</td>
      <td>21.1</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>127</th>
      <td>Pune</td>
      <td>2013</td>
      <td>89900</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>26.6 km/kg</td>
      <td>998 CC</td>
      <td>58.16 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>3.25</td>
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>6</td>
      <td>26.6</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>328</th>
      <td>Pune</td>
      <td>2008</td>
      <td>42496</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>26.3 km/kg</td>
      <td>998 CC</td>
      <td>67.1 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.40</td>
      <td>MARUTI</td>
      <td>ZEN</td>
      <td>ESTILO LXI GREEN (CNG)</td>
      <td>West</td>
      <td>11</td>
      <td>26.3</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>440</th>
      <td>Kochi</td>
      <td>2017</td>
      <td>31841</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>15.1 km/kg</td>
      <td>1196 CC</td>
      <td>73 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>4.70</td>
      <td>MARUTI</td>
      <td>EECO</td>
      <td>5 STR WITH AC PLUS HTR CNG</td>
      <td>South</td>
      <td>2</td>
      <td>15.1</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6538</th>
      <td>Kochi</td>
      <td>2015</td>
      <td>47490</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>32.26 km/kg</td>
      <td>998 CC</td>
      <td>58.2 bhp</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MARUTI</td>
      <td>ALTO</td>
      <td>K10 LXI CNG</td>
      <td>South</td>
      <td>4</td>
      <td>32.26</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>6788</th>
      <td>Delhi</td>
      <td>2017</td>
      <td>33000</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>32.26 km/kg</td>
      <td>998 CC</td>
      <td>58.2 bhp</td>
      <td>4.0</td>
      <td>4.66 Lakh</td>
      <td>NaN</td>
      <td>MARUTI</td>
      <td>ALTO</td>
      <td>K10 LXI CNG OPTIONAL</td>
      <td>North</td>
      <td>2</td>
      <td>32.26</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>6918</th>
      <td>Jaipur</td>
      <td>2010</td>
      <td>40000</td>
      <td>LPG</td>
      <td>Manual</td>
      <td>Second</td>
      <td>13.2 km/kg</td>
      <td>1495 CC</td>
      <td>93.7 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HYUNDAI</td>
      <td>ACCENT</td>
      <td>EXECUTIVE LPG</td>
      <td>North</td>
      <td>9</td>
      <td>13.2</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>6926</th>
      <td>Mumbai</td>
      <td>2017</td>
      <td>15500</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>15.1 km/kg</td>
      <td>1196 CC</td>
      <td>73 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MARUTI</td>
      <td>EECO</td>
      <td>5 STR WITH AC PLUS HTR CNG</td>
      <td>West</td>
      <td>2</td>
      <td>15.1</td>
      <td>km/kg</td>
    </tr>
    <tr>
      <th>6958</th>
      <td>Kolkata</td>
      <td>2013</td>
      <td>14311</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>30.46 km/kg</td>
      <td>796 CC</td>
      <td>47.3 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MARUTI</td>
      <td>ALTO</td>
      <td>800 CNG LXI</td>
      <td>East</td>
      <td>6</td>
      <td>30.46</td>
      <td>km/kg</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 19 columns</p>
</div>



Let's explore the fuel types for km/kg mileage unit


```python
data[data.mileage_unit=='km/kg'].fuel_type.value_counts()
```




    fuel_type
    CNG    62
    LPG    12
    Name: count, dtype: int64



Let's explore the transmissions for km/kg mileage unit


```python
data[data.mileage_unit=='km/kg'].transmission.value_counts()
```




    transmission
    Manual    74
    Name: count, dtype: int64



**Observation:**
* There are 2 different mileage units of which **kmpl** is the most popular
* The **km/kg** accounts for ~1% of the dataset.
* Since there is no empirical conversion for `kmpl` to `km/kg`, let's leave it in the dataset.
* Cars that have mileage unit as **km/kg** are `Manual` transmission powered by either `CNG` or `LPG`

Since we dont need the initial **mileage** column anymore, let's drop the **mileage** column


```python
data.drop(columns='mileage',inplace=True)
```

For consistency, lets rename **mileage_no** to **mileage**


```python
data = data.rename(columns={'mileage_no': 'mileage'})
```

Based on the string operations above the **mileage** column may still be a string.  
Let's convert to a float.


```python
if data.mileage.dtypes != np.number:
    # This will convert all numeric strings to float.
    # If a value is non numeric it will set to NaN
    data['mileage'] = pd.to_numeric(data.mileage, errors='coerce')
```

---

## `engine` column

The `engine` column contains strings leading with number and unit. This shall be split into the `engine_disp` and `eng_disp_unit` unit. 

Further analysis will determine the following:
* If the there is only one unit in the `eng_disp_unit` then the `eng_disp_unit` column can be dropped.
* If the there is more than one unit in the `eng_disp_unit` then the `eng_disp_unit` can be converted to normalise all values for analysis (based on a conversion factor).

Let's split the `engine` column into `engine_disp` and `eng_disp_unit`
This will enable further analysis in car data.


```python
# The engine will be split by the " " separator
# The first term will be assigned to the engine_disp
data["engine_disp"] = data.engine.str.split(' ').str[0]
# The second term will be assigned to the eng_disp_unit
data["eng_disp_unit"] = data.engine.str.split(' ').str[1]
```

Let's check the first row in the dataframe to verify the additional of the car columns


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>engine</th>
      <th>power</th>
      <th>seats</th>
      <th>new_price</th>
      <th>price</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>engine_disp</th>
      <th>eng_disp_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mumbai</td>
      <td>2010</td>
      <td>72000</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>998 CC</td>
      <td>58.16 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.75</td>
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>9</td>
      <td>26.6</td>
      <td>km/kg</td>
      <td>998</td>
      <td>CC</td>
    </tr>
  </tbody>
</table>
</div>



Now let's see the number of unique `eng_disp_unit`


```python
data.eng_disp_unit.nunique()
```




    1



Since there is only type of unit in the eng_disp_unit, this would not add significant value to the analysis.  
Let's drop the `eng_disp_unit` column


```python
data.drop(columns = 'eng_disp_unit', inplace=True)
```

Since we dont need the initial `engine` column anymore, let's drop the `engine` column


```python
data.drop(columns='engine',inplace=True)
```

For consistency, let's rename `engine_disp` to `engine`


```python
data = data.rename(columns={'engine_disp': 'engine'})
```

Based on the string operations above the **engine** column may still be a string.  
Let's convert to a float.


```python
if data.engine.dtypes != np.number:
    # This will convert all numeric strings to float.
    # If a value is non numeric it will set to NaN
    data['engine'] = pd.to_numeric(data.engine, errors='coerce')
```

---

## `power` column

The `power` column contains strings leading with number and unit. This shall be split into the `power_val` and `power_unit` unit. 

Further analysis will determine the following:
* If the there is only one unit in the `power_unit` then the `power_unit` column can be dropped.
* If the there is more than one unit in the `power_unit` then the `power_unit` can be converted to normalise all values for analysis (based on a conversion factor).

Let's split the `power` column into `power_val` and `power_unit`.
This will enable further analysis in car data.


```python
# The engine will be split by the " " separator
# The first term will be assigned to the engine_disp
data["power_val"] = data.power.str.split(' ').str[0]
# The second term will be assigned to the eng_disp_unit
data["power_unit"] = data.power.str.split(' ').str[1]
```

Let's check the first row in the dataframe to verify the additional of the car columns


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>power</th>
      <th>seats</th>
      <th>new_price</th>
      <th>price</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>engine</th>
      <th>power_val</th>
      <th>power_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mumbai</td>
      <td>2010</td>
      <td>72000</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>58.16 bhp</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.75</td>
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>9</td>
      <td>26.6</td>
      <td>km/kg</td>
      <td>998.0</td>
      <td>58.16</td>
      <td>bhp</td>
    </tr>
  </tbody>
</table>
</div>



**Now let's see the number of unique power_unit**


```python
data.power_unit.nunique()
```




    1



Since there is only type of unit in the power_unit.  
**Let's drop the power_unit column**


```python
data.drop(columns='power_unit', inplace=True)
```

Since we dont need the initial **power** column anymore, let's drop the **power** column


```python
data.drop(columns='power', inplace=True)
```

For consistency, let's rename **power_val** to **power**


```python
data = data.rename(columns={'power_val': 'power'})
```

Based on the string operations above the **power** column may still be a string.  
Let's convert to a float.


```python
if data.power.dtypes != np.number:
    # This will convert all numeric strings to float.
    # If a value is non numeric it will set to NaN
    data['power'] = pd.to_numeric(data.power, errors='coerce')
```

---

## `seats` column

**Now let's see the number of different seats**


```python
data.seats.nunique()
```




    9



**Now let's see the counts by seats**


```python
data.seats.value_counts()
```




    seats
    5.0     6047
    7.0      796
    8.0      170
    4.0      119
    6.0       38
    2.0       18
    10.0       8
    9.0        3
    0.0        1
    Name: count, dtype: int64



**Observation:**
* Cars with 5 seats is the most popular option.
* There is one instance where a car has 0 seats, this may be an error in the row as every production vehicle has at least one seat, which is the driver seat.


Lets see the row that has 0 seats.


```python
data[data.seats==0]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>seats</th>
      <th>new_price</th>
      <th>price</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>engine</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3999</th>
      <td>Hyderabad</td>
      <td>2012</td>
      <td>125000</td>
      <td>Petrol</td>
      <td>Automatic</td>
      <td>First</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>AUDI</td>
      <td>A4</td>
      <td>3.2 FSI TIPTRONIC QUATTRO</td>
      <td>South</td>
      <td>7</td>
      <td>10.5</td>
      <td>kmpl</td>
      <td>3197.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Observation:**
* Audi A4 are sedans with more than seat. This may have been an error when compiling the dataset.

Let's create a sub dataframe of Audi A4s and check the median seats.


```python
audi_a4 = data[data.car_make=="AUDI"][data.car_model=="A4"]
print("The Audi A4 median seats = {}".format(audi_a4.seats.median()))
```

    The Audi A4 median seats = 5.0
    

Let's replace the seats value of 0 with 5


```python
data.seats.replace(to_replace=0, value= 5.0, inplace=True)
```

Let's check to see if seats min value was changed


```python
if data.seats.min()!=0:
    print("The seats min value was updated")
```

    The seats min value was updated
    

---

## `new_price` column

From above we have seen that the `new_price` contains missing values as well strings with cost in Lakh

**Now let's see the number of different strings for new_price**


```python
data.new_price.nunique()
```




    625



Now let's see the counts by **new_price** to see if there are similar strings


```python
data.new_price.value_counts()
```




    new_price
    63.71 Lakh    6
    95.13 Lakh    6
    4.78 Lakh     6
    33.36 Lakh    6
    44.28 Lakh    5
                 ..
    6.83 Lakh     1
    9.09 Lakh     1
    7.17 Lakh     1
    11.77 Lakh    1
    18.16 Lakh    1
    Name: count, Length: 625, dtype: int64



**Observation:**
* The majority of the new_price strings contain "Lakh" while there is an instance of "Cr"

Let's split the **new_price** column into **new_price_no** and **new_price_unit**.
This will enable further analysis in car data.


```python
# The new_price will be split by the " " separator
# The first term will be assigned to the new_price_no
data["new_price_no"] = data.new_price.str.split(' ').str[0]
# The second term will be assigned to the new_price_unit
data["new_price_unit"] = data.new_price.str.split(' ').str[1]
```

Let's check the first row in the dataframe to verify the additional of the car columns


```python
data.head(1)
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>seats</th>
      <th>new_price</th>
      <th>price</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>engine</th>
      <th>power</th>
      <th>new_price_no</th>
      <th>new_price_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mumbai</td>
      <td>2010</td>
      <td>72000</td>
      <td>CNG</td>
      <td>Manual</td>
      <td>First</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.75</td>
      <td>MARUTI</td>
      <td>WAGON</td>
      <td>R LXI CNG</td>
      <td>West</td>
      <td>9</td>
      <td>26.6</td>
      <td>km/kg</td>
      <td>998.0</td>
      <td>58.16</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Now let's see the number of different strings for new_price_unit**


```python
data.new_price_unit.nunique()
```




    2



**Now let's see the counts by new_price_unit to see if there are similar strings**


```python
data.new_price_unit.value_counts()
```




    new_price_unit
    Lakh    986
    Cr       20
    Name: count, dtype: int64



There 2 different new_price_units. 
Since the Data Dictionary has Lakh has the unit of price
Let's convert the instances of "Cr" to "Lakh"
**new_price_no** based on the following: 
* 1 Cr = 100 Lakh 

Convert `new_price_no` from *Cr INR* to *Lakh INR*. Then drop the `new_price_unit` column


```python
# Convert to a string to numeric
if data.new_price_no.dtypes != np.number:
    # This will convert all numeric strings to float.
    # If a value is non numeric it will set to NaN
    data['new_price_no'] = pd.to_numeric(data.new_price_no, errors='coerce')

# Conversion factor
Cr_conv = 100

# Multiplier function to convert instance of new_price_no in Cr to Lakh
data.loc[data['new_price_unit'] == 'Cr', ['new_price_no']] *= Cr_conv

# Drop 'new_price_unit' column
data = data.drop(columns='new_price_unit')
```

Since we dont need the initial `new_price` column anymore, let's drop the `new_price` column


```python
data.drop(columns='new_price', inplace=True)
```

For consistency, let's rename **new_price_no** to **new_price**


```python
data = data.rename(columns={'new_price_no': 'new_price'})
```

---

## Check for duplicated rows

Let's check if there are any duplicated rows in the dataset

**Duplicate row check function**


```python
def df_duplicate_removal(data):
    """
    This function checks if there are any duplicated rows in the dataframe.
    If any, it displays the rows, keep the first occurence and drops the 
    duplicates.

    The new dataframe shape is calculated and returned
    """
    # Check if duplicated rows exist
    if any(data.duplicated() == True):
        print("The following is/are the duplicated row(s) in the dataframe")
        # Displays the duplicated row(s)
        display(data[data.duplicated() == True])
        # Drops the duplicates inplace while keeping the first occurence
        data.drop_duplicates(keep="first", inplace=True)

        # Check and returns the shape of the new dataframe
        new_df_shape = data.shape
        n_rows = new_df_shape[0]
        n_cols = new_df_shape[1]
        print(
            f"The new dataframe  consists of '{n_rows}' rows and '{n_cols}' columns")
    else:
        print("There is/are no duplicated row(s) in the dataframe")
```


```python
df_duplicate_removal(data)
```

    The following is/are the duplicated row(s) in the dataframe
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>seats</th>
      <th>price</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>engine</th>
      <th>power</th>
      <th>new_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4781</th>
      <td>Hyderabad</td>
      <td>2007</td>
      <td>52195</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>5.0</td>
      <td>1.75</td>
      <td>MARUTI</td>
      <td>ALTO</td>
      <td>LXI</td>
      <td>South</td>
      <td>12</td>
      <td>19.7</td>
      <td>kmpl</td>
      <td>796.0</td>
      <td>46.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6582</th>
      <td>Mumbai</td>
      <td>2010</td>
      <td>52000</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>HONDA</td>
      <td>CITY</td>
      <td>1.5 E MT</td>
      <td>West</td>
      <td>9</td>
      <td>17.0</td>
      <td>kmpl</td>
      <td>1497.0</td>
      <td>118.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7078</th>
      <td>Kolkata</td>
      <td>2017</td>
      <td>13000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>HONDA</td>
      <td>CITY</td>
      <td>I DTEC SV</td>
      <td>East</td>
      <td>2</td>
      <td>26.0</td>
      <td>kmpl</td>
      <td>1498.0</td>
      <td>98.6</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    The new dataframe  consists of '7250' rows and '18' columns
    

---

## Data type conversion

Datatype conversion Converting pandas object types to category


```python
# Convert variables with "object" type to "category" type
for i in data.columns:
    if data[i].dtypes == "object":
        data[i] = data[i].astype("category")

# Confirm if there no variables with "object" type
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 7250 entries, 0 to 7252
    Data columns (total 18 columns):
     #   Column             Non-Null Count  Dtype   
    ---  ------             --------------  -----   
     0   location           7250 non-null   category
     1   year               7250 non-null   int64   
     2   kilometers_driven  7250 non-null   int64   
     3   fuel_type          7250 non-null   category
     4   transmission       7250 non-null   category
     5   owner_type         7250 non-null   category
     6   seats              7197 non-null   float64 
     7   price              6018 non-null   float64 
     8   car_make           7250 non-null   category
     9   car_model          7250 non-null   category
     10  car_spec           7250 non-null   category
     11  region             7250 non-null   category
     12  car_age            7250 non-null   int64   
     13  mileage            7248 non-null   float64 
     14  mileage_unit       7248 non-null   category
     15  engine             7204 non-null   float64 
     16  power              7075 non-null   float64 
     17  new_price          1006 non-null   float64 
    dtypes: category(9), float64(6), int64(3)
    memory usage: 735.7 KB
    

---

## The 5 Point Summary

Let's run the 5 point descriptive statistics again


```python
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
      <th>year</th>
      <td>7250.0</td>
      <td>2013.366207</td>
      <td>3.253715</td>
      <td>1996.00</td>
      <td>2011.000</td>
      <td>2014.00</td>
      <td>2016.0000</td>
      <td>2019.00</td>
    </tr>
    <tr>
      <th>kilometers_driven</th>
      <td>7250.0</td>
      <td>58707.187586</td>
      <td>84443.411557</td>
      <td>171.00</td>
      <td>34000.000</td>
      <td>53469.50</td>
      <td>73000.0000</td>
      <td>6500000.00</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>7197.0</td>
      <td>5.280534</td>
      <td>0.809426</td>
      <td>2.00</td>
      <td>5.000</td>
      <td>5.00</td>
      <td>5.0000</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>price</th>
      <td>6018.0</td>
      <td>9.480753</td>
      <td>11.188403</td>
      <td>0.44</td>
      <td>3.500</td>
      <td>5.64</td>
      <td>9.9500</td>
      <td>160.00</td>
    </tr>
    <tr>
      <th>car_age</th>
      <td>7250.0</td>
      <td>5.633793</td>
      <td>3.253715</td>
      <td>0.00</td>
      <td>3.000</td>
      <td>5.00</td>
      <td>8.0000</td>
      <td>23.00</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>7248.0</td>
      <td>18.140439</td>
      <td>4.562150</td>
      <td>0.00</td>
      <td>15.170</td>
      <td>18.16</td>
      <td>21.1000</td>
      <td>33.54</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>7204.0</td>
      <td>1616.720433</td>
      <td>595.327259</td>
      <td>72.00</td>
      <td>1198.000</td>
      <td>1493.00</td>
      <td>1968.0000</td>
      <td>5998.00</td>
    </tr>
    <tr>
      <th>power</th>
      <td>7075.0</td>
      <td>112.775871</td>
      <td>53.498756</td>
      <td>34.20</td>
      <td>75.000</td>
      <td>94.00</td>
      <td>138.1000</td>
      <td>616.00</td>
    </tr>
    <tr>
      <th>new_price</th>
      <td>1006.0</td>
      <td>22.779692</td>
      <td>27.759344</td>
      <td>3.91</td>
      <td>7.885</td>
      <td>11.57</td>
      <td>26.0425</td>
      <td>375.00</td>
    </tr>
  </tbody>
</table>
</div>



**Observation:**
* **year**  - The dataset has cars manufactured from 1996 to 2019
* **price** - There are many missing values. Approximately 17% is missing
* **car_age** - The mean and median car age are close.
* **mileage**  - The min value of 0 implies that the manufacturer supplied mileage stats was not input into the dataset.
* **engine** - The min value of 72 appears to an outlier and not realistic for car engine size. This will be dealt with later on.
* **new_price** - There are lot of missing values. Approximately 86% is missing.

## `mileage` column revisited

The `mileage` column has been observed to have a minimum of **0**.  
This implies that the manufacturer supplied mileage stats were not input into the dataset.  
Let's investigate further.

Lets check how many rows in the dataset has 0 in the mileage


```python
print("There are {} rows with '0' in the mileage".format(len(data[data.mileage==0])))
```

    There are 81 rows with '0' in the mileage
    

Lets view the rows with 0 mileage


```python
data[data.mileage==0]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>year</th>
      <th>kilometers_driven</th>
      <th>fuel_type</th>
      <th>transmission</th>
      <th>owner_type</th>
      <th>seats</th>
      <th>price</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>region</th>
      <th>car_age</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>engine</th>
      <th>power</th>
      <th>new_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Pune</td>
      <td>2012</td>
      <td>85000</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>Second</td>
      <td>5.0</td>
      <td>17.50</td>
      <td>LAND</td>
      <td>ROVER</td>
      <td>FREELANDER 2 TD4 SE</td>
      <td>West</td>
      <td>7</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>2179.0</td>
      <td>115.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Coimbatore</td>
      <td>2019</td>
      <td>15369</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>5.0</td>
      <td>35.67</td>
      <td>MERCEDES-BENZ</td>
      <td>C-CLASS</td>
      <td>PROGRESSIVE C 220D</td>
      <td>South</td>
      <td>0</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>1950.0</td>
      <td>194.0</td>
      <td>49.14</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Hyderabad</td>
      <td>2005</td>
      <td>87591</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>5.0</td>
      <td>1.30</td>
      <td>HYUNDAI</td>
      <td>SANTRO</td>
      <td>XING XL</td>
      <td>South</td>
      <td>14</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>1086.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Ahmedabad</td>
      <td>2007</td>
      <td>60006</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>NaN</td>
      <td>2.95</td>
      <td>HONDA</td>
      <td>CITY</td>
      <td>1.5 GXI</td>
      <td>West</td>
      <td>12</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Bangalore</td>
      <td>2015</td>
      <td>70436</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>NaN</td>
      <td>3.60</td>
      <td>FORD</td>
      <td>FIGO</td>
      <td>DIESEL</td>
      <td>South</td>
      <td>4</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>1498.0</td>
      <td>99.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6633</th>
      <td>Kolkata</td>
      <td>2016</td>
      <td>27000</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>First</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MAHINDRA</td>
      <td>TUV</td>
      <td>300 P4</td>
      <td>East</td>
      <td>3</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6697</th>
      <td>Jaipur</td>
      <td>2007</td>
      <td>85000</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>Second</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>HYUNDAI</td>
      <td>SANTRO</td>
      <td>XING XL</td>
      <td>North</td>
      <td>12</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>1086.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6857</th>
      <td>Mumbai</td>
      <td>2011</td>
      <td>87000</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>First</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>LAND</td>
      <td>ROVER</td>
      <td>FREELANDER 2 TD4 SE</td>
      <td>West</td>
      <td>8</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>2179.0</td>
      <td>115.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6957</th>
      <td>Kochi</td>
      <td>2019</td>
      <td>11574</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HONDA</td>
      <td>JAZZ</td>
      <td>2020 PETROL</td>
      <td>South</td>
      <td>0</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>1199.0</td>
      <td>88.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7226</th>
      <td>Ahmedabad</td>
      <td>2014</td>
      <td>41000</td>
      <td>Petrol</td>
      <td>Manual</td>
      <td>First</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>HYUNDAI</td>
      <td>SANTRO</td>
      <td>XING GL</td>
      <td>West</td>
      <td>5</td>
      <td>0.0</td>
      <td>kmpl</td>
      <td>1086.0</td>
      <td>62.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 18 columns</p>
</div>



There are cars of spanning a wide range of manufacture year and other specification.
Since this incorrect value of 0 will affect the analysis let replace 0 with NaN.


```python
data.mileage.replace(to_replace=0, value= np.nan, inplace=True)
```

Let's check to see if mileage 0 values were changed


```python
if any(data.mileage!=0):
    print("There are no 0 values in the mileage column")
```

    There are no 0 values in the mileage column
    

---

# Missing Value Treatment

Let check the missing values in the dataframe


```python
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
      <td>6244</td>
      <td>86.124</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1232</td>
      <td>16.993</td>
    </tr>
    <tr>
      <th>power</th>
      <td>175</td>
      <td>2.414</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>83</td>
      <td>1.145</td>
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
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.028</td>
    </tr>
  </tbody>
</table>
</div>


---

## `new_price` column

In the `new_price` series, 86% of the values are missing. Since more than 60% of the values are missing, Let's drop the entire `new price` series.


```python
data.drop(columns = 'new_price', inplace=True)
```

Let check the missing values in the dataframe


```python
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
      <th>price</th>
      <td>1232</td>
      <td>16.993</td>
    </tr>
    <tr>
      <th>power</th>
      <td>175</td>
      <td>2.414</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>83</td>
      <td>1.145</td>
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
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.028</td>
    </tr>
  </tbody>
</table>
</div>


**There are no `new_price` missing values as the column has been dropped.**

---

## `price` column

We are trying to build a model to predict the used car prices, this indicates that the response variable is `price`.

Since the response variable is `price` we cannot simply impute values for missing `price` series.  
Therefore we shall drop all rows in the `price` column with NaN.


```python
data.dropna(subset=['price'], inplace=True)
```

Let check the missing values in the dataframe


```python
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
      <th>power</th>
      <td>143</td>
      <td>2.376</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>70</td>
      <td>1.163</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>42</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


**There are no missing values for the `price` series**

---

## Data imputation rationale

Products manufactured in large scale production facilities are designed and constructed with strict guidelines to maintain quality and performance.  

This implies that cars of the same make and model will have similiar and uniform design characteristics such as seats, engine, mileage and power.  

Therefore **assuming that all the cars are of stock configuration**, it is possible to impute the missing values of the `engine`, `seats`, `power` and `mileage` by comparing like for like **car names** where the values are available in the dataset.

To begin let's combine the `car_make`and `car_model` and add it to the dataframe


```python
data['car_name'] = data[['car_make', 'car_model']].apply(lambda x: ' '.join(x), axis=1)
```

---

## `power` column

Let's create a `car_power` dataframe by filtering all non-null `power` values and calculate the `mean power` values by the `car_name`.


```python
car_power = data[data.power.notnull()].filter(["car_name", "power"])
```

From the car_power dataframe let's get the `mean power` by `car_name`


```python
car_power = car_power.groupby("car_name").agg('mean')
car_power
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power</th>
    </tr>
    <tr>
      <th>car_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMBASSADOR CLASSIC</th>
      <td>35.500000</td>
    </tr>
    <tr>
      <th>AUDI A3</th>
      <td>143.000000</td>
    </tr>
    <tr>
      <th>AUDI A4</th>
      <td>163.634416</td>
    </tr>
    <tr>
      <th>AUDI A6</th>
      <td>186.290000</td>
    </tr>
    <tr>
      <th>AUDI A7</th>
      <td>241.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>VOLVO S60</th>
      <td>185.857143</td>
    </tr>
    <tr>
      <th>VOLVO S80</th>
      <td>215.000000</td>
    </tr>
    <tr>
      <th>VOLVO V40</th>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>VOLVO XC60</th>
      <td>200.666667</td>
    </tr>
    <tr>
      <th>VOLVO XC90</th>
      <td>200.000000</td>
    </tr>
  </tbody>
</table>
<p>208 rows × 1 columns</p>
</div>



To impute the missing power values with the `mean power` by `car_name` we need to configure the `data` dataframe to be similiar to the `car_power` dataframe.   
We shall set the `data`  index to the `car_name`


```python
data = data.set_index('car_name')
```

Let's used the `combine_first` function to update null elements with value in the same location.


```python
data = data.combine_first(car_power)
```

Reset `data` index


```python
data = data.reset_index()
```

Let check the missing values in the dataframe


```python
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
      <th>mileage</th>
      <td>70</td>
      <td>1.163</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>42</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>power</th>
      <td>11</td>
      <td>0.183</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


Let's investigate further to determine why there are still 11 missing `power` values.


```python
data[data.power.isnull()]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_name</th>
      <th>car_age</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>engine</th>
      <th>fuel_type</th>
      <th>kilometers_driven</th>
      <th>location</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>owner_type</th>
      <th>power</th>
      <th>price</th>
      <th>region</th>
      <th>seats</th>
      <th>transmission</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>660</th>
      <td>FIAT PETRA</td>
      <td>14</td>
      <td>FIAT</td>
      <td>PETRA</td>
      <td>1.2 EL</td>
      <td>1242.0</td>
      <td>Petrol</td>
      <td>120000</td>
      <td>Pune</td>
      <td>15.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>NaN</td>
      <td>0.85</td>
      <td>West</td>
      <td>5.0</td>
      <td>Manual</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>666</th>
      <td>FIAT SIENA</td>
      <td>18</td>
      <td>FIAT</td>
      <td>SIENA</td>
      <td>1.2 ELX</td>
      <td>1242.0</td>
      <td>Petrol</td>
      <td>70000</td>
      <td>Jaipur</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>Third</td>
      <td>NaN</td>
      <td>0.55</td>
      <td>North</td>
      <td>5.0</td>
      <td>Manual</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>2820</th>
      <td>MAHINDRA JEEP</td>
      <td>10</td>
      <td>MAHINDRA</td>
      <td>JEEP</td>
      <td>MM 550 PE</td>
      <td>2498.0</td>
      <td>Diesel</td>
      <td>26000</td>
      <td>Hyderabad</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>NaN</td>
      <td>6.99</td>
      <td>South</td>
      <td>6.0</td>
      <td>Manual</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>2821</th>
      <td>MAHINDRA JEEP</td>
      <td>17</td>
      <td>MAHINDRA</td>
      <td>JEEP</td>
      <td>MM 540 DP</td>
      <td>2112.0</td>
      <td>Diesel</td>
      <td>75000</td>
      <td>Chennai</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>NaN</td>
      <td>1.70</td>
      <td>South</td>
      <td>6.0</td>
      <td>Manual</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>3076</th>
      <td>MARUTI 1000</td>
      <td>21</td>
      <td>MARUTI</td>
      <td>1000</td>
      <td>AC</td>
      <td>970.0</td>
      <td>Petrol</td>
      <td>104000</td>
      <td>Hyderabad</td>
      <td>15.0</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>NaN</td>
      <td>0.85</td>
      <td>South</td>
      <td>5.0</td>
      <td>Manual</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>3562</th>
      <td>MARUTI ESTILO</td>
      <td>11</td>
      <td>MARUTI</td>
      <td>ESTILO</td>
      <td>LXI</td>
      <td>1061.0</td>
      <td>Petrol</td>
      <td>56000</td>
      <td>Chennai</td>
      <td>19.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>NaN</td>
      <td>1.50</td>
      <td>South</td>
      <td>NaN</td>
      <td>Manual</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3563</th>
      <td>MARUTI ESTILO</td>
      <td>9</td>
      <td>MARUTI</td>
      <td>ESTILO</td>
      <td>LXI</td>
      <td>1061.0</td>
      <td>Petrol</td>
      <td>40000</td>
      <td>Kolkata</td>
      <td>19.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>NaN</td>
      <td>2.50</td>
      <td>East</td>
      <td>NaN</td>
      <td>Manual</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3564</th>
      <td>MARUTI ESTILO</td>
      <td>11</td>
      <td>MARUTI</td>
      <td>ESTILO</td>
      <td>LXI</td>
      <td>1061.0</td>
      <td>Petrol</td>
      <td>51000</td>
      <td>Chennai</td>
      <td>19.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>NaN</td>
      <td>1.75</td>
      <td>South</td>
      <td>NaN</td>
      <td>Manual</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>4756</th>
      <td>PORSCHE CAYMAN</td>
      <td>9</td>
      <td>PORSCHE</td>
      <td>CAYMAN</td>
      <td>2009-2012 S</td>
      <td>3436.0</td>
      <td>Petrol</td>
      <td>10000</td>
      <td>Hyderabad</td>
      <td>9.0</td>
      <td>kmpl</td>
      <td>First</td>
      <td>NaN</td>
      <td>40.00</td>
      <td>South</td>
      <td>2.0</td>
      <td>Manual</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4757</th>
      <td>PORSCHE CAYMAN</td>
      <td>8</td>
      <td>PORSCHE</td>
      <td>CAYMAN</td>
      <td>2009-2012 S TIPTRONIC</td>
      <td>3436.0</td>
      <td>Petrol</td>
      <td>8000</td>
      <td>Mumbai</td>
      <td>9.0</td>
      <td>kmpl</td>
      <td>First</td>
      <td>NaN</td>
      <td>43.00</td>
      <td>West</td>
      <td>2.0</td>
      <td>Automatic</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>5084</th>
      <td>SMART FORTWO</td>
      <td>11</td>
      <td>SMART</td>
      <td>FORTWO</td>
      <td>CDI AT</td>
      <td>799.0</td>
      <td>Diesel</td>
      <td>103000</td>
      <td>Pune</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>West</td>
      <td>2.0</td>
      <td>Automatic</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
</div>



From the dataset, the `power` data is manufacturer supplied specification.  
These 11 cases highlight instances where unique cars dont have `power` data.  
For simplicity we shall impute the `mean power` in this analysis.


```python
data.power.fillna(data.power.mean(), inplace=True)
```

Let check the missing values in the dataframe


```python
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
      <th>mileage</th>
      <td>70</td>
      <td>1.163</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>42</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


---

## `mileage` column

Let's create a `car_mileage` dataframe by filtering all non-null `mileage` values and calculate the `mean mileage` values by the `car_name`.


```python
car_mileage = data[data.mileage.notnull()].filter(["car_name", "mileage"])
```

From the car_mileage dataframe let's get the `mean mileage` by `car_name`


```python
car_mileage = car_mileage.groupby("car_name").agg('mean')
car_mileage
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mileage</th>
    </tr>
    <tr>
      <th>car_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMBASSADOR CLASSIC</th>
      <td>12.800000</td>
    </tr>
    <tr>
      <th>AUDI A3</th>
      <td>20.380000</td>
    </tr>
    <tr>
      <th>AUDI A4</th>
      <td>16.400897</td>
    </tr>
    <tr>
      <th>AUDI A6</th>
      <td>16.581837</td>
    </tr>
    <tr>
      <th>AUDI A7</th>
      <td>14.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>VOLVO S60</th>
      <td>16.108571</td>
    </tr>
    <tr>
      <th>VOLVO S80</th>
      <td>14.550000</td>
    </tr>
    <tr>
      <th>VOLVO V40</th>
      <td>20.050000</td>
    </tr>
    <tr>
      <th>VOLVO XC60</th>
      <td>13.516667</td>
    </tr>
    <tr>
      <th>VOLVO XC90</th>
      <td>11.100000</td>
    </tr>
  </tbody>
</table>
<p>209 rows × 1 columns</p>
</div>



To impute the missing mileage values with the `mean mileage` by `car_name` we need to configure the `data` dataframe to be similiar to the `car_mileage` dataframe.   
We shall set the `data`  index to the `car_name`


```python
data = data.set_index('car_name')
```

Let's used the `combine_first` function to update null elements with value in the same location.


```python
data = data.combine_first(car_mileage)
```

Reset `data` index


```python
data = data.reset_index()
```

Let check the missing values in the dataframe


```python
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
      <th>seats</th>
      <td>42</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>9</td>
      <td>0.150</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


Let's investigate further to determine why there are still 9 missing `mileage` values.


```python
data[data.mileage.isnull()]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_name</th>
      <th>car_age</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>engine</th>
      <th>fuel_type</th>
      <th>kilometers_driven</th>
      <th>location</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>owner_type</th>
      <th>power</th>
      <th>price</th>
      <th>region</th>
      <th>seats</th>
      <th>transmission</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>666</th>
      <td>FIAT SIENA</td>
      <td>18</td>
      <td>FIAT</td>
      <td>SIENA</td>
      <td>1.2 ELX</td>
      <td>1242.0</td>
      <td>Petrol</td>
      <td>70000</td>
      <td>Jaipur</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>Third</td>
      <td>112.682838</td>
      <td>0.55</td>
      <td>North</td>
      <td>5.0</td>
      <td>Manual</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>2819</th>
      <td>MAHINDRA E</td>
      <td>3</td>
      <td>MAHINDRA</td>
      <td>E</td>
      <td>VERITO D4</td>
      <td>72.0</td>
      <td>Electric</td>
      <td>50000</td>
      <td>Chennai</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>First</td>
      <td>41.000000</td>
      <td>13.00</td>
      <td>South</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>2820</th>
      <td>MAHINDRA JEEP</td>
      <td>10</td>
      <td>MAHINDRA</td>
      <td>JEEP</td>
      <td>MM 550 PE</td>
      <td>2498.0</td>
      <td>Diesel</td>
      <td>26000</td>
      <td>Hyderabad</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>112.682838</td>
      <td>6.99</td>
      <td>South</td>
      <td>6.0</td>
      <td>Manual</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>2821</th>
      <td>MAHINDRA JEEP</td>
      <td>17</td>
      <td>MAHINDRA</td>
      <td>JEEP</td>
      <td>MM 540 DP</td>
      <td>2112.0</td>
      <td>Diesel</td>
      <td>75000</td>
      <td>Chennai</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>112.682838</td>
      <td>1.70</td>
      <td>South</td>
      <td>6.0</td>
      <td>Manual</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>4306</th>
      <td>MERCEDES-BENZ C-CLASS</td>
      <td>0</td>
      <td>MERCEDES-BENZ</td>
      <td>C-CLASS</td>
      <td>PROGRESSIVE C 220D</td>
      <td>1950.0</td>
      <td>Diesel</td>
      <td>15369</td>
      <td>Coimbatore</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>194.000000</td>
      <td>35.67</td>
      <td>South</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4307</th>
      <td>MERCEDES-BENZ C-CLASS</td>
      <td>1</td>
      <td>MERCEDES-BENZ</td>
      <td>C-CLASS</td>
      <td>PROGRESSIVE C 220D</td>
      <td>1950.0</td>
      <td>Diesel</td>
      <td>8682</td>
      <td>Mumbai</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>194.000000</td>
      <td>39.50</td>
      <td>West</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4308</th>
      <td>MERCEDES-BENZ C-CLASS</td>
      <td>0</td>
      <td>MERCEDES-BENZ</td>
      <td>C-CLASS</td>
      <td>PROGRESSIVE C 220D</td>
      <td>1950.0</td>
      <td>Diesel</td>
      <td>4000</td>
      <td>Ahmedabad</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>First</td>
      <td>194.000000</td>
      <td>35.00</td>
      <td>West</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>5084</th>
      <td>SMART FORTWO</td>
      <td>11</td>
      <td>SMART</td>
      <td>FORTWO</td>
      <td>CDI AT</td>
      <td>799.0</td>
      <td>Diesel</td>
      <td>103000</td>
      <td>Pune</td>
      <td>NaN</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>112.682838</td>
      <td>3.00</td>
      <td>West</td>
      <td>2.0</td>
      <td>Automatic</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>5677</th>
      <td>TOYOTA PRIUS</td>
      <td>8</td>
      <td>TOYOTA</td>
      <td>PRIUS</td>
      <td>2009-2016 Z4</td>
      <td>1798.0</td>
      <td>Electric</td>
      <td>44000</td>
      <td>Mumbai</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>First</td>
      <td>73.000000</td>
      <td>12.75</td>
      <td>West</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>
</div>



From the dataset, the `mileage` data is manufacturer supplied specification.  
These 9 cases highlight instances where unique cars dont have `mileage` data.  
For simplicity we shall impute the `mean mileage` in this analysis.


```python
data.mileage.fillna(data.mileage.mean(), inplace=True)
```

Let check the missing values in the dataframe


```python
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
      <th>seats</th>
      <td>42</td>
      <td>0.698</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


All the missing `mileage` values have been dealt with.

---

## `seats` column

Let's create a `car_seats` dataframe by filtering all non-null `seats` values and calculate the `median seats` values by the `car_name`.


```python
car_seats = data[data.seats.notnull()].filter(["car_name", "seats"])
```

From the car_seats dataframe let's get the `median seats` by `car_name`


```python
car_seats = car_seats.groupby("car_name").agg('median')
car_seats
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seats</th>
    </tr>
    <tr>
      <th>car_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMBASSADOR CLASSIC</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>AUDI A3</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>AUDI A4</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>AUDI A6</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>AUDI A7</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>VOLVO S60</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>VOLVO S80</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>VOLVO V40</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>VOLVO XC60</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>VOLVO XC90</th>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 1 columns</p>
</div>



To impute the missing seats values with the `median seats` by `car_name` we need to configure the `data` dataframe to be similiar to the `car_seats` dataframe.   
We shall set the `data`  index to the `car_name`


```python
data= data.set_index('car_name')
```

Let's used the `combine_first` function to update null elements with value in the same location.


```python
data = data.combine_first(car_seats)
```

Reset `data` index


```python
data = data.reset_index()
```

Let check the missing values in the dataframe


```python
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
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>3</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


Let's investigate further to determine why there are still 3 missing `seats` values.


```python
data[data.seats.isnull()]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_name</th>
      <th>car_age</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>engine</th>
      <th>fuel_type</th>
      <th>kilometers_driven</th>
      <th>location</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>owner_type</th>
      <th>power</th>
      <th>price</th>
      <th>region</th>
      <th>seats</th>
      <th>transmission</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3562</th>
      <td>MARUTI ESTILO</td>
      <td>11</td>
      <td>MARUTI</td>
      <td>ESTILO</td>
      <td>LXI</td>
      <td>1061.0</td>
      <td>Petrol</td>
      <td>56000</td>
      <td>Chennai</td>
      <td>19.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>112.682838</td>
      <td>1.50</td>
      <td>South</td>
      <td>NaN</td>
      <td>Manual</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3563</th>
      <td>MARUTI ESTILO</td>
      <td>9</td>
      <td>MARUTI</td>
      <td>ESTILO</td>
      <td>LXI</td>
      <td>1061.0</td>
      <td>Petrol</td>
      <td>40000</td>
      <td>Kolkata</td>
      <td>19.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>112.682838</td>
      <td>2.50</td>
      <td>East</td>
      <td>NaN</td>
      <td>Manual</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3564</th>
      <td>MARUTI ESTILO</td>
      <td>11</td>
      <td>MARUTI</td>
      <td>ESTILO</td>
      <td>LXI</td>
      <td>1061.0</td>
      <td>Petrol</td>
      <td>51000</td>
      <td>Chennai</td>
      <td>19.5</td>
      <td>kmpl</td>
      <td>Second</td>
      <td>112.682838</td>
      <td>1.75</td>
      <td>South</td>
      <td>NaN</td>
      <td>Manual</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
</div>



From above we can see that there are 3 cases where the `seats` values is missing.  
The MARUTI ESTILO is a unique vehicle in the dataset where there are no instances of `seats` being given.  
Let's assume this is a sedan and has the typical number of `seats` as any other car in the dataset.  
Therefore we shall impute the `median seats` for these 3 cases.


```python
data.seats.fillna(data.seats.median(), inplace=True)
```

Let check the missing values in the dataframe


```python
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
      <th>engine</th>
      <td>36</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


All the missing `seats` values have been dealt with.

---

## `engine` column

Let's create a `car_name` dataframe by filtering all non-null `engine` values and calculate the `mean engine` values by the `car_name`.


```python
car_engine = data[data.engine.notnull()].filter(["car_name", "engine"])
```

From the car_engine dataframe let's get the `mean engine size` by `car_name`


```python
car_engine = car_engine.groupby("car_name").agg('mean')
car_engine
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>engine</th>
    </tr>
    <tr>
      <th>car_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMBASSADOR CLASSIC</th>
      <td>1489.000000</td>
    </tr>
    <tr>
      <th>AUDI A3</th>
      <td>1968.000000</td>
    </tr>
    <tr>
      <th>AUDI A4</th>
      <td>2020.871795</td>
    </tr>
    <tr>
      <th>AUDI A6</th>
      <td>2189.857143</td>
    </tr>
    <tr>
      <th>AUDI A7</th>
      <td>2967.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>VOLVO S60</th>
      <td>2039.285714</td>
    </tr>
    <tr>
      <th>VOLVO S80</th>
      <td>2400.000000</td>
    </tr>
    <tr>
      <th>VOLVO V40</th>
      <td>1980.250000</td>
    </tr>
    <tr>
      <th>VOLVO XC60</th>
      <td>2261.500000</td>
    </tr>
    <tr>
      <th>VOLVO XC90</th>
      <td>2400.000000</td>
    </tr>
  </tbody>
</table>
<p>215 rows × 1 columns</p>
</div>



To impute the missing engine values with the `mean engine size` by `car_name` we need to configure the `data` dataframe to be similiar to the `car_engine` dataframe.   
We shall set the `data`  index to the `car_name` 


```python
data = data.set_index('car_name')
```

Let's used the `combine_first` function to update null elements with value in the same location.


```python
data = data.combine_first(car_engine)
```

Reset `data` index


```python
data = data.reset_index()
```

Let check the missing values in the dataframe


```python
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
      <th>mileage_unit</th>
      <td>2</td>
      <td>0.033</td>
    </tr>
  </tbody>
</table>
</div>


All the missing `engine` values have been dealt with.

---

## `mileage_unit` column

Let's check the rows where mileage_unit is NaN


```python
data[data.mileage_unit.isnull()]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_name</th>
      <th>car_age</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>engine</th>
      <th>fuel_type</th>
      <th>kilometers_driven</th>
      <th>location</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>owner_type</th>
      <th>power</th>
      <th>price</th>
      <th>region</th>
      <th>seats</th>
      <th>transmission</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2819</th>
      <td>MAHINDRA E</td>
      <td>3</td>
      <td>MAHINDRA</td>
      <td>E</td>
      <td>VERITO D4</td>
      <td>72.0</td>
      <td>Electric</td>
      <td>50000</td>
      <td>Chennai</td>
      <td>18.317139</td>
      <td>NaN</td>
      <td>First</td>
      <td>41.0</td>
      <td>13.00</td>
      <td>South</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>5677</th>
      <td>TOYOTA PRIUS</td>
      <td>8</td>
      <td>TOYOTA</td>
      <td>PRIUS</td>
      <td>2009-2016 Z4</td>
      <td>1798.0</td>
      <td>Electric</td>
      <td>44000</td>
      <td>Mumbai</td>
      <td>18.317139</td>
      <td>NaN</td>
      <td>First</td>
      <td>73.0</td>
      <td>12.75</td>
      <td>West</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the two rows are Electric powered cars.  
Let's check the dataframe to see if there are other Electric powered cars.


```python
data[data.fuel_type=="Electric"]
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_name</th>
      <th>car_age</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_spec</th>
      <th>engine</th>
      <th>fuel_type</th>
      <th>kilometers_driven</th>
      <th>location</th>
      <th>mileage</th>
      <th>mileage_unit</th>
      <th>owner_type</th>
      <th>power</th>
      <th>price</th>
      <th>region</th>
      <th>seats</th>
      <th>transmission</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2819</th>
      <td>MAHINDRA E</td>
      <td>3</td>
      <td>MAHINDRA</td>
      <td>E</td>
      <td>VERITO D4</td>
      <td>72.0</td>
      <td>Electric</td>
      <td>50000</td>
      <td>Chennai</td>
      <td>18.317139</td>
      <td>NaN</td>
      <td>First</td>
      <td>41.0</td>
      <td>13.00</td>
      <td>South</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>5677</th>
      <td>TOYOTA PRIUS</td>
      <td>8</td>
      <td>TOYOTA</td>
      <td>PRIUS</td>
      <td>2009-2016 Z4</td>
      <td>1798.0</td>
      <td>Electric</td>
      <td>44000</td>
      <td>Mumbai</td>
      <td>18.317139</td>
      <td>NaN</td>
      <td>First</td>
      <td>73.0</td>
      <td>12.75</td>
      <td>West</td>
      <td>5.0</td>
      <td>Automatic</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>
</div>



Since there are only 2 Electric powered cars in the dataset and they both are missing the `mileage_unit`, then let's drop  the rows where mileage_unit is `NaN`


```python
data = data.dropna(subset=['mileage_unit'])
```

Let check the missing values in the dataframe


```python
missing_val_chk(data)
```

    There are NO missing values in the dataset
    

---

**The 5 Point Summary**

Let's run the 5 point descriptive statistics again


```python
data.describe(include='all').T
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
      <th>car_name</th>
      <td>6016</td>
      <td>213</td>
      <td>MARUTI SWIFT</td>
      <td>353</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>car_age</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.640791</td>
      <td>3.269211</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>car_make</th>
      <td>6016</td>
      <td>30</td>
      <td>MARUTI</td>
      <td>1210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>car_model</th>
      <td>6016</td>
      <td>209</td>
      <td>SWIFT</td>
      <td>353</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>car_spec</th>
      <td>6016</td>
      <td>1703</td>
      <td>VDI</td>
      <td>110</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1621.006295</td>
      <td>600.658687</td>
      <td>624.0</td>
      <td>1198.0</td>
      <td>1493.0</td>
      <td>1984.0</td>
      <td>5998.0</td>
    </tr>
    <tr>
      <th>fuel_type</th>
      <td>6016</td>
      <td>4</td>
      <td>Diesel</td>
      <td>3205</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>kilometers_driven</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>58743.370346</td>
      <td>91291.294238</td>
      <td>171.0</td>
      <td>34000.0</td>
      <td>53000.0</td>
      <td>73000.0</td>
      <td>6500000.0</td>
    </tr>
    <tr>
      <th>location</th>
      <td>6016</td>
      <td>11</td>
      <td>Mumbai</td>
      <td>789</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.317139</td>
      <td>4.165363</td>
      <td>6.4</td>
      <td>15.3</td>
      <td>18.16</td>
      <td>21.1</td>
      <td>33.54</td>
    </tr>
    <tr>
      <th>mileage_unit</th>
      <td>6016</td>
      <td>2</td>
      <td>kmpl</td>
      <td>5950</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>owner_type</th>
      <td>6016</td>
      <td>4</td>
      <td>First</td>
      <td>4926</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>power</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>112.70135</td>
      <td>53.70084</td>
      <td>34.2</td>
      <td>75.0</td>
      <td>93.7</td>
      <td>138.1</td>
      <td>560.0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.479624</td>
      <td>11.190091</td>
      <td>0.44</td>
      <td>3.5</td>
      <td>5.64</td>
      <td>9.95</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>region</th>
      <td>6016</td>
      <td>4</td>
      <td>South</td>
      <td>2879</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.278092</td>
      <td>0.803955</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>transmission</th>
      <td>6016</td>
      <td>2</td>
      <td>Manual</td>
      <td>4298</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>year</th>
      <td>6016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013.359209</td>
      <td>3.269211</td>
      <td>1998.0</td>
      <td>2011.0</td>
      <td>2014.0</td>
      <td>2016.0</td>
      <td>2019.0</td>
    </tr>
  </tbody>
</table>
</div>



**Observation:**
* **car_name** - Most popular car name is MARUTI SWIFT
* **fuel_type** - Most popular fuel_type is Diesel
* **location** - Most popular location is Mumbai
* **mileage_unit** - Most popular mileage unit is kmpl
* **owner_type** - Most popular owner_type is First.
* **trasmission** - Most popular trasmission is Manual
* **year**  - The dataset has cars manufactured from 1996 to 2019
* **car_age** - The mean and median car age are close.

---

Export cleaned dataframe


```python
data.to_csv("cleaned_used_cars_data.csv")
```

---
