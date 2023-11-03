<p align="center">
  <img src="https://forthebadge.com/images/badges/made-with-python.svg" />&nbsp;&nbsp;&nbsp;
  <img src="https://forthebadge.com/images/badges/made-with-markdown.svg" />&nbsp;&nbsp;&nbsp;
  <img src="https://forthebadge.com/images/badges/powered-by-oxygen.svg" />&nbsp;&nbsp;
</p>




<h1 align="center">🚗 Used Car Price Prediction 💲🔮</h1>

<p align="center">The primary objective of this project is to develop a pricing model that can effectively predict the price of used cars and can help the business devise profitable strategies using differential pricing.</p>

---

## 📝 Table of Contents

- [🤓 Description](#description)
- [💻 Dataset Overview](#dataset-overview)
- [🛠️ Feature Engineering](#feature-engineering)
- [📊 Exploratory Data Analysis](#exploratory-data-analysis)
- [🏗️ Model Building](#model-building)
- [✨ Recommendations](#recommendations)
- [📗 Notebooks](#notebooks)
- [📧 Contact Information](#contact-information)

## 🤓 Description <a name = "description"></a>

There is a huge demand for used cars in the Indian Market today. As sales of new cars have slowed down in the recent past, the pre-owned car market has continued to grow over the past years and is larger than the new car market now. Cars4U is a budding tech start-up that aims to find footholes in this market.  

In 2018-19, while new car sales were recorded at 3.6 million units, around 4 million second-hand cars were bought and sold. There is a slowdown in new car sales and that could mean that the demand is shifting towards the pre-owned market. In fact, some car sellers replace their old cars with pre-owned cars instead of buying new ones. Unlike new cars, where price and supply are fairly deterministic and managed by OEMs (Original Equipment Manufacturer / except for dealership level discounts which come into play only in the last stage of the customer journey), used cars are very different beasts with huge uncertainty in both pricing and supply. Keeping this in mind, the pricing scheme of these used cars becomes important in order to grow in the market.   

As a data analyst at Cars4U, you have to come up with a pricing model that can effectively predict the price of used cars and can help the business in devising profitable strategies using differential pricing. For example, if the business knows the market price, it will never sell anything below it. 

`Objectives`

1. **Explore and visualize the dataset.**
2. **Build a linear regression model to predict the prices of used cars.**
3. **Generate a set of insights and recommendations that will help the business.**


---

## 💻 Dataset Overview <a name = "database-overview"></a>

The dataset source file can found through the following link:
### Click to view 👇:

[![Data_link](https://github.com/seandhan/Scale-Model-Cars-Database-Analysis/blob/main/images/Data-LINK-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/data/used_cars_data.csv)

The used cars database contains 14 variables. The data dictionary below explains each variable:

<details>
<summary>Data Dictionary</summary>
<br>

1. `S.No.` : Serial Number
2. `Name` : Name of the car which includes Brand name and Model name
3. `Location` : The location in which the car is being sold or is available for purchase Cities
4. `Year` : Manufacturing year of the car
5. `Kilometers_driven` : The total kilometers driven in the car by the previous owner(s) in KM.
6. `Fuel_Type` : The type of fuel used by the car. (Petrol, Diesel, Electric, CNG, LPG)
7. `Transmission` : The type of transmission used by the car. (Automatic / Manual)
8. `Owner` : Type of ownership
9. `Mileage` : The standard mileage offered by the car company in kmpl or km/kg
10. `Engine` : The displacement volume of the engine in CC.
11. `Power` : The maximum power of the engine in bhp.
12. `Seats` : The number of seats in the car.
13. `New_Price` : The price of a new car of the same model in INR Lakhs.(1 Lakh = 100, 000)
14. `Price` : The price of the used car in INR Lakhs (1 Lakh = 100, 000)

</details>


### Click to view 👇:

[![Data Exploration](https://github.com/seandhan/image_database/blob/main/Solution-Dataset%20Exploration-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Dataset%20Exploration/ReadMe.md)

There was a significant amount of data pre-processing required prior data visualization. These steps can be seen in the following section.

----

## 🛠️ Feature Engineering <a name = "feature-engineering"></a>

The step by step data cleaning and wrangling can be observed in this section

### Click to view 👇:

[![Feature Engineering](https://github.com/seandhan/image_database/blob/main/Solution-Feature%20Engineering-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Feature%20Engineering/ReadME.MD)


----

## 📊 Exploratory Data Analysis <a name = "exploratory-data-analysis"></a>

The Univariate and Bivariate analysis can be seen here.

### Click to view 👇:

[![Exploratory Data Analysis](https://github.com/seandhan/image_database/blob/main/Solution-Exploratory%20Data%20Analysis-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Exploratory%20Data%20Analysis/ReadME.MD)


----

## 🏗️ Model Building <a name = "model-building"></a>

The data model preparation and linear regression steps can be seen here.

### Click to view 👇:

[![Model Building](https://github.com/seandhan/image_database/blob/main/Solution-Model%20Building-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Model%20Building/README.MD)


----


## ✨ Recommendations <a name = "recommendations"></a>

1. **Engine Size:** Consider offering a range of engine sizes to cater to different customer preferences. Smaller engines are typically more fuel-efficient, so emphasize their benefits for cost-conscious buyers. For those looking for more power, highlight the advantages of larger engines in terms of performance.
<br>

2. **Car Category:** Focus on popular car categories in your region. If compact cars are in demand, ensure you have a diverse selection of models in that category. Promote the benefits of each category, such as fuel efficiency for compact cars and spaciousness for SUVs.
<br>

3. **Region:** Tailor your inventory and marketing to match regional preferences. For example, if SUVs are popular in suburban areas, stock a variety of SUV models and emphasize their suitability for family and outdoor activities.
<br>

4. **Fuel Type:** Offer cars with different fuel types, including gasoline, diesel, and hybrid options. Highlight the cost savings and environmental benefits of fuel-efficient and hybrid vehicles, especially in regions where eco-friendliness is a priority.
<br>

5. **Mileage:** Clearly communicate the mileage and maintenance history of each vehicle. Lower mileage vehicles can be priced higher, so ensure prospective buyers have access to this information. Offer special promotions or warranties for low-mileage cars.



----

## 📗 Notebooks <a name = "notebooks"></a>

The Notebook for the "Data Exploration" can be accessed below:

### Click to view 👇:

[![DataExp Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Dataset%20Exploration-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Data%20exploration.ipynb)

The Notebook for the "Feature Engineering" can be accessed below:

### Click to view 👇:

[![Feature Engineering Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Feature%20engineering-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Feature_engineering.ipynb)

The Notebook for the "Exploratory Data Analysis" can be accessed below:

### Click to view 👇:

[![EDA Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Exploratory%20Data%20analysis-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Exploratory%20Data%20Analysis.ipynb)

The Notebook for the "Model Building" can be accessed below:

### Click to view 👇:

[![Model Building Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Model%20Building-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Model_Building.ipynb)

----



## 📧 Contact Information <a name = "contact-information"></a>

- Email: [sean_dhanasar@msn.com](mailto:sean_dhanasar@msn.com)
- LinkedIn: [Sean Dhanasar](https://www.linkedin.com/in/sdhanasar)


