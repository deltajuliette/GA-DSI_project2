# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Regression Challenge

### Background

In this project, we will use the well known Ames housing data (from 2006-10) to create a regression model that can predict the price of houses in Ames, Iowa.

### Problem statement

**What we plan to do**

There are many variables that determine how much a home can fetch.

Using the Ames (IA) dataset (train, test), we want to find out which variables matter for home saleprices and produce accurate saleprice predictions.

**What type of models will we be exploring?**

We will be focusing our project on supervised machine learning models. Linear models and regularization are areas of particular interest for us.

**How will success be evaluated?**

We will focus on optimising  𝑅2 , as well as RMSE across different models. Our aim is to create a model that significantly outperforms the baseline heuristic which we've identified.

**Is this the scope of the project appropriate? Who are our important stakeholders and why is this important to investigate?**

This model will provide the Outside View, helping to reduce information asymmetry between potential home-buyers, home-sellers and real estate agents.

An Outside View* involves ignoring these details and using an estimate based on a class of roughly similar previous cases. Your Inside View involves making predictions based on your understanding of the details of the process - not that it’s wrong, it’s just that it’s often biased!

*Excerpt from Daniel Kahneman - Thinking Fast and Slow: The outside view asks if there are similar situations that can provide a statistical basis for making a decision. Rather than seeing a problem as unique, the outside view wants to know if others have faced comparable problems and, if so, what happened*

---

### Data dictionary

This is a data dictionary of our cleaned dataset, which will provide a quick overview of features/variables/columns, alongside data types and descriptions. We will explain our methodology in arriving at this dataset below.

**train_trimmed**: Cleaned training dataset
**test_trimmed**: Similar to cleaned training dataset (excluding saleprice)

|Feature|Type|Description|
|:-|:-:|:-|
|saleprice|int64|The property's sale price in dollars. This is our target variable|
|overall_qual|int64|Overall material and finish quality|
|gr_liv_area|int64|Above grade (ground) living area square feet|
|garage_area|float64|Size of garage in square feet|
|garage_cars|float64|Size of garage in car capacity|
|total_bsmt_sf|float64|Total square feet of basement area|
|1st_flr_sf|int64|First Floor square feet|
|full_bath|int64|Full bathrooms above grade|
|mas_vnr_area|float64|Masonry veneer area in square feet|
|totrms_abvgrd|int64|Total rooms above grade (does not include bathrooms)|
|fireplaces|int64|Number of fireplaces|
|bsmtfin_sf_1|float64|Type 1 finished square feet|
|lot_frontage|float64|Linear feet of street connected to property|
|open_porch_sf|int64|Open porch area in square feet|
|wood_deck_sf|int64|Wood deck area in square feet|
|lot_area|int64|Lot size in square feet|
|age_remodel|int64|Time since remodel (Calculated)|
|age_home|int64|Age of home (Calculated)|
|street|object|Type of road access to property|
|land_contour|object|Flatness of the property|
|neighborhood|object|Physical locations within Ames city limits|
|condition_1|object|Proximity to main road or railroad|
|condition_2|object|Proximity to main road or railroad (if a second is present)|
|bldg_type|object|Type of dwelling|
|house_style|object|Style of dwelling|
|roof_matl|object|Roof material|
|exterior_1st|object|Exterior covering on house|
|exterior_2nd|object|Exterior covering on house (if more than one material)|
|mas_vnr_type|object|Masonry veneer type|
|exter_qual|object|Exterior material quality|
|exter_cond|object|Present condition of the material on the exterior|
|foundation|object|Type of foundation|
|bsmt_qual|object|Height of the basement|
|bsmt_cond|object|General condition of the basement|
|bsmt_exposure|object|Walkout or garden level basement walls|
|bsmtfin_type_1|object|Quality of basement finished area|
|bsmtfin_type_2|object|Quality of second finished area (if present)|
|heating_qc|object|Heating quality and condition|
|central_air|object|Central air conditioning|
|kitchen_qual|object|Kitchen quality|
|functional|object|Home functionality rating|
|garage_type|object|Garage location|
|garage_finish|object|Interior finish of the garage|
|garage_qual|object|Garage quality|
|garage_cond|object|Garage condition|
|sales_type|object|Type of sale|

---

### Methodology

**Data Cleaning and EDA**

Our train dataset has a total of 81 columns. After creating additional features like ```age_remodel``` and ```age_home```, and dropping features like ```year_built```, ```yr_sold```, ```year_remod/add```, and ```garage_yr_blt```. We have 42 qualitative columns vs. 37 quantitative columns

*Addressing quantiative variables**

We used a correlation matrix with a pre-determined threshold to narrow down the list of quantitative features to select for our model. The features selected here include: ```overall_qual```, ```gr_liv_area```, ```garage_area```, ```garage_cars```, ```total_bsmt_sf```, ```1st_flr_sf```, ```full_bath```, ```mas_vnr_area```, ```totrms_abvgrd```, ```fireplaces```, ```bsmtfin_sf_1```, ```lot_frontage```, ```open_porch_sf```, ```wood_deck_sf```, ```lot_area```, ```age_remodel```, ```age_home```

We then identify features with many missing values. We dropped columns with more than 20% of values missing. 

For remaining features, several quantitative features like ```garage_area```, ```garage_cars```, ```mas_vnr_area```, ```total_bsmt_sf```, ```bsmtfin_sf_1```, ```lot_frontage``` are missing values. These do not appear to be missing at random. Same goes for qualitative features - which will address later

For missing values in ```garage_cars```, ```garage_area```, ```total_bsmt_sf```, ```bsmtfin_sf_1```, ```bsmt_unf_sf```, we can impute them with 0. As there are significant price differences across houses in different neighbourhoods, we will impute missing values in ```lot_frontage``` with the median lot frontage for each neighbourhood.

After this step, we will visually inspect the distributions of these features relatie to ```saleprice``` and drop any outliers.

*Addressing qualitative variables*

We can technically see if categorical features have a significant impact on saleprice by visually inspecting individual boxplots. We've decided to drop the following columns: ```ms_zoning```, ```lot_shape```, ```utilities```, ```roof_style```, ```heating```, ```electrical```, ```lot_config```, ```land_slope```, ```paved_drive```.

Imputing missing values in qualitative features takes place thereafter

Similar steps will be applied for our test dataset.

**Processing our data**

To increase efficiency, we will combine both train and test datasets for pre-processing - Creating a custom label for both datasets. This way, we'll be able to capture both train and test datasets' unique qualitative features when one-hot encoding them. 

After dummifying qualitative features, we will split up our dataset into respective train and test datasets again (using the custom label)

Following that, we've also decided to log-transform several quantitative features in both train and test such as ```gr_liv_area```, ```garage_area```, ```total_bsmt_sf```, ```mas_vnr_area```, ```1st_flr_sf```, ```wood_deck_sf```, ```bsmtfin_sf_1```, ```open_porch_sf```, ```lot_frontage```, ```lot_area```, as well as our target ```saleprice``` variable.

**Building models**

We define our X and y variables accordingly before carrying out a train, test split for our train dataset. We always want to have a holdout set to test our models, AFTER training. This holdout set is used to simulate new future data outside of this dataset, to build a future-proof model that generalizes and performs well.

We will try out several models such as (1) Linear regression, (2) Ridge regression, (3) Lasso regression and see whether they outperform our baseline model (Based on a simple average). We will use cross validation (n=10) to evaluate these models.

Before we carry out ridge / lasso regressions, we will first need to standardize our features (X). Since we are adding a loss function (with penalty) in Ridge and Lasso regressions, scaling is required so that regularization penalises each variable equally fairly.

Standardization does not change the skew of the distribution. What it does is to transform the values so that the overall distribution has  μ  = 0 and  σ2  = 1. The shape of the actual distribution remains unchanged.

Many elements used in the objective function of a learning algorithm assume that all features are centered around zero and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

When a feature does not follow a linear distribution, it would be unwise to use the mean and the standard deviation to scale it. Log-transformation changes the skew of the distribution, and is useful when you deal with right-skewed distributions (fat right tails).

We will also optimise alpha hyperparameters for both ridge and lasso regression models.


---

### Key findings

**Model selection**

|Model|R2 scores|RMSE|MAE|Selection|
|:-|:-:|:-|:-|:-|
|Baseline (Mean)|-0.0105|73309.49|55817.79||
|Linear Regression|0.8734|20718.47|14091.24||
|Ridge Regression|0.9003|20024.02|13836.78||
|Lasso Regression|0.9084|19566.67|13697.66|Y|

We've decided to select the  Lasso regression model from the list of models presented above. Why?

* Both Lasso and ridge regression models have $R^2$ scores exceeding 90%; That means over 90% of the variability in home saleprices (*y*) can be explained by the features we've selected in both models
* However, our Lasso regression model is a tad better given its lower RMSE and MAE scores
    * Root mean squared errors (**RMSE**) represents the approximate average distance from predicted values; In this case, actual saleprices are on average ~19.5K away from our predicted values
    * Mean absolute errors (**MAE**) represents the mean distance from the predicted value; Tying this back to our Lasso regression model, actual saleprices have a mean distance of 13.7k from our predicted values
    
**Our thoughts**: Our lasso regression model tops the rankings for R2 sores, RMSEs and MAEs. So let's move forward with that model in mind.

We will then extract the largest (+/-) coefficients from our Lasso regression model to determine which features matter for a house's saleprice.

With that concluded, let's now fit the Lasso regression model to the entire training dataset and make predictions for our test dataset below. These predictions will be stored on a separate .csv file and submitted to Kaggle.

---

### Conclusions and recommendations
Recall our problem statement: 'There are are many variables that determine how much a home can fetch, and there is traditionally alot of information asymmetry between buyers, sellers and realtors. 

Our lasso regression model shrinks the coefficients of features that are not relevant or not a good predictor of saleprices, to zero. 

The key features to bear in mind when trying to gauge the potential selling price of a home are the following:
- ```gr_liv_area``` - The larger the above grade (ground) living area (sq ft), the higher the saleprice of a home
- ```overall_qual``` - The higher the overall material and finish quality, needless to say, the higher the saleprice of a home (1-10 with 1 being 'Very poor', and 10 being 'Very excellent')
- ```lot_area``` / ```bsmt_fin_sf_1``` - The larger the lot area / basement size, it is likely the house is larger and hence more expensive-  
- ```age_remodel``` / ```age_home``` - The older the home / The longer the period since remodelling, the lower the saleprice of a home

Of course, every model has its limitations, ours included.

* Geographical constraints; The dataset is restricted to Ames, a city in Iowa with a relatively small population
* The dataset included saleprices from 2006-2010; Structural and cyclical trends may have changed since then
* The model does a fairly decent job for predicting salepries of home within the range of 100k to 300k (See residual plots)
* Other featurs such as the proximity to amenities / educational institutions could be helpful
* Level of crime rates for different neighbourhoods may also add some value

---

