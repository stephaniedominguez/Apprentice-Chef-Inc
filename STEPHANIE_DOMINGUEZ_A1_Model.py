# timeit

# Student Name : STEPHANIE DOMINGUEZ
# Cohort       : 1

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
# importing libraries
import pandas as pd # data science essentials
import matplotlib.pyplot  as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # linear regression (statsmodels)
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # linear regression (scikit-learn)
import sklearn.linear_model #Linear regression to see models and testings
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.preprocessing import StandardScaler # standard scaler
#GradientBoostingRegressor https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
from sklearn.ensemble import GradientBoostingRegressor # Model who gave the best result+ model used in this template



################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel(Apprentice Chef Dataset.xlsx)

# setting pandas print options
pd.set_option(display.max_rows+ 500)
pd.set_option(display.max_columns+ 500)
pd.set_option(display.width+ 1000)

# specifying file name
file =  Apprentice_Chef_Dataset.xlsx

original_df = pd.read_excel(file)

apprentice_chef = original_df


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization+ do this here as well

# setting outlier thresholds
#thresholds based on graphs an quantile

TOTAL_MEALS_ORDERED_lo         = 11
TOTAL_MEALS_ORDERED_hi         = 250

UNIQUE_MEALS_PURCH_lo          = 1
UNIQUE_MEALS_PURCH_hi          = 9.5

CONTACTS_W_CUSTOMER_SERVICE_lo = 2
CONTACTS_W_CUSTOMER_SERVICE_hi = 12.5

PRODUCT_CATEGORIES_VIEWED_lo   = 1
PRODUCT_CATEGORIES_VIEWED_hi   = 10

AVG_TIME_PER_SITE_VISIT_hi     = 250

CANCELLATIONS_BEFORE_NOON_hi   = 8

EARLY_DELIVERIES_hi            = 8

LATE_DELIVERIES_hi             = 9

AVG_PREP_VID_TIME_lo           = 0
AVG_PREP_VID_TIME_hi           = 300

TOTAL_PHOTOS_VIEWED_hi         = 600

#higgest revenue
REVENUE_hi                     = 8793.75 


##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers


# TOTAL_MEALS_ORDERED
apprentice_chef[out_TOTAL_MEALS_ORDERED] = 0
condition_hi = apprentice_chef.loc[0:+out_TOTAL_MEALS_ORDERED][apprentice_chef[TOTAL_MEALS_ORDERED] > TOTAL_MEALS_ORDERED_hi]
condition_lo = apprentice_chef.loc[0:+out_TOTAL_MEALS_ORDERED][apprentice_chef[TOTAL_MEALS_ORDERED] < TOTAL_MEALS_ORDERED_lo]

apprentice_chef[out_TOTAL_MEALS_ORDERED].replace(to_replace = condition_hi+
                                    value      = 1 +
                                    inplace    = True)

apprentice_chef[out_TOTAL_MEALS_ORDERED].replace(to_replace = condition_lo+
                                    value      = 1 +
                                    inplace    = True)

# UNIQUE_MEALS_PURCH
apprentice_chef[out_UNIQUE_MEALS_PURCH] = 0
condition_hi = apprentice_chef.loc[0:+out_UNIQUE_MEALS_PURCH][apprentice_chef[UNIQUE_MEALS_PURCH] > UNIQUE_MEALS_PURCH_hi]
condition_lo = apprentice_chef.loc[0:+out_UNIQUE_MEALS_PURCH][apprentice_chef[UNIQUE_MEALS_PURCH] < UNIQUE_MEALS_PURCH_lo]

apprentice_chef[out_UNIQUE_MEALS_PURCH].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

apprentice_chef[out_UNIQUE_MEALS_PURCH].replace(to_replace = condition_lo+
                                    value      = 1+
                                    inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
apprentice_chef[out_CONTACTS_W_CUSTOMER_SERVICE] = 0
condition_hi = apprentice_chef.loc[0:+out_CONTACTS_W_CUSTOMER_SERVICE][apprentice_chef[CONTACTS_W_CUSTOMER_SERVICE] > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = apprentice_chef.loc[0:+out_CONTACTS_W_CUSTOMER_SERVICE][apprentice_chef[CONTACTS_W_CUSTOMER_SERVICE] < CONTACTS_W_CUSTOMER_SERVICE_lo]

apprentice_chef[out_CONTACTS_W_CUSTOMER_SERVICE].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

apprentice_chef[out_CONTACTS_W_CUSTOMER_SERVICE].replace(to_replace = condition_lo+
                                    value      = 1+
                                    inplace    = True)

# PRODUCT_CATEGORIES_VIEWED
apprentice_chef[out_PRODUCT_CATEGORIES_VIEWED] = 0
condition_hi = apprentice_chef.loc[0:+out_PRODUCT_CATEGORIES_VIEWED][apprentice_chef[PRODUCT_CATEGORIES_VIEWED] > PRODUCT_CATEGORIES_VIEWED_hi]
condition_lo = apprentice_chef.loc[0:+out_PRODUCT_CATEGORIES_VIEWED][apprentice_chef[PRODUCT_CATEGORIES_VIEWED] < PRODUCT_CATEGORIES_VIEWED_lo]

apprentice_chef[out_PRODUCT_CATEGORIES_VIEWED].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

apprentice_chef[out_PRODUCT_CATEGORIES_VIEWED].replace(to_replace = condition_lo+
                                    value      = 1+
                                    inplace    = True)

# AVG_TIME_PER_SITE_VISIT
apprentice_chef[out_AVG_TIME_PER_SITE_VISIT] = 0
condition_hi = apprentice_chef.loc[0:+out_AVG_TIME_PER_SITE_VISIT][apprentice_chef[AVG_TIME_PER_SITE_VISIT] > AVG_TIME_PER_SITE_VISIT_hi]

apprentice_chef[out_AVG_TIME_PER_SITE_VISIT].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

# CANCELLATIONS_BEFORE_NOON
apprentice_chef[out_CANCELLATIONS_BEFORE_NOON] = 0
condition_hi = apprentice_chef.loc[0:+out_CANCELLATIONS_BEFORE_NOON][apprentice_chef[CANCELLATIONS_BEFORE_NOON] > CANCELLATIONS_BEFORE_NOON_hi]

apprentice_chef[out_CANCELLATIONS_BEFORE_NOON].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

# EARLY_DELIVERIES
apprentice_chef[out_EARLY_DELIVERIES] = 0
condition_hi = apprentice_chef.loc[0:+out_EARLY_DELIVERIES][apprentice_chef[EARLY_DELIVERIES] > EARLY_DELIVERIES_hi]

apprentice_chef[out_EARLY_DELIVERIES].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

# LATE_DELIVERIES
apprentice_chef[out_LATE_DELIVERIES] = 0
condition_hi = apprentice_chef.loc[0:+out_LATE_DELIVERIES][apprentice_chef[LATE_DELIVERIES] > LATE_DELIVERIES_hi]

apprentice_chef[out_LATE_DELIVERIES].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

# AVG_PREP_VID_TIME
apprentice_chef[out_AVG_PREP_VID_TIME] = 0
condition_hi = apprentice_chef.loc[0:+out_AVG_PREP_VID_TIME][apprentice_chef[AVG_PREP_VID_TIME] > AVG_PREP_VID_TIME_hi]
condition_lo = apprentice_chef.loc[0:+out_AVG_PREP_VID_TIME][apprentice_chef[AVG_PREP_VID_TIME] < AVG_PREP_VID_TIME_lo]

apprentice_chef[out_AVG_PREP_VID_TIME].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

apprentice_chef[out_AVG_PREP_VID_TIME].replace(to_replace = condition_lo+
                                    value      = 1+
                                    inplace    = True)

# TOTAL_PHOTOS_VIEWED_hi
apprentice_chef[out_TOTAL_PHOTOS_VIEWED] = 0
condition_hi = apprentice_chef.loc[0:+out_TOTAL_PHOTOS_VIEWED][apprentice_chef[TOTAL_PHOTOS_VIEWED] > TOTAL_PHOTOS_VIEWED_hi]

apprentice_chef[out_TOTAL_PHOTOS_VIEWED].replace(to_replace = condition_hi+
                                    value      = 1+
                                    inplace    = True)

# setting trend-based thresholds
#thresholds based on scatter plot analisys

AVG_TIME_PER_SITE_VISIT_changes_hi      = 400 # trend changes at this point
CONTACTS_W_CUSTOMER_SERVICE_changes_hi  = 11  # data scatters above this point


TOTAL_MEALS_ORDERED_changes_at          = 0   # data inflated only in this point
PRODUCT_CATEGORIES_VIEWED_changes_at    = 5   # different at 5 higest values
CANCELLATIONS_AFTER_NOON_changes_at     = 3   # Trend changes at this point
LATE_DELIVERIES_changes_at              = 11  # Trend changes at this points
AVG_PREP_VID_TIME_changes_at            = 350 # Trend changes at this points
LARGEST_ORDER_SIZE_changes_at           = 0   # Differnet at 0 revenue of 0


AVG_CLICKS_PER_VISIT_changes_lo         = 8   # trend changes before this point

##############################################################################
## Feature Engineering (trend changes)                                      ##
##############################################################################

# developing features (columns) for outliers

########################################
## change above threshold             ##
########################################

# greater than sign

# AVG_TIME_PER_SITE_VISIT
apprentice_chef[change_AVG_TIME_PER_SITE_VISIT] = 0
condition = apprentice_chef.loc[0:+change_AVG_TIME_PER_SITE_VISIT][apprentice_chef[AVG_TIME_PER_SITE_VISIT] > AVG_TIME_PER_SITE_VISIT_changes_hi]

apprentice_chef[change_AVG_TIME_PER_SITE_VISIT].replace(to_replace = condition+
                                   value      = 1+
                                   inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
apprentice_chef[change_CONTACTS_W_CUSTOMER_SERVICE] = 0
condition = apprentice_chef.loc[0:+change_CONTACTS_W_CUSTOMER_SERVICE][apprentice_chef[CONTACTS_W_CUSTOMER_SERVICE] > CONTACTS_W_CUSTOMER_SERVICE_changes_hi]

apprentice_chef[change_CONTACTS_W_CUSTOMER_SERVICE].replace(to_replace = condition+
                                   value      = 1+
                                   inplace    = True)


########################################
## change at threshold                ##
########################################

# double-equals sign

# TOTAL_MEALS_ORDERED
apprentice_chef[change_TOTAL_MEALS_ORDERED] = 0
condition = apprentice_chef.loc[0:+change_TOTAL_MEALS_ORDERED][apprentice_chef[TOTAL_MEALS_ORDERED] == TOTAL_MEALS_ORDERED_changes_at]

apprentice_chef[change_TOTAL_MEALS_ORDERED].replace(to_replace = condition+
                                       value      = 1+
                                       inplace    = True)

# PRODUCT_CATEGORIES_VIEWED
apprentice_chef[change_PRODUCT_CATEGORIES_VIEWED] = 0
condition = apprentice_chef.loc[0:+change_PRODUCT_CATEGORIES_VIEWED][apprentice_chef[PRODUCT_CATEGORIES_VIEWED] == PRODUCT_CATEGORIES_VIEWED_changes_at]

apprentice_chef[change_PRODUCT_CATEGORIES_VIEWED].replace(to_replace = condition+
                                       value      = 1+
                                       inplace    = True)

# CANCELLATIONS_AFTER_NOON
apprentice_chef[change_CANCELLATIONS_AFTER_NOON] = 0
condition = apprentice_chef.loc[0:+change_CANCELLATIONS_AFTER_NOON][apprentice_chef[CANCELLATIONS_AFTER_NOON] == CANCELLATIONS_AFTER_NOON_changes_at]

apprentice_chef[change_CANCELLATIONS_AFTER_NOON].replace(to_replace = condition+
                                       value      = 1+
                                       inplace    = True)

# AVG_PREP_VID_TIME
apprentice_chef[change_AVG_PREP_VID_TIME] = 0
condition = apprentice_chef.loc[0:+change_AVG_PREP_VID_TIME][apprentice_chef[AVG_PREP_VID_TIME] == AVG_PREP_VID_TIME_changes_at]

apprentice_chef[change_AVG_PREP_VID_TIME].replace(to_replace = condition+
                                       value      = 1+
                                       inplace    = True)

# LARGEST_ORDER_SIZE
apprentice_chef[change_LARGEST_ORDER_SIZE] = 0
condition = apprentice_chef.loc[0:+change_LARGEST_ORDER_SIZE][apprentice_chef[LARGEST_ORDER_SIZE] == LARGEST_ORDER_SIZE_changes_at]

apprentice_chef[change_LARGEST_ORDER_SIZE].replace(to_replace = condition+
                                       value      = 1+
                                       inplace    = True)

########################################
## change above threshold             ##
########################################

# less than sign

# AVG_CLICKS_PER_VISIT
apprentice_chef[change_AVG_CLICKS_PER_VISIT] = 0
condition = apprentice_chef.loc[0:+change_AVG_CLICKS_PER_VISIT][apprentice_chef[AVG_CLICKS_PER_VISIT] < AVG_CLICKS_PER_VISIT_changes_lo]

apprentice_chef[change_AVG_CLICKS_PER_VISIT].replace(to_replace = condition+
                                   value      = 1+
                                   inplace    = True)

# preparing explanatory variable data
apprentice_chef_data   = apprentice_chef.drop([REVENUE+
                               NAME+
                                EMAIL+
                                FIRST_NAME+
                                FAMILY_NAME              
                               ]+
                                axis = 1)


# preparing response variable data
apprentice_chef_target = apprentice_chef.loc[:+ REVENUE]

# preparing response variable data
apprentice_chef_target = apprentice_chef.loc[:+ REVENUE]

x_variables = [
               TOTAL_MEALS_ORDERED+ 
               UNIQUE_MEALS_PURCH+ 
               CONTACTS_W_CUSTOMER_SERVICE+ 
               AVG_TIME_PER_SITE_VISIT+
               TASTES_AND_PREFERENCES+
               MOBILE_LOGINS+ 
               AVG_PREP_VID_TIME+ 
               LARGEST_ORDER_SIZE+ 
               MASTER_CLASSES_ATTENDED+ 
               MEDIAN_MEAL_RATING+ 
               AVG_CLICKS_PER_VISIT+
               out_TOTAL_MEALS_ORDERED+
               out_UNIQUE_MEALS_PURCH+
               out_CANCELLATIONS_BEFORE_NOON+
                out_AVG_PREP_VID_TIME+
               out_TOTAL_PHOTOS_VIEWED+
               change_AVG_TIME_PER_SITE_VISIT+ 
               ]

################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

apprentice_chef_data = apprentice_chef[x_variables]

#INSTANTIATING a model object: StandardScaler()
apprentice_chef_model = StandardScaler()


# FITTING the training data
apprentice_chef_model.fit(apprentice_chef_data)


# TRANSFORMING  model
apprentice_chef_X_scaled = apprentice_chef_model.transform(apprentice_chef_data)


# converting to DataFrame
apprentice_chef_model_scaled_df = pd.DataFrame(apprentice_chef_X_scaled)

#training the data set. saving results in X_train+ X_test+ y_train+ y_test
#random state: in assigment

X_train+ X_test+ y_train+ y_test = train_test_split(apprentice_chef_model_scaled_df,
                                                    apprentice_chef_target,
                                                    test_size = 0.25,
                                                    random_state = 222)

################################################################################
# Final Model (instantiate+ fit+ and predict) GradientBoostingRegressor
#GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. 
# In each stage a regression tree is fit on the negative gradient of the given loss function.
################################################################################

# use this space to instantiate+ fit+ and predict on your final model

# Instantiating a GradientBoostingRegressor as apprentice_chef_model
apprentice_chef_model = GradientBoostingRegressor(learning_rate = 0.1+
                                n_estimators = 100+
                                random_state = 222+
                                max_depth = 3+
                                max_features = auto
                                 )

# Fitting this object with training data
apprentice_chef_model.fit(X_train+ y_train)

# Predicting on the test set
apprentice_chef_model_pred = apprentice_chef_model.predict(X_test)

# SCORING the results
print(Training Score:+ apprentice_chef_model.score(X_train+ y_train).round(3))
print(Testing Score:+  apprentice_chef_model.score(X_test+ y_test).round(3))

################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test+ y_test)

# SCORING the results

test_score = apprentice_chef_model.score(X_train+ y_train).round(3)
train_score = apprentice_chef_model.score(X_train+ y_train).round(3)


