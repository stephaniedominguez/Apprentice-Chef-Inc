# timeit

# Student Name : STEPHANIE DOMINGUEZ
# Cohort       : 1

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
# importing libraries
import pandas as pd                                                         # data science essentials
import matplotlib.pyplot  as plt                                            # data visualization
import seaborn as sns                                                       # enhanced data visualization
import statsmodels.formula.api as smf                                       # linear regression (statsmodels)
from sklearn.model_selection import train_test_split                        # train/test split
from sklearn.linear_model import LinearRegression                           # linear regression (scikit-learn)
from sklearn.preprocessing import StandardScaler                            # standard scaler
#GradientBoostingRegressor https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
from sklearn.linear_model import LogisticRegression                         # logistic regression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier 
# GradientBoostingClassifier logistic regression Model who gave the best result, model used in this template
from sklearn.model_selection import GridSearchCV                            # hyperparameter tuning
from sklearn.metrics import make_scorer                                     # customizable scorer
from sklearn.metrics import roc_auc_score                                   # auc score
from sklearn.ensemble import RandomForestClassifier                         # random forest
from sklearn.neighbors import KNeighborsClassifier                          # KNN for classification
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# specifying file name
file =  'Apprentice_Chef_Dataset.xlsx'

original_df = pd.read_excel(file)

apprentice_chef = original_df


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

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

REVENUE_lo                     = 200
REVENUE_hi                     = 6000 # Data change at this point



##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers


# TOTAL_MEALS_ORDERED
apprentice_chef['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = apprentice_chef.loc[0:,'out_TOTAL_MEALS_ORDERED'][apprentice_chef['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]
condition_lo = apprentice_chef.loc[0:,'out_TOTAL_MEALS_ORDERED'][apprentice_chef['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]

apprentice_chef['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice_chef['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# UNIQUE_MEALS_PURCH
apprentice_chef['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = apprentice_chef.loc[0:,'out_UNIQUE_MEALS_PURCH'][apprentice_chef['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]
condition_lo = apprentice_chef.loc[0:,'out_UNIQUE_MEALS_PURCH'][apprentice_chef['UNIQUE_MEALS_PURCH'] < UNIQUE_MEALS_PURCH_lo]

apprentice_chef['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice_chef['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
apprentice_chef['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = apprentice_chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = apprentice_chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]

apprentice_chef['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice_chef['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# PRODUCT_CATEGORIES_VIEWED
apprentice_chef['out_PRODUCT_CATEGORIES_VIEWED'] = 0
condition_hi = apprentice_chef.loc[0:,'out_PRODUCT_CATEGORIES_VIEWED'][apprentice_chef['PRODUCT_CATEGORIES_VIEWED'] > PRODUCT_CATEGORIES_VIEWED_hi]
condition_lo = apprentice_chef.loc[0:,'out_PRODUCT_CATEGORIES_VIEWED'][apprentice_chef['PRODUCT_CATEGORIES_VIEWED'] < PRODUCT_CATEGORIES_VIEWED_lo]

apprentice_chef['out_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice_chef['out_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# AVG_TIME_PER_SITE_VISIT
apprentice_chef['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = apprentice_chef.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][apprentice_chef['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

apprentice_chef['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# CANCELLATIONS_BEFORE_NOON
apprentice_chef['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = apprentice_chef.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][apprentice_chef['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

apprentice_chef['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# EARLY_DELIVERIES
apprentice_chef['out_EARLY_DELIVERIES'] = 0
condition_hi = apprentice_chef.loc[0:,'out_EARLY_DELIVERIES'][apprentice_chef['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

apprentice_chef['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# LATE_DELIVERIES
apprentice_chef['out_LATE_DELIVERIES'] = 0
condition_hi = apprentice_chef.loc[0:,'out_LATE_DELIVERIES'][apprentice_chef['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]

apprentice_chef['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# AVG_PREP_VID_TIME
apprentice_chef['out_AVG_PREP_VID_TIME'] = 0
condition_hi = apprentice_chef.loc[0:,'out_AVG_PREP_VID_TIME'][apprentice_chef['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
condition_lo = apprentice_chef.loc[0:,'out_AVG_PREP_VID_TIME'][apprentice_chef['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]

apprentice_chef['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice_chef['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# TOTAL_PHOTOS_VIEWED_hi
apprentice_chef['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = apprentice_chef.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][apprentice_chef['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]

apprentice_chef['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)



# REVENUE
apprentice_chef['out_REVENUE'] = 0
condition_hi = apprentice_chef.loc[0:,'out_REVENUE'][apprentice_chef['REVENUE'] > AVG_PREP_VID_TIME_hi]
condition_lo = apprentice_chef.loc[0:,'out_REVENUE'][apprentice_chef['REVENUE'] < REVENUE_lo]

apprentice_chef['out_REVENUE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice_chef['out_REVENUE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
# setting trend-based thresholds
#thresholds based on scatter plot analisys

AVG_TIME_PER_SITE_VISIT_changes_hi      = 400 # trend changes at this point
CONTACTS_W_CUSTOMER_SERVICE_changes_hi  = 12.5  # data scatters above this point
TOTAL_PHOTOS_VIEWED_changes_hi          = 800 #data scatters above this points
UNIQUE_MEALS_PURCH_changes_hi           = 12.5
LATE_DELIVERIES_changes_hi              = 15  #data scatter at no with CROOS_SELL_SUCCESS at 15
FOLLOWED_RECOMMENDATIONS_PCT_changes_hi = 40  #data scatter at no with CROOS_SELL_SUCCESS at 40
REVENUE_changes_hi                      = 6000 # Data change at this point

TOTAL_MEALS_ORDERED_changes_at          = 300   # data inflated only in this point
CANCELLATIONS_BEFORE_NOON_changes_at    = 7   # Trend changes at this point
CANCELLATIONS_AFTER_NOON_changes_at     = 3   # Trend changes at this point
AVG_PREP_VID_TIME_changes_at            = 300 # Trend changes at this points
LARGEST_ORDER_SIZE_changes_at           = 0   # Differnet at 0 revenue of 0
AVG_CLICKS_PER_VISIT_changes_at         = 6   # trend changes before this point

##############################################################################
## Feature Engineering (trend changes)                                      ##
##############################################################################

# developing features (columns) for outliers

########################################
## change above threshold             ##
########################################

# greater than sign

# AVG_TIME_PER_SITE_VISIT
apprentice_chef['change_AVG_TIME_PER_SITE_VISIT'] = 0
condition = apprentice_chef.loc[0:,'change_AVG_TIME_PER_SITE_VISIT'][apprentice_chef['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_changes_hi]

apprentice_chef['change_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
apprentice_chef['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = apprentice_chef.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE'][apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_changes_hi]

apprentice_chef['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# TOTAL_PHOTOS_VIEWED
apprentice_chef['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = apprentice_chef.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][apprentice_chef['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_changes_hi]

apprentice_chef['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# UNIQUE_MEALS_PURCH
apprentice_chef['change_UNIQUE_MEALS_PURCH'] = 0
condition = apprentice_chef.loc[0:,'change_UNIQUE_MEALS_PURCH'][apprentice_chef['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_changes_hi]

apprentice_chef['change_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# LATE_DELIVERIES
apprentice_chef['change_LATE_DELIVERIES'] = 0
condition = apprentice_chef.loc[0:,'change_LATE_DELIVERIES'][apprentice_chef['LATE_DELIVERIES'] > LATE_DELIVERIES_changes_hi]

apprentice_chef['change_LATE_DELIVERIES'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# FOLLOWED_RECOMMENDATIONS_PCT
apprentice_chef['change_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition = apprentice_chef.loc[0:,'change_FOLLOWED_RECOMMENDATIONS_PCT'][apprentice_chef['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_changes_hi]

apprentice_chef['change_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# REVENUE
apprentice_chef['change_REVENUE'] = 0
condition = apprentice_chef.loc[0:,'change_REVENUE'][apprentice_chef['REVENUE'] > REVENUE_changes_hi]

apprentice_chef['change_REVENUE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

########################################
## change at threshold                ##
########################################

# double-equals sign

# TOTAL_MEALS_ORDERED
apprentice_chef['change_TOTAL_MEALS_ORDERED'] = 0
condition = apprentice_chef.loc[0:,'change_TOTAL_MEALS_ORDERED'][apprentice_chef['TOTAL_MEALS_ORDERED'] == TOTAL_MEALS_ORDERED_changes_at]

apprentice_chef['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# CANCELLATIONS_BEFORE_NOON
apprentice_chef['change_CANCELLATIONS_BEFORE_NOON'] = 0
condition = apprentice_chef.loc[0:,'change_CANCELLATIONS_BEFORE_NOON'][apprentice_chef['CANCELLATIONS_BEFORE_NOON'] == CANCELLATIONS_BEFORE_NOON_changes_at]

apprentice_chef['change_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# CANCELLATIONS_AFTER_NOON
apprentice_chef['change_CANCELLATIONS_AFTER_NOON'] = 0
condition = apprentice_chef.loc[0:,'change_CANCELLATIONS_AFTER_NOON'][apprentice_chef['CANCELLATIONS_AFTER_NOON'] == CANCELLATIONS_AFTER_NOON_changes_at]

apprentice_chef['change_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# AVG_PREP_VID_TIME
apprentice_chef['change_AVG_PREP_VID_TIME'] = 0
condition = apprentice_chef.loc[0:,'change_AVG_PREP_VID_TIME'][apprentice_chef['AVG_PREP_VID_TIME'] == AVG_PREP_VID_TIME_changes_at]

apprentice_chef['change_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# LARGEST_ORDER_SIZE
apprentice_chef['change_LARGEST_ORDER_SIZE'] = 0
condition = apprentice_chef.loc[0:,'change_LARGEST_ORDER_SIZE'][apprentice_chef['LARGEST_ORDER_SIZE'] == LARGEST_ORDER_SIZE_changes_at]

apprentice_chef['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


# AVG_CLICKS_PER_VISIT
apprentice_chef['change_AVG_CLICKS_PER_VISIT'] = 0
condition = apprentice_chef.loc[0:,'change_AVG_CLICKS_PER_VISIT'][apprentice_chef['AVG_CLICKS_PER_VISIT'] == AVG_CLICKS_PER_VISIT_changes_at]

apprentice_chef['change_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#Creating categorical data from columns mails
#see if makes a different the type of mail
# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in apprentice_chef.iterrows():
    
    # splitting email domain at ''@'
    split_email = apprentice_chef.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# safety measure in case of multiple concatenations
apprentice_chef_email = apprentice_chef


# renaming column to concatenate
email_df.columns = ['0' , 'personal_email_domain']


# concatenating personal_email_domain with apprentice_chef_email DataFrame
apprentice_chef_email = pd.concat([apprentice_chef_email, email_df['personal_email_domain']],
                     axis = 1)


# email domain types
personal_email_domains = ['@gmail.com',
                            '@yahoo.com',
                            '@protonmail.com']

professional_email_domains  = [  '@mmm.com',
                            '@amex.com',
                            '@apple.com',
                            '@boeing.com',
                            '@caterpillar.com',
                            '@chevron.com',
                            '@cisco.com',
                            '@cocacola.com',
                            '@disney.com',
                            '@dupont.com',
                            '@exxon.com',
                            '@ge.org' ,
                            '@goldmansacs.com',
                            '@homedepot.com',
                            '@ibm.com',
                            '@intel.com',
                            '@jnj.com',
                            '@jpmorgan.com',
                            '@mcdonalds.com',
                            '@merck.com',
                            '@microsoft.com',
                            '@nike.com',
                            '@pfizer.com',
                            '@pg.com',
                            '@travelers.com',
                            '@unitedtech.com',
                            '@unitedhealth.com',
                            '@verizon.com',
                            '@visa.com',
                            '@walmart.com']
junk_email_domains = ['@me.com',
                        '@aol.com', 
                        '@hotmail.com', 
                        '@live.com', 
                        '@msn.com', 
                        '@passport.com']

# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in apprentice_chef_email['personal_email_domain']:
    
    if '@' + domain in personal_email_domains:
        placeholder_lst.append('personal')
        

    elif '@' + domain in professional_email_domains:
        placeholder_lst.append('professional')
        
    elif '@' + domain in junk_email_domains:
        placeholder_lst.append('junk')


    else:
            print('Unknown')


# concatenating with original DataFrame
apprentice_chef_email['domain_group'] = pd.Series(placeholder_lst)


# checking results
apprentice_chef_email['domain_group'].value_counts()

#creating new columns for categorical data 
#adding it to original data set apprentice_chef
apprentice_chef['domain_group'] = apprentice_chef_email['domain_group']
#Creating dummy variables from email
dummy = pd.get_dummies(apprentice_chef['domain_group'])
#concat to the original dataframe
apprentice_chef = pd.concat([apprentice_chef,dummy], axis=1 )

#Creating new variables for the model
#no output
# REVENUE_TOTAL_MEALS_ORDERED
#Revenue for each total meals order 
apprentice_chef['REVENUE_TOTAL_MEALS_ORDERED'] = 0
apprentice_chef['REVENUE_TOTAL_MEALS_ORDERED'] = apprentice_chef['REVENUE']/apprentice_chef['TOTAL_MEALS_ORDERED']

#TOTAL_CANCELLATION
#Revenue for each total meals order 
apprentice_chef['TOTAL_CANCELLATION'] = 0
apprentice_chef['TOTAL_CANCELLATION'] = apprentice_chef['CANCELLATIONS_BEFORE_NOON']+apprentice_chef['CANCELLATIONS_AFTER_NOON']

#Revenue for each total meals order 
apprentice_chef['REVENUE_TOTAL_CANCELLATION'] = 0
apprentice_chef['REVENUE_TOTAL_CANCELLATION'] = abs(apprentice_chef['TOTAL_MEALS_ORDERED'] - apprentice_chef['TOTAL_CANCELLATION']) / apprentice_chef['REVENUE']

#DELIEVERY_EFFECT
#Sum of delivery effect on people since they are busy
apprentice_chef['DELIEVERY_EFFECT'] = 0
apprentice_chef['DELIEVERY_EFFECT'] = apprentice_chef['EARLY_DELIVERIES'] + apprentice_chef['LATE_DELIVERIES']

#TOTAL_LOGINS
#Sum of delivery effect on people since they are busy
apprentice_chef['TOTAL_LOGINS'] = 0
apprentice_chef['TOTAL_LOGINS'] = apprentice_chef['MOBILE_LOGINS'] + apprentice_chef['PC_LOGINS']

#TOTAL_LOCKER
#Sum of delivery effect on people since they are busy
apprentice_chef['TOTAL_LOCKER'] = 0
apprentice_chef['TOTAL_LOCKER'] = apprentice_chef['PACKAGE_LOCKER'] + apprentice_chef['REFRIGERATED_LOCKER']

#Variables that are not junk
#REVENUE_AVG_CLICKS_PER_VISIT
apprentice_chef['REVENUE_AVG_CLICKS_PER_VISIT'] = 0
apprentice_chef['REVENUE_AVG_CLICKS_PER_VISIT'] = apprentice_chef['AVG_CLICKS_PER_VISIT'] * apprentice_chef['REVENUE']


#Variables that are not junk
#REVENUE_CONTACTS_W_CUSTOMER_SERVICE
apprentice_chef['REVENUE_CONTACTS_W_CUSTOMER_SERVICE'] = 0
apprentice_chef['REVENUE_CONTACTS_W_CUSTOMER_SERVICE'] = apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE'] * apprentice_chef['REVENUE']


#Variables that are not junk
#REVENUE_CANCELLATIONS_BEFORE_NOON
apprentice_chef['REVENUE_CANCELLATIONS_BEFORE_NOON'] = 0
apprentice_chef['REVENUE_CANCELLATIONS_BEFORE_NOON'] = apprentice_chef['CANCELLATIONS_BEFORE_NOON'] * apprentice_chef['REVENUE']


#Variables that are not junk
#REVENUE_CANCELLATIONS_AFTER_NOON
apprentice_chef['REVENUE_CANCELLATIONS_AFTER_NOON'] = 0
apprentice_chef['REVENUE_CANCELLATIONS_AFTER_NOON'] = apprentice_chef['CANCELLATIONS_AFTER_NOON'] * apprentice_chef['REVENUE']


# building a full model
# blueprinting a model type
apprentice_chef_data   = apprentice_chef.drop(['CROSS_SELL_SUCCESS',
                               'NAME',
                                'EMAIL',
                                'FIRST_NAME',
                                'FAMILY_NAME' ,
                                'domain_group'            
                               ],
                                axis = 1)
logit_full = smf.logit(formula = """ CROSS_SELL_SUCCESS ~
                                 CONTACTS_W_CUSTOMER_SERVICE+  
                                  PRODUCT_CATEGORIES_VIEWED+  
                                  MOBILE_NUMBER+  
                                  CANCELLATIONS_BEFORE_NOON+  
                                  CANCELLATIONS_AFTER_NOON+  
                                  TASTES_AND_PREFERENCES+  
                                  MOBILE_LOGINS+  
                                  PC_LOGINS+ 
                                  WEEKLY_PLAN+ 
                                  MASTER_CLASSES_ATTENDED+ 
                                  AVG_PREP_VID_TIME+ 
                                  FOLLOWED_RECOMMENDATIONS_PCT+ 
                                  EARLY_DELIVERIES+  
                                  PACKAGE_LOCKER+  
                                  FOLLOWED_RECOMMENDATIONS_PCT+  
                                  LARGEST_ORDER_SIZE+  
                                  MEDIAN_MEAL_RATING+  
                                  AVG_CLICKS_PER_VISIT+  
                                  out_UNIQUE_MEALS_PURCH+ 
                                   out_CONTACTS_W_CUSTOMER_SERVICE+ 
                                  REVENUE_TOTAL_MEALS_ORDERED+
                                  junk+
                                  professional
                               """,
                                data = apprentice_chef)


# fitting the model object
logit_full = logit_full.fit()


# checking the results SUMMARY
#logit_full.summary()

################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25
# preparing response variable data
apprentice_chef_target = apprentice_chef.loc[:, 'CROSS_SELL_SUCCESS']
#creating the list for independent variables
x_variables = [             'REVENUE', 
                           'TOTAL_MEALS_ORDERED', 
                           'UNIQUE_MEALS_PURCH',
                           'CONTACTS_W_CUSTOMER_SERVICE',
                           'PRODUCT_CATEGORIES_VIEWED',
                           'AVG_TIME_PER_SITE_VISIT',
                           'MOBILE_NUMBER',
                           'CANCELLATIONS_BEFORE_NOON',
                           'CANCELLATIONS_AFTER_NOON', 
                           'TASTES_AND_PREFERENCES', 
                           'MOBILE_LOGINS', 
                           'WEEKLY_PLAN', 
                           'LATE_DELIVERIES', 
                           'PACKAGE_LOCKER', 
                           'FOLLOWED_RECOMMENDATIONS_PCT', 
                           'change_CONTACTS_W_CUSTOMER_SERVICE',
                           'change_FOLLOWED_RECOMMENDATIONS_PCT',
                           'change_REVENUE',
                           'REVENUE_TOTAL_MEALS_ORDERED',
                           'DELIEVERY_EFFECT',
                           'professional'
               ]



#Creating new data set with the independent variable called apprentice_chef_data
apprentice_chef_data = apprentice_chef_data[x_variables]

#INSTANTIATING a model object: StandardScaler()
apprentice_chef_model = StandardScaler()


# FITTING the training data
apprentice_chef_model.fit(apprentice_chef_data)


# TRANSFORMING  model
apprentice_chef_X_scaled = apprentice_chef_model.transform(apprentice_chef_data)


# converting to DataFrame
apprentice_chef_model_scaled_df = pd.DataFrame(apprentice_chef_X_scaled)

# train-test split with the scaled data
X_train, X_test, y_train, y_test = train_test_split(apprentice_chef_model_scaled_df,
                                                    apprentice_chef_target,
                                                    test_size = 0.25,
                                                    random_state = 222,
                                                    )



################################################################################
# Final Model (instantiate, fit, and predict) GradientBoostingRegressor
#GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. 
# In each stage a regression tree is fit on the negative gradient of the given loss function.
################################################################################

# use this space to instantiate, fit, and predict on your final model

# Instantiating a GradientBoostingClassifier as logreg

logreg = GradientBoostingClassifier(max_depth=1, 
            subsample=0.4,
            max_features=.2,
            n_estimators=210,                                
            random_state=300,
            learning_rate=0.09)

# FITTING the training data
logreg_fit = logreg.fit(X_train, y_train)

# PREDICTING based on the testing set
logreg_pred = logreg_fit.predict(X_test)


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

# SCORING the results
# SCORING the results
print('Training ACCURACY:', logreg_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', logreg_fit.score(X_test, y_test).round(4))

# area under the roc curve (auc)
print('AUC',roc_auc_score(y_true  = y_test,
              y_score = logreg_pred).round(4))


test_score  = logreg_fit.score(X_train, y_train).round(3)
train_score = logreg_fit.score(X_train, y_train).round(3)
auc         = roc_auc_score(y_true  = y_test,
              y_score = logreg_pred).round(4)
