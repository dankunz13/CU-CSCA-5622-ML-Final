
import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import dataframe_image as dfi
import numpy as np


####################################################################################################
# Data file information
####################################################################################################

# Eating Habits and Physical Condition in Individuals from Columbia, Peru, and Mexico
# 16 Features

# Gender : [Male or Female]
# Age    : Float - age
# Height : Float - height in meters
# Weight : Float - weight in kilograms
# family_history_with_overweight : [yes or no] - Has family member been or is a family member overweight
# FAVC   : [yes or no] - Frequently eats high caloric foods
# FCVC   : Float - Meals per day that vegetables are eaten with  (Mix of integer and float) [Never, Sometimes, Always] - Usually eat vegetables in your meals ???
# NCP    : Float - Main meals eaten daily [Between 1 and 2, Three, More than Three] ???
# CAEC   : [No, Sometimes, Frequently, Always] - Eats food between meals
# SMOKE  : [yes or no] - Smokes
# CH2O   : Float - Daily water intake in liters (Mix of integer and float) [Less than 1 liter, Between 1 and 2 Liters, More than 2L] ???
# SCC    : [yes or no] - Monitors daily calories eaten
# FAF    : Float - Frequency of physical activity (Mix of integer and float) [I do not have, 1 to 2 days, 2 to 4 days, 4 to 5 days] ???
# TUE    : Float - Time using technilogical devices (cell phone, videogames, tv, computer, others) [0 to 2 Hours, 3-5 Hours, More than 5 Hours] ???
# CALC   : [No, Sometimes, Frequently, Always] - Frequency of drinking alcholog [Never, Sometimes, Frequently, Always]
# MTRANS : [Automobile, Bike, Motorbike, Public_Transportation, Walking] - Usual transportation
# NObeyesdad : [Underweight, Normal, Overweight I, Overweight II, Obesity I, Obesity II, Obesity III] - Obesity level

# Mass Body Index = Weight / (Height*Height)

####################################################################################################
# Load data file
####################################################################################################

obesity_df = pd.read_csv ('data\\ObesityDataSet_raw_and_data_sinthetic.csv')


####################################################################################################
# Validation of no missing values
####################################################################################################

assert not obesity_df.isnull ().sum().any (), 'Dataframe contains null values'
# print (f'Missing Values:\n{obesity_df.isnull ().sum()}')


####################################################################################################
# Remove Height and Weight predictor variables
####################################################################################################
 
obesity_df = obesity_df.drop (columns=['Height', 'Weight'])


####################################################################################################
# Show summary information about data columns
####################################################################################################

# print (obesity_df.info ())


####################################################################################################
# Show unique values for each column before encoding
####################################################################################################

# unique_values = pd.Series ({column_name: obesity_df[column_name].unique () for column_name in obesity_df})
# print (unique_values)


####################################################################################################
# Boxplots for Age, Weight, and Height
####################################################################################################

# plt.clf ()
# figure, axes = plt.subplots (1, 1, figsize=(5, 5), sharey=False)
# # figure, axes = plt.subplots (1, 3, figsize=(15, 5), sharey=False)
# axes.boxplot (obesity_df['Age'])
# axes.set_title ('Age')
# axes.set_ylabel ('Values')
# # axes[0].boxplot (obesity_df['Age'])
# # axes[0].set_title ('Age')
# # axes[0].set_ylabel ('Values')
# # axes[1].boxplot (obesity_df['Height'])
# # axes[1].set_title ('Height')
# # axes[1].set_ylabel ('Values')
# # axes[2].boxplot (obesity_df['Weight'])
# # axes[2].set_title ('Weight')
# # axes[2].set_ylabel ('Values')
# plt.tight_layout ()
# # plt.show ()
# # plt.savefig ('images\\4 - Data Cleaning\\Boxplot - Age, Height, Weight.png', dpi=96, bbox_inches='tight')
# plt.savefig ('images\\4 - Data Cleaning\\Boxplot - Age.png', dpi=96, bbox_inches='tight')


####################################################################################################
# Histogram for TechnologyUsage Before Processing
####################################################################################################

# category_counts = obesity_df['TUE'].value_counts ()

# plt.clf ()
# obesity_df['TUE'].hist (bins=20, edgecolor='black')
# # plt.xticks (ticks=[0, 1, 2], labels=['0 to 2 hours', '3 to 5 hours', 'Over 5 hours'], rotation=0)
# plt.title ('Distribution of TechnologyUsage (Before)')
# plt.xlabel = 'Technology Usage Categories'
# plt.ylabel = 'Frequency'
# plt.grid (False)
# plt.savefig ('images\\4 - Data Cleaning\\Histogram - Technology Usage - Before.png', dpi=96, bbox_inches='tight')




####################################################################################################
# Histograms for Age, Weight, and Height
####################################################################################################

# plt.clf ()
# obesity_df.hist (column=['Age'], edgecolor='black')
# plt.title ('Histogram of Age')
# plt.grid (False)
# plt.savefig ('images\\histogram_Age.png', dpi=96, bbox_inches='tight')

# print (f'Age Summary:\n{obesity_df['Age'].describe ()}')

# plt.clf ()
# obesity_df.hist (column=['Height'], edgecolor='black')
# plt.title ('Histogram of Height')
# plt.grid (False)
# plt.savefig ('images\\histogram_Height.png', dpi=96, bbox_inches='tight')

# print (f'Height Summary:\n{obesity_df['Height'].describe ()}')

# plt.clf ()
# obesity_df.hist (column=['Weight'], edgecolor='black')
# plt.title ('Histogram of Weight')
# plt.grid (False)
# plt.savefig ('images\\histogram_Weight.png', dpi=96, bbox_inches='tight')

# print (f'Weight Summary:\n{obesity_df['Weight'].describe ()}')




####################################################################################################
# Convert categorical columns to numeric
####################################################################################################

# Convert Gender to numeric - Female = 1, Male = 0

obesity_df['iGender'] = None
obesity_df.loc[obesity_df['Gender'] == 'Male',   'iGender'] = 0
obesity_df.loc[obesity_df['Gender'] == 'Female', 'iGender'] = 1


# Convert family_history_with_overweight to numeric - yes = 1, no = 0

obesity_df['iFamilyHistory'] = None
obesity_df.loc[obesity_df['family_history_with_overweight'] == 'no',  'iFamilyHistory'] = 0
obesity_df.loc[obesity_df['family_history_with_overweight'] == 'yes', 'iFamilyHistory'] = 1


# Convert FAVC to numeric - yes = 1, no = 0

obesity_df['iFAVC'] = None
obesity_df.loc[obesity_df['FAVC'] == 'no',  'iFAVC'] = 0
obesity_df.loc[obesity_df['FAVC'] == 'yes', 'iFAVC'] = 1


# Convert FCVC to numeric
#  - Never     = 0
#  - Sometimes = 1
#  - Always    = 2

obesity_df['iFCVC'] = None
obesity_df.loc[(obesity_df['FCVC'] >= 1.0) & (obesity_df['FCVC'] <  1.5),  'iFCVC'] = 0
obesity_df.loc[(obesity_df['FCVC'] >= 1.5) & (obesity_df['FCVC'] <  2.5),  'iFCVC'] = 1
obesity_df.loc[(obesity_df['FCVC'] >= 2.5) & (obesity_df['FCVC'] <= 3.0),  'iFCVC'] = 2


# Convert NCP to numeric
#  - Between 1 and 2 Meals = 1
#  - Three                 = 2
#  - More Than Three       = 3

obesity_df['iNCP'] = None
obesity_df.loc[(obesity_df['NCP'] >= 1.0) & (obesity_df['NCP'] <  2.0),  'iNCP'] = 1
obesity_df.loc[(obesity_df['NCP'] >= 2.0) & (obesity_df['NCP'] <  3.0),  'iNCP'] = 2
obesity_df.loc[(obesity_df['NCP'] >= 3.0) & (obesity_df['NCP'] <= 4.0),  'iNCP'] = 3


# Convert CAEC to numeric
#  - No         = 1
#  - Sometimes  = 2
#  - Frequently = 3
#  - Always     = 4

obesity_df['iCAEC'] = None
obesity_df.loc[obesity_df['CAEC'] == 'no',         'iCAEC'] = 1
obesity_df.loc[obesity_df['CAEC'] == 'Sometimes',  'iCAEC'] = 2
obesity_df.loc[obesity_df['CAEC'] == 'Frequently', 'iCAEC'] = 3
obesity_df.loc[obesity_df['CAEC'] == 'Always',     'iCAEC'] = 4


# Convert SMOKE to numeric - yes = 1, no = 0

obesity_df['iSmoke'] = None
obesity_df.loc[obesity_df['SMOKE'] == 'no',  'iSmoke'] = 0
obesity_df.loc[obesity_df['SMOKE'] == 'yes', 'iSmoke'] = 1


# Convert CH2O to numeric
#  - Less Than 1L          = 1
#  - Between 1L and 2L     = 2
#  - More Than Three       = 3

obesity_df['iCH2O'] = None
obesity_df.loc[(obesity_df['CH2O'] >= 1.0) & (obesity_df['CH2O'] <  1.5),  'iCH2O'] = 1
obesity_df.loc[(obesity_df['CH2O'] >= 1.5) & (obesity_df['CH2O'] <  2.5),  'iCH2O'] = 2
obesity_df.loc[(obesity_df['CH2O'] >= 2.5) & (obesity_df['CH2O'] <= 3.0),  'iCH2O'] = 3


# Convert SCC to numeric - yes = 1, no = 0

obesity_df['iSCC'] = None
obesity_df.loc[obesity_df['SCC'] == 'no',  'iSCC'] = 0
obesity_df.loc[obesity_df['SCC'] == 'yes', 'iSCC'] = 1


# Convert FAF to numeric (Assumption is that 0 was an answer of Never, which does not occur in the data (RED FLAG))
#  - Never       = 0
#  - 1 or 2 Days = 1
#  - 2 or 3 Days = 2
#  - 4 or 5 Days = 3

obesity_df['iFAF'] = None
obesity_df.loc[(obesity_df['FAF'] == 0.0),                                'iFAF'] = 0
obesity_df.loc[(obesity_df['FAF'] >  0.0) & (obesity_df['FAF'] <= 1.0),  'iFAF'] = 1
obesity_df.loc[(obesity_df['FAF'] >  1.0) & (obesity_df['FAF'] <= 2.0),  'iFAF'] = 2
obesity_df.loc[(obesity_df['FAF'] >  2.0) & (obesity_df['FAF'] <= 3.0),  'iFAF'] = 3


# Convert TUE to numeric (Round float numbers to 0, 1, or 2 to get categories (RED FLAG))
#  - 0 to 2 Hours      = 1
#  - 3 to 5 Hours      = 2
#  - More Than 5 Hours = 3

obesity_df['iTUE'] = None
obesity_df.loc[(obesity_df['TUE'] >= 0.0) & (obesity_df['TUE'] <= 0.5),  'iTUE'] = 1
obesity_df.loc[(obesity_df['TUE'] >  0.5) & (obesity_df['TUE'] <= 1.5),  'iTUE'] = 2
obesity_df.loc[(obesity_df['TUE'] >  1.5) & (obesity_df['TUE'] <= 2.0),  'iTUE'] = 3


# Convert CALC to numeric
#  - No         = 1
#  - Sometimes  = 2
#  - Frequently = 3
#  - Always     = 4

obesity_df['iCALC'] = None
obesity_df.loc[obesity_df['CALC'] == 'no',         'iCALC'] = 1
obesity_df.loc[obesity_df['CALC'] == 'Sometimes',  'iCALC'] = 2
obesity_df.loc[obesity_df['CALC'] == 'Frequently', 'iCALC'] = 3
obesity_df.loc[obesity_df['CALC'] == 'Always',     'iCALC'] = 4


# Convert MTRANS to numeric
#  - Automobile            = 1
#  - Bike                  = 2
#  - Motorbike             = 3
#  - Public_Transportation = 4
#  - Walking               = 5

obesity_df['iMTRANS'] = None
obesity_df.loc[obesity_df['MTRANS'] == 'Automobile',            'iMTRANS'] = 1
obesity_df.loc[obesity_df['MTRANS'] == 'Bike',                  'iMTRANS'] = 2
obesity_df.loc[obesity_df['MTRANS'] == 'Motorbike',             'iMTRANS'] = 3
obesity_df.loc[obesity_df['MTRANS'] == 'Public_Transportation', 'iMTRANS'] = 4
obesity_df.loc[obesity_df['MTRANS'] == 'Walking',               'iMTRANS'] = 5


# Convert NObeyesdad to numeric
#  - Insufficient_Weight = 1
#  - Normal_Weight       = 2
#  - Overweight_Level_I  = 3
#  - Overweight_Level_II = 4
#  - Obesity_Type_I      = 5
#  - Obesity_Type_II     = 6
#  - Obesity_Type_III    = 7

obesity_df['iNObeyesdad'] = None
obesity_df.loc[obesity_df['NObeyesdad'] == 'Insufficient_Weight', 'iNObeyesdad'] = 1
obesity_df.loc[obesity_df['NObeyesdad'] == 'Normal_Weight',       'iNObeyesdad'] = 2
obesity_df.loc[obesity_df['NObeyesdad'] == 'Overweight_Level_I',  'iNObeyesdad'] = 3
obesity_df.loc[obesity_df['NObeyesdad'] == 'Overweight_Level_II', 'iNObeyesdad'] = 4
obesity_df.loc[obesity_df['NObeyesdad'] == 'Obesity_Type_I',      'iNObeyesdad'] = 5
obesity_df.loc[obesity_df['NObeyesdad'] == 'Obesity_Type_II',     'iNObeyesdad'] = 6
obesity_df.loc[obesity_df['NObeyesdad'] == 'Obesity_Type_III',    'iNObeyesdad'] = 7

# obesity_df['ObesityCateogry'] = None
# obesity_df.loc[obesity_df['iNObeyesdad'] == 1, 'ObesityCateogry'] = 'Underweight'
# obesity_df.loc[obesity_df['iNObeyesdad'] == 2, 'ObesityCateogry'] = 'Normal'
# obesity_df.loc[obesity_df['iNObeyesdad'] == 3, 'ObesityCateogry'] = 'Overweight I'
# obesity_df.loc[obesity_df['iNObeyesdad'] == 4, 'ObesityCateogry'] = 'Overweight II'
# obesity_df.loc[obesity_df['iNObeyesdad'] == 5, 'ObesityCateogry'] = 'Obesity I'
# obesity_df.loc[obesity_df['iNObeyesdad'] == 6, 'ObesityCateogry'] = 'Obesity II'
# obesity_df.loc[obesity_df['iNObeyesdad'] == 7, 'ObesityCateogry'] = 'Obesity III'

# print (obesity_df.head (5))


# Replace previous columns with new columns

# obesity_df = obesity_df[['iGender', 'Age', 'Height', 'Weight', 'iFamilyHistory', 'iFAVC', 'iFCVC', 'iNCP', 'iCAEC', 'iSmoke', 'iCH2O', 'iSCC', 'iFAF', 'iTUE', 'iCALC', 'iMTRANS', 'iNObeyesdad']]
obesity_df = obesity_df[['iGender', 'Age', 'iFamilyHistory', 'iFAVC', 'iFCVC', 'iNCP', 'iCAEC', 'iSmoke', 'iCH2O', 'iSCC', 'iFAF', 'iTUE', 'iCALC', 'iMTRANS', 'iNObeyesdad']]


####################################################################################################
# Histogram for TechnologyUsage After Processing
####################################################################################################

# category_counts = obesity_df['iTUE'].value_counts ()

# plt.clf ()
# obesity_df['iTUE'].hist (bins=20, edgecolor='black')
# plt.xticks (ticks=[1, 2, 3], labels=['0 to 2 hours', '3 to 5 hours', 'Over 5 hours'], rotation=0)
# plt.title ('Distribution of TechnologyUsage (After)')
# plt.xlabel = 'Technology Usage Categories'
# plt.ylabel = 'Frequency'
# plt.grid (False)
# # plt.show ()
# plt.savefig ('images\\4 - Data Cleaning\\Histogram - Technology Usage - After.png', dpi=96, bbox_inches='tight')


####################################################################################################
# Show unique values for each column after encoding
####################################################################################################

# unique_values = pd.Series ({column_name: obesity_df[column_name].unique () for column_name in obesity_df})
# print (unique_values)




####################################################################################################
# Rename column names
####################################################################################################

obesity_df = obesity_df.rename (columns={
    'iGender': 'Gender',
    'Age': 'Age',
    'Height': 'Height',
    'Weight': 'Weight',
    'iFamilyHistory': 'FamilyHistory',
    'iFAVC': 'HighCalorieFoods',
    'iFCVC': 'VegetablesWithMeals',
    'iNCP': 'MainDailyMeals',
    'iCAEC': 'SnackFrequency',
    'iSmoke': 'Smoke',
    'iCH2O': 'DailyWater',
    'iSCC': 'MonitorCalories',
    'iFAF': 'WeeklyActivity',
    'iTUE': 'TechnologyUsage',
    'iCALC': 'AlcoholFrequency',
    'iMTRANS': 'Transportation',
    'iNObeyesdad': 'ObesityCateogry'})


####################################################################################################
# Change object columns to integer
####################################################################################################

obesity_df['Gender'] = obesity_df['Gender'].apply (pd.to_numeric)
obesity_df['FamilyHistory'] = obesity_df['FamilyHistory'].apply (pd.to_numeric)
obesity_df['HighCalorieFoods'] = obesity_df['HighCalorieFoods'].apply (pd.to_numeric)
obesity_df['VegetablesWithMeals'] = obesity_df['VegetablesWithMeals'].apply (pd.to_numeric)
obesity_df['MainDailyMeals'] = obesity_df['MainDailyMeals'].apply (pd.to_numeric)
obesity_df['SnackFrequency'] = obesity_df['SnackFrequency'].apply (pd.to_numeric)
obesity_df['Smoke'] = obesity_df['Smoke'].apply (pd.to_numeric)
obesity_df['DailyWater'] = obesity_df['DailyWater'].apply (pd.to_numeric)
obesity_df['MonitorCalories'] = obesity_df['MonitorCalories'].apply (pd.to_numeric)
obesity_df['WeeklyActivity'] = obesity_df['WeeklyActivity'].apply (pd.to_numeric)
obesity_df['TechnologyUsage'] = obesity_df['TechnologyUsage'].apply (pd.to_numeric)
obesity_df['AlcoholFrequency'] = obesity_df['AlcoholFrequency'].apply (pd.to_numeric)
obesity_df['Transportation'] = obesity_df['Transportation'].apply (pd.to_numeric)
obesity_df['ObesityCateogry'] = obesity_df['ObesityCateogry'].apply (pd.to_numeric)

# print (obesity_df.info ())


####################################################################################################
# Write to file for validation
####################################################################################################

# obesity_df.to_csv (os.path.join (current_path, 'data/obesity_df.csv'), index=None, sep=',', mode='w+')


####################################################################################################
# Barplot of ObesityCategory
# ObesityCateogry | 1<br>2<br>3<br>4<br>5<br>6<br>7 | Underweight<br>Normal<br>Overweight I<br>Overweight II<br>Obesity I<br>Obesity II<br>Obesity III
####################################################################################################

# plt.clf ()
# obesity_df['ObesityCateogry'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Underweight (1)', 'Normal (2)', 'Overweight I (3)', 'Overweight II(4)', 'Obesity I (5)', 'Obesity II (6)', 'Obesity III (7)'], rotation=270)
# plt.title ('Distribution of ObesityCateogry')
# plt.xlabel = 'ObesityCateogry'
# # plt.show ()
# plt.savefig ('images\\4 - Data Cleaning\\Barplot - ObesityCategory.png', dpi=96, bbox_inches='tight')


####################################################################################################
# Split data into predictors and response variables
####################################################################################################

# Set response variable to seperate dataframe

obesity_response_df = obesity_df['ObesityCateogry']
# print (f'obesity_response_df:\n{obesity_response_df[:5]}')

# Set predictor varables to separate dataframe

obesity_predictors_df = obesity_df.drop ('ObesityCateogry', axis=1)
# print (f'obesity_predictors_df:\n{obesity_predictors_df[:5]}')


####################################################################################################
# Correlation Matrix / Heatmap of predictor variables
####################################################################################################

# correlation_matrix = obesity_predictors_df.corr ()
# # print (correlation_matrix)

# plt.clf ()
# plt.figure (figsize=(10, 10))
# axes = plt.axes ()
# sns.heatmap (data=correlation_matrix, ax=axes, cmap='coolwarm', annot=True, annot_kws={'size' : 8}, fmt='.2f', square=True, linewidths=.5, linecolor='white', cbar=True)
# axes.xaxis.tick_top ()
# axes.yaxis.tick_left ()
# axes.tick_params (axis='x', rotation=90, labelsize=8)
# axes.tick_params (axis='y', rotation=0, labelsize=8)
# axes.set_title ('Correlation Matrix for Obesity Features', fontsize=24)
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Correlation Matrix - Predictor Features - Baseline.png', dpi=96, bbox_inches='tight')


####################################################################################################
# Heatmap of Correlation Matrix for predictor variables
####################################################################################################

# correlation_matrix_full = obesity_df.corr ()
# # print (correlation_matrix_full)

# plt.clf ()
# plt.figure (figsize=(10, 8))
# axes = plt.axes ()
# sns.heatmap (data=correlation_matrix_full, ax=axes, cmap='coolwarm', annot=True, annot_kws={'size' : 8}, fmt='.2f', square=True, linewidths=.5, linecolor='white', cbar=True)
# axes.xaxis.tick_top ()
# axes.yaxis.tick_left ()
# axes.tick_params (axis='x', rotation=90, labelsize=8)
# axes.tick_params (axis='y', rotation=0, labelsize=8)
# axes.set_title ('Heatmap of Correlation Matrix', fontsize=24)
# column_index = list(correlation_matrix_full.columns).index ('ObesityCateogry')
# plt.axvline (column_index, color='black', linestyle='-', linewidth=2)  # Highlight vertical line
# plt.axhline (column_index, color='black', linestyle='-', linewidth=2)  # Highlight horizontal line
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Correlation Matrix - All Features - Baseline.png', dpi=96, bbox_inches='tight')

# print ('Figure: Heatmap of Correlation Matrix for Predictor and Result Variables')


####################################################################################################
# Histogram and Boxplot for Age
####################################################################################################

# plt.clf ()
# figure, axes = plt.subplots (1, 2, figsize=(10, 5), sharey=False)
# axes[0].hist (obesity_df['Age'], edgecolor='black')
# axes[0].set_title ('Age')
# axes[0].set_ylabel ('Count')
# axes[1].boxplot (obesity_df['Age'])
# axes[1].set_title ('Age')
# axes[1].set_ylabel ('Values')
# plt.tight_layout ()
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Histogram & Boxplot - Age.png', dpi=96, bbox_inches='tight')

# print (f'Age Summary:\n{obesity_df['Age'].describe ()}')


# ####################################################################################################
# # Histograms and Boxplot for Height
# ####################################################################################################

# plt.clf ()
# figure, axes = plt.subplots (1, 2, figsize=(10, 5), sharey=False)
# axes[0].hist (obesity_df['Height'], edgecolor='black')
# axes[0].set_title ('Height')
# axes[0].set_ylabel ('Count')
# axes[1].boxplot (obesity_df['Height'])
# axes[1].set_title ('Height')
# axes[1].set_ylabel ('Values')
# plt.tight_layout ()
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Histogram & Boxplot - Height.png', dpi=96, bbox_inches='tight')

# # print (f'Height Summary:\n{obesity_df['Height'].describe ()}')


# ####################################################################################################
# # Histogram and Boxplot for Weight
# ####################################################################################################

# plt.clf ()
# figure, axes = plt.subplots (1, 2, figsize=(10, 5), sharey=False)
# axes[0].hist (obesity_df['Weight'], edgecolor='black')
# axes[0].set_title ('Weight')
# axes[0].set_ylabel ('Count')
# axes[1].boxplot (obesity_df['Weight'])
# axes[1].set_title ('Weight')
# axes[1].set_ylabel ('Values')
# plt.tight_layout ()
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Histogram & Boxplot - Weight.png', dpi=96, bbox_inches='tight')

# # print (f'Weight Summary:\n{obesity_df['Weight'].describe ()}')


####################################################################################################
# Barplot of Gender
# | Gender | 0<br>1 | Male<br>Female |
####################################################################################################

# | Gender | 0<br>1 | Male<br>Female  |

# plt.clf ()
# obesity_df['Gender'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1], labels=['Male (0)', 'Female (1)'], rotation=0)
# plt.title ('Distribution of Gender')
# plt.xlabel = 'Gender'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - Gender.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of FamilyHistory
# | FamilyHistory | 1<br>0 | yes<br>no |
####################################################################################################

# plt.clf ()
# obesity_df['FamilyHistory'].value_counts ().sort_index (ascending=False).plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1], labels=['Yes (1)', 'No (0)'], rotation=0)
# plt.title ('Distribution of FamilyHistory')
# plt.xlabel = 'FamilyHistory'
# plt.ylabel = 'Frequency'
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Barplot - FamilyHistory.png', dpi=96, bbox_inches='tight')


####################################################################################################
# Barplot of HighCalorieFoods
# HighCalorieFoods | 1<br>0 | yes<br>no
####################################################################################################

# plt.clf ()
# obesity_df['HighCalorieFoods'].value_counts ().sort_index (ascending=False).plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1], labels=['Yes (1)', 'No (0)'], rotation=0)
# plt.title ('Distribution of HighCalorieFoods')
# plt.xlabel = 'HighCalorieFoods'
# plt.ylabel = 'Frequency'
# # plt.show ()
# plt.savefig ('images\\5 - EDA\\Barplot - HighCalorieFoods.png', dpi=96, bbox_inches='tight')


####################################################################################################
# Barplot of VegetablesWithMeals
# | VegetablesWithMeals | 0<br>1<br>2 | Never<br>Sometimes<br>Always |
####################################################################################################

# plt.clf ()
# obesity_df['VegetablesWithMeals'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2], labels=['Never (0)', 'Sometimes (1)', 'Always (2)'], rotation=0)
# plt.title ('Distribution of VegetablesWithMeals')
# plt.xlabel = 'VegetablesWithMeals'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - VegetablesWithMeals.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of MainDailyMeals
# | MainDailyMeals | 1<br>2<br>3 | 1 to 2 meals<br>3 meals<br>Over 3 meals |
####################################################################################################

# plt.clf ()
# obesity_df['MainDailyMeals'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2], labels=['1 to 2 Meals (1)', '3 Meals (2)', 'Over 3 Meals (3)'], rotation=0)
# plt.title ('Distribution of MainDailyMeals')
# plt.xlabel = 'MainDailyMeals'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - MainDailyMeals.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of SnackFrequency
# | SnackFrequency | 1<br>2<br>3<br>4 | No<br>Sometimes<br>Frequently<br>Always |
####################################################################################################

# plt.clf ()
# obesity_df['SnackFrequency'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2, 3], labels=['No (1)', 'Sometimes (2)', 'Frequently (3)', 'Always (4)'], rotation=0)
# plt.title ('Distribution of SnackFrequency')
# plt.xlabel = 'SnackFrequency'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - SnackFrequency.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of Smoke
# | Smoke | 1<br>0 | yes<br>no | yes<br>no |
####################################################################################################

# plt.clf ()
# obesity_df['Smoke'].value_counts ().sort_index (ascending=False).plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1], labels=['Yes (1)', 'No (0)'], rotation=0)
# plt.title ('Distribution of Smoke')
# plt.xlabel = 'Smoke'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - Smoke.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of DailyWater
# | DailyWater | 1<br>2<br>3 | Less than 1L<br>1L to 2L<br>Over 3L |
####################################################################################################

# plt.clf ()
# obesity_df['DailyWater'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2], labels=['Less than 1L (1)', '1L to 2L (2)', 'Over 3L (3)'], rotation=0)
# plt.title ('Distribution of DailyWater')
# plt.xlabel = 'DailyWater'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - DailyWater.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of MonitorCalories
# | MonitorCalories | 1<br>0 | yes<br>no |
####################################################################################################

# plt.clf ()
# obesity_df['MonitorCalories'].value_counts ().sort_index (ascending=False).plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1], labels=['Yes (1)', 'No (0)'], rotation=0)
# plt.title ('Distribution of MonitorCalories')
# plt.xlabel = 'MonitorCalories'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - MonitorCalories.png', dpi=96, bbox_inches='tight')

####################################################################################################
# Barplot of WeeklyActivity
# | WeeklyActivity | 0<br>1<br>2<br>3 | Never<br>1 or 2 days<br>2 or 3 days<br>4 or 5 days |
####################################################################################################

# plt.clf ()
# obesity_df['WeeklyActivity'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2, 3], labels=['Never (0)', '1 or 2 Days (1)', '2 or 3 Days (2)', '4 or 5 Days (3)'], rotation=0)
# plt.title ('Distribution of WeeklyActivity')
# plt.xlabel = 'WeeklyActivity'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - WeeklyActivity.png', dpi=96, bbox_inches='tight')


####################################################################################################
# | TechnologyUsage | 1<br>2<br>3 | 0 to 2 hours<br>3 to 5 hours<br>Over 5 hours |
####################################################################################################

# plt.clf ()
# obesity_df['TechnologyUsage'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2], labels=['Never (0)', '0 to 2 hours (1)', 'Over 5 hours (2)'], rotation=0)
# plt.title ('Distribution of TechnologyUsage')
# plt.xlabel = 'TechnologyUsage'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - TechnologyUsage.png', dpi=96, bbox_inches='tight')


####################################################################################################
# | AlcoholFrequency | 1<br>2<br>3<br>4 | No<br>Sometimes<br>Frequently<br>Always |
####################################################################################################

# plt.clf ()
# obesity_df['AlcoholFrequency'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2, 3], labels=['No (1)', 'Sometimes (2)', 'Frequently (3)', 'Always (4)'], rotation=0)
# plt.title ('Distribution of AlcoholFrequency')
# plt.xlabel = 'AlcoholFrequency'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - AlcoholFrequency.png', dpi=96, bbox_inches='tight')


####################################################################################################
# | Transportation | 1<br>2<br>3<br>4<br>5 | Automobile<br>Bike<br>Motorbike<br>Public Transportation<br>Walking |
####################################################################################################

# plt.clf ()
# obesity_df['Transportation'].value_counts ().sort_index ().plot (kind='bar', edgecolor='black')
# plt.xticks (ticks=[0, 1, 2, 3, 4], labels=['Automobile (1)', 'Bike (2)', 'Motorbike (3)', 'Public\nTransportation\n(4)', 'Walking (5)'], rotation=0)
# plt.title ('Distribution of Transportation')
# plt.xlabel = 'Transportation'
# plt.ylabel = 'Frequency'
# plt.savefig ('images\\5 - EDA\\Barplot - Transportation.png', dpi=96, bbox_inches='tight')




####################################################################################################
# Split Data into Train (80%) and Test (20%)
####################################################################################################

# test_size = .20

# x_train, x_test, y_train, y_test = train_test_split (obesity_predictors_df, obesity_response_df, test_size=test_size, shuffle=True, random_state=42)

# # print (f'df: {obesity_df.shape}')
# # print (f'x_train: {x_train.shape }')
# # print (f'y_train: {y_train.shape}')
# # print (f'x_test: {x_test.shape }')
# # print (f'y_test: {y_test.shape}')


####################################################################################################
# Create Multinomial Logistic Regression model
####################################################################################################

# mlr_model = LogisticRegression (solver='lbfgs', max_iter=5000000, random_state=0)
# mlr_model.fit (x_train, y_train)


####################################################################################################
# Create Classification Report and Accuracy Score
####################################################################################################

# y_predictions = mlr_model.predict (x_test)

# obesity_classification_report = classification_report (y_test, y_predictions, target_names=['Underweight', 'Normal', 'Overweight I', 'Overweight II', 'Obesity I', 'Obesity II', 'Obesity III'])
# print (f'\n{obesity_classification_report}')

# obesity_accuracy_score = accuracy_score (y_test, y_predictions)
# print (f'Accuracy Score: {(obesity_accuracy_score * 100):.2f}%')


####################################################################################################
# Create Confusion Matrix
####################################################################################################

# label_mapping = {1: 'Underweight', 2: 'Normal', 3: 'Overweight I', 4: 'Overweight II', 5: 'Obese I', 6: 'Obese II', 7: 'Obese III'}
# obesity_category_labels = ['Underweight', 'Normal', 'Overweight I', 'Overweight II', 'Obese I', 'Obese II', 'Obese III']

# y_test_strings = [label_mapping[label] for label in y_test]
# y_prediction_strings = [label_mapping[label] for label in y_predictions]

# mlr_confusion_matrix = confusion_matrix (y_true=y_test_strings, y_pred=y_prediction_strings, labels=obesity_category_labels)
# # print (f'Confusion Matrix:\n{mlr_confusion_matrix}')

# display = ConfusionMatrixDisplay (confusion_matrix=mlr_confusion_matrix, display_labels=obesity_category_labels)
# display.plot ()
# plt.xticks (rotation=270)
# plt.xlabel = 'Predicted'
# plt.ylabel = 'True'
# plt.savefig ('images\\6 - Model\\MLR - Confusion Matrix - Baseline.png', dpi=96, bbox_inches='tight')




# ####################################################################################################
# # Calculate VIF
# ####################################################################################################

# vif_data = pd.DataFrame ()
# vif_data["Feature"] = obesity_predictors_df.columns
# vif_data["VIF"] = [variance_inflation_factor (obesity_predictors_df.values, i) for i in range (obesity_predictors_df.shape[1])]

# dfi.export (vif_data, 'images\\6 - Model\\MLR - VIF Table - Baseline.png')

# print (vif_data)


# ####################################################################################################
# # Correlation Matrix / Heatmap of predictor variables
# ####################################################################################################

# correlation_matrix = obesity_predictors_df.corr ()
# # print (correlation_matrix)

# plt.clf ()
# plt.figure (figsize=(10, 10))
# axes = plt.axes ()
# sns.heatmap (data=correlation_matrix, ax=axes, cmap='coolwarm', annot=True, annot_kws={'size' : 8}, fmt='.2f', square=True, linewidths=.5, linecolor='white', cbar=True)
# axes.xaxis.tick_top ()
# axes.yaxis.tick_left ()
# axes.tick_params (axis='x', rotation=90, labelsize=8)
# axes.tick_params (axis='y', rotation=0, labelsize=8)
# axes.set_title ('Correlation Matrix for Obesity Features', fontsize=24)
# # plt.show ()
# plt.savefig ('images\\6 - Model\\Correlation Matrix - Features - Baseline.png', dpi=96, bbox_inches='tight')



# ####################################################################################################
# # Remove predictor variables to test for multicollinearity
# ####################################################################################################

# # Remove ObesityCategory, Height, and Weight as predictors of BMI

# # obesity_predictors_df = obesity_predictors_df.drop (['Height'], axis=1)
# # obesity_predictors_df = obesity_predictors_df.drop (['Weight'], axis=1)

# # Remove SnackFrequence (18.69)- likely associated with HighCalorieFoods or MainDailyMeals

# # obesity_predictors_df = obesity_predictors_df.drop (['SnackFrequency'], axis=1)

# # # Remove Age (13.58)- likely associated with other lifestyle or health predictors

# # obesity_predictors_df = obesity_predictors_df.drop (['Age'], axis=1)

# # Remove MainDailyMeals (10.26)- likely associated with HighCaloriefoods, VegetablesWithMeals, or DailyWater

# # obesity_predictors_df = obesity_predictors_df.drop (['MainDailyMeals'], axis=1)

# # # Remove AlcoholFrequency (10.35)- likely associated with HighCaloriefoods, VegetablesWithMeals, or DailyWater

# # obesity_predictors_df = obesity_predictors_df.drop (['AlcoholFrequency'], axis=1)

# # Remove WeeklyActivity and Transportation - Low predictive impact on model based on coefficients

# # obesity_predictors_df = obesity_predictors_df.drop (['WeeklyActivity'], axis=1)
# # obesity_predictors_df = obesity_predictors_df.drop (['Transportation'], axis=1)




# ####################################################################################################
# # Interaction Variables
# ####################################################################################################

# # Add HighCalorieFoods x SnackFrequency interaction variable

# # obesity_predictors_df['HighCalorieSnack'] = obesity_predictors_df['HighCalorieFoods'] * obesity_predictors_df['SnackFrequency']
# # obesity_predictors_df = obesity_predictors_df.drop (['HighCalorieFoods'], axis=1)
# # obesity_predictors_df = obesity_predictors_df.drop (['SnackFrequency'], axis=1)

# # Add HighCalorieFoods x DailyWater interaction variable

# # obesity_predictors_df['HighCalorieWater'] = obesity_predictors_df['HighCalorieFoods'] * obesity_predictors_df['DailyWater']
# # obesity_predictors_df = obesity_predictors_df.drop (['HighCalorieFoods'], axis=1)
# # obesity_predictors_df = obesity_predictors_df.drop (['DailyWater'], axis=1)







# ####################################################################################################
# # Plot Count and Proportions of Obesity Category to Validate That Imbalance Is Addressed
# ####################################################################################################

# # obesity_counts      = obesity_df['ObesityCateogry'].value_counts ().sort_index ()
# # obesity_proportions = obesity_df['ObesityCateogry'].value_counts (normalize=True).sort_index ()
# # print (f'Obesity Category Counts:\n{obesity_counts}')
# # print (f'Obesity Category Proportions:\n{obesity_proportions}')

# # plt.clf ()
# # obesity_counts.plot (kind='bar', title='Obesity Category Counts', xlabel='')
# # # obesity_counts.plot (kind='bar', title='Obesity Category Counts', xlabel='', figsize=(1656/300, 1611/300))
# # obesity_category_labels = {'Underweight (1)', 'Normal (2)', 'Overweight I (3)', 'Overweight II (4)', 'Obesity I (5)', 'Obesity II (6)', 'Obesity III (7)'}
# # plt.xticks (ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Underweight (1)', 'Normal (2)', 'Overweight I (3)', 'Overweight II (4)', 'Obesity I (5)', 'Obesity II (6)', 'Obesity III (7)'])
# # plt.savefig ('images\obesity_category_bar.png', dpi=96, bbox_inches='tight')

# # majority_count = obesity_counts.max ()
# # minority_count = obesity_counts.min ()

# # imbalance_ratio = majority_count / minority_count
# # print (f'Imbalance Ratio: {imbalance_ratio}')

















# ####################################################################################################
# # Create Coefficient Heatmap
# ####################################################################################################

# # obesity_coefficients = pd.DataFrame ({
# #     'Feature': obesity_predictors_df.columns,
# #     'Underweight': mlr_model.coef_[0],
# #     'Normal': mlr_model.coef_[1],
# #     'Overweight I': mlr_model.coef_[2],
# #     'Overweight II': mlr_model.coef_[3],
# #     'Obesity I': mlr_model.coef_[4],
# #     'Obesity II': mlr_model.coef_[5],
# #     'Obesity III': mlr_model.coef_[6]
# # })

# # obesity_coefficients_df = obesity_coefficients.set_index ('Feature')
# # sns.heatmap (obesity_coefficients_df, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# # plt.title ("Logistic Regression Coefficients Heatmap")
# # plt.xlabel ("Feature")
# # plt.ylabel ("ObesityCateogry")
# # plt.savefig ('heatmap_lr_coefficients.png', dpi=96, bbox_inches='tight')

# # intercept = LogReg.intercept_
# # coefficients = LogReg.coef_
# # print (f'Intercept: {intercept}')
# # print (f'coefficients:\n{coefficients}')


# ####################################################################################################
# # Create Confusion Matrix
# ####################################################################################################

# # obesity_confusion_matrix = confusion_matrix (y_test, y_predictions)
# # print (f'Confusion Matrix:\n{obesity_confusion_matrix}')

# # cm = confusion_matrix (y_true=y_test, y_pred=y_predictions, labels=['Underweight', 'Normal', 'Overweight I', 'Overweight II', 'Obesity I', 'Obesity II', 'Obesity III'])
# # display = ConfusionMatrixDisplay (confusion_matrix=cm)
# # display.plot ()
# # plt.savefig ('confusion.png', dpi=96, bbox_inches='tight')














# ####################################################################################################
# ####################################################################################################
# # Random Forest
# ####################################################################################################
# ####################################################################################################

####################################################################################################
# Split Data into Train (80%) and Test (20%)
####################################################################################################

test_size = .20

x_train, x_test, y_train, y_test = train_test_split (obesity_predictors_df, obesity_response_df, test_size=test_size, shuffle=True, random_state=42)

# print (f'df: {obesity_df.shape}')
print (f'x_train: {x_train.shape }')
print (f'y_train: {y_train.shape}')
print (f'x_test: {x_test.shape }')
print (f'y_test: {y_test.shape}')
# print (f'Columns: {x_test.columns}')

# ####################################################################################################
# # Random Forest Classifier
# ####################################################################################################

# Starting Accuracy: 81.80%
# rf_model = RandomForestClassifier (random_state=42) # 80.85
# rf_model = RandomForestClassifier (n_estimators=50, random_state=42) # 80.38
# rf_model = RandomForestClassifier (n_estimators=50, max_depth=None, random_state=42)  # None --> 80.38
# rf_model = RandomForestClassifier (n_estimators=50, max_depth=None, min_samples_split=2, random_state=42)  # 2 --> 80.38
# rf_model = RandomForestClassifier (n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)  # 1 --> 80.38
rf_model = RandomForestClassifier (n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=4, random_state=42)  # 8 --> 81.80

rf_model.fit (x_train, y_train)

y_predictions = rf_model.predict (x_test)


# ####################################################################################################
# # Display Classification Report and Accuracy Score
# ####################################################################################################

print (f'Classification_report:\n\n{classification_report (y_test, y_predictions)}')

rf_accuracy_score = accuracy_score (y_test, y_predictions)
print (f'Accuracy Score: {(rf_accuracy_score * 100):.2f}%')


####################################################################################################
# GridSearchCV Estimator
####################################################################################################

# parameter_grid = {
#     'n_estimators'     : [45, 50, 55],
#     'max_depth'        : [None, 2, 5],
#     'min_samples_split': [2, 3, 4],
#     'min_samples_leaf' : [1, 2, 3],
#     'max_features'     : ['sqrt', 4, 6]
# }

# rf_model = RandomForestClassifier (random_state=42)

# grid_search = GridSearchCV (estimator=rf_model, param_grid=parameter_grid, scoring='accuracy')
# grid_search.fit (x_train, y_train)

# print (f'Best parameters:\n{grid_search.best_params_}')

# Result: 
# Best parameters:
# n_estimators=50, max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=6
# {'max_depth': None, 'max_features': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 50}
# Accuracy Score: 81.09%


####################################################################################################
# Show average depth of trees and max_depth of all trees
####################################################################################################

# max_depth = list ()
# for tree in rf_model.estimators_:
#     max_depth.append (tree.tree_.max_depth)

# print (f'Average max_depth: {(sum(max_depth) / len(max_depth)):0.2f}')
# print (f'max_depth {max_depth}')


####################################################################################################
# Display Feature Importance
####################################################################################################

feature_importance = pd.DataFrame ({
    'Feature': x_train.columns,
    'Importance': rf_model.feature_importances_
})
# }).sort_values (by='Importance', ascending=False)

plt.clf ()
feature_importance.plot (kind='bar', edgecolor='black', legend=False, ylabel='Gini Importance', title='Feature Importance Rankings')
plt.xticks (ticks=list (range (len (x_test.columns))), labels=x_test.columns, rotation=270)
# plt.show ()
plt.savefig ('images\\6 - Model\\Random Forest - Feature Importance - Barplot.png', dpi=96, bbox_inches='tight')

print ('Figure: Feature Importance Rankings\n')

print (f'Feature Importance (Sorted):\n{feature_importance.sort_values (by='Importance', ascending=False)}')


# ###################################################################################################
# Remove Predictor Variables for Random Forest Testing based on Feature Importance
# ###################################################################################################

obesity_predictors_test_df = obesity_predictors_df

# Starting Accuracy: 81.80%                                                                    Individual   Combined
obesity_predictors_test_df = obesity_predictors_test_df.drop (['Smoke'], axis=1)               # 80.85
obesity_predictors_test_df = obesity_predictors_test_df.drop (['MonitorCalories'], axis=1)     # 80.61      81.56
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['HighCalorieFoods'], axis=1)    # 78.49      76.83
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['TechnologyUsage'], axis=1)     # 79.43      75.89
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['Transportation'], axis=1)      # 78.49      74.00
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['DailyWater'], axis=1)          # 79.43      74.23
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['SnackFrequency'], axis=1)      # 76.60      70.21
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['FamilyHistory'], axis=1)       # 76.83      67.38
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['MainDailyMeals'], axis=1)      # 80.85      62.88
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['AlcoholFrequency'], axis=1)    # 76.60      56.50
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['VegetablesWithMeals'], axis=1) # 81.32      53.19
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['Gender'], axis=1)              # 77.87      47.52
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['WeeklyActivity'], axis=1)      # 78.49      39.48
# obesity_predictors_test_df = obesity_predictors_test_df.drop (['Age'], axis=1)                 # 72.58

x_train, x_test, y_train, y_test = train_test_split (obesity_predictors_test_df, obesity_response_df, test_size=0.20, shuffle=True, random_state=42)
rf_model = RandomForestClassifier (n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=4, random_state=42)
rf_model.fit (x_train, y_train)
y_predictions = rf_model.predict (x_test)

####################################################################################################
# Display Classification Report and Accuracy Score
####################################################################################################

print (f'Classification Report - Random Forest - Feature Importance:\n\n{classification_report (y_test, y_predictions)}')

rf_accuracy_score = accuracy_score (y_test, y_predictions)
print (f'Accuracy Score: {(rf_accuracy_score * 100):.2f}%')



# ###################################################################################################
# Create Confusion Matrix
# ###################################################################################################

label_mapping = {1: 'Underweight', 2: 'Normal', 3: 'Overweight I', 4: 'Overweight II', 5: 'Obese I', 6: 'Obese II', 7: 'Obese III'}
obesity_category_labels = ['Underweight', 'Normal', 'Overweight I', 'Overweight II', 'Obese I', 'Obese II', 'Obese III']

y_test_strings = [label_mapping[label] for label in y_test]
y_prediction_strings = [label_mapping[label] for label in y_predictions]

rf_confusion_matrix = confusion_matrix (y_true=y_test_strings, y_pred=y_prediction_strings, labels=obesity_category_labels)
# print (f'Confusion Matrix:\n{rf_confusion_matrix}')

display = ConfusionMatrixDisplay (confusion_matrix=rf_confusion_matrix, display_labels=obesity_category_labels)
display.plot ()
plt.xticks (rotation=270)
plt.xlabel = 'Predicted'
plt.ylabel = 'True'
plt.savefig ('images\\6 - Model\\Random Forest - Confusion Matrix - Final.png', dpi=96, bbox_inches='tight')