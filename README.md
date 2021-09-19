# Salary-Prediction
#final project
# Import Libraries for Analysis
import numpy as np
import pandas as pd

# Import Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for train test split
from sklearn.model_selection import train_test_split

# import Ilbrary for Scaling
from sklearn.preprocessing import StandardScaler

# import Ilbrary for Model Building
from sklearn.linear_model import LinearRegression

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the data set
salary_org = pd.read_csv("train.csv")

# Print top 5 rows of dataset
salary_org.head()
 
# Check the info of data set
salary_org.info()

# Create a copy of data set
salary_cpy = salary_org.copy()

# Check column names
salary_cpy.columns

# Remove leading and trailing edges
salary_cpy.columns = salary_cpy.columns.str.strip()

# print coloumns after stripping spaces
print("After Removing leading and trailing spaces ",  salary_cpy.columns)

# Check Null Values in data set
salary_cpy.isnull().sum()

# Remove rows having empty hire date
salary_cpy = salary_cpy.dropna(subset=['HireDate'])

# Check null Values
salary_cpy.isnull().sum()

# Drop Gross Pay column
salary_cpy=salary_cpy.drop('GrossPay',axis=1)

# Check null Values
salary_cpy.isnull().sum()

# Value_counts for AgencyID
salary_cpy.AgencyID.value_counts()

# Value_counts for Agency
salary_cpy.Agency.value_counts()

# Value_counts for JobTitle
salary_cpy.JobTitle.value_counts()

#Value counts on HireDate
salary_cpy.HireDate.value_counts()

#Value counts on Annual Salary
salary_cpy.AnnualSalary.value_counts()

# Removing $ from Annual Salary and converting it into Integer format
salary_cpy['AnnualSalary'] = salary_cpy['AnnualSalary'].apply(lambda x : (float)(str(x)[1:]))

salary_cpy['HireDay'] = salary_cpy['HireDate'].apply(lambda x : (int)(str(x[3:5])))
salary_cpy['HireMonth'] = salary_cpy['HireDate'].apply(lambda x : (int)(str(x[0:2])))
salary_cpy['HireYear'] = salary_cpy['HireDate'].apply(lambda x : (int)(str(x[6:])))

# Print info to check whether columns are added
salary_cpy.info()

# Trim spaces
salary_cpy['JobTitle'] = salary_cpy['JobTitle'].apply(lambda x : str(x).strip().replace("  "," "))
salary_cpy['AgencyID'] = salary_cpy['AgencyID'].apply(lambda x : str(x).strip().replace("  "," "))
salary_cpy['Agency'] = salary_cpy['Agency'].apply(lambda x : str(x).strip().replace("  "," "))

# Trim spaces
salary_cpy['JobTitle'] = salary_cpy['JobTitle'].apply(lambda x : str(x).upper())
salary_cpy['AgencyID'] = salary_cpy['AgencyID'].apply(lambda x : str(x).upper())
salary_cpy['Agency'] = salary_cpy['Agency'].apply(lambda x : str(x).upper())

# Create Box Plot for Annual Salary
salary_cpy.AnnualSalary.plot.box()
plt.show()

# SUMMARY STATS OF AnnualSalary
salary_cpy.AnnualSalary.describe()

salary_cpy = salary_cpy[salary_cpy['AnnualSalary']<150000]
salary_cpy.shape

# Check distribution of Target Variable
sns.distplot(salary_cpy.AnnualSalary)
plt.title("Annual Salary Distribution Plot",fontsize=15)
plt.show()

#Top 10 Jobs that based on hirings
plt.figure(figsize=(10,5))
salary_cpy.groupby(['JobTitle'])['Name'].count().sort_values(ascending=False).head(10).plot.bar()
plt.ylabel('No of people Working')
plt.title("Top 10 Jobs for which Hiring is Highest",fontsize=20)
plt.show()

#Top 10 Jobs that fetche highest Salary
plt.figure(figsize=(10,5))
salary_cpy.groupby(['JobTitle'])['AnnualSalary'].mean().sort_values(ascending=False).head(10).plot.bar()
plt.ylabel('Avg Salary')
plt.title("Top 10 Highest Paying Jobs",fontsize=20)
plt.show()

# Find mean Slary
mean_sal = salary_cpy.AnnualSalary.mean()

# Number of Jobs paying more than mean salary
good_pay_jobs = salary_cpy.groupby(['JobTitle'])['AnnualSalary'].mean().reset_index()
good_pay_jobs[good_pay_jobs.AnnualSalary>mean_sal]['JobTitle'].count()

#Top 10 Agencies that has highest number of employees
plt.figure(figsize=(10,5))
salary_cpy.groupby(['Agency'])['Name'].count().sort_values(ascending=False).head(10).plot.bar()
plt.ylabel('No Of Employees')
plt.title("Top Agencies with Highest number of Employees",fontsize=18)
plt.show()

#Top 10 Jobs that has highest number of employees
plt.figure(figsize=(10,5))
salary_cpy.groupby(['AgencyID'])['Name'].count().sort_values(ascending=False).head(10).plot.bar()
plt.ylabel('No Of Employees')
plt.title("Top AgencyID's with Highest number of Employees",fontsize=18)
plt.show()

# Salary vs Hire Year
plt.figure(figsize=(10,5))
salary_cpy.groupby(['HireYear'])['AnnualSalary'].mean().sort_values().head(10).plot.bar()
plt.ylabel(' AverageSalary')
plt.title("Average Salary of Employees based on Hire Year",fontsize=18)
plt.show()

# Checking if Month hired has any such effect
plt.figure(figsize=(10,5))
salary_cpy.groupby(['HireMonth'])['AnnualSalary'].mean().plot.bar()
plt.ylabel('Average Salary')
plt.title("Average Salary of Employees based on Hire Month",fontsize=18)
plt.show()

# Checking on which Month most people are hired
plt.figure(figsize=(10,5))
salary_cpy.groupby(['HireMonth'])['Name'].count().plot.bar()
plt.ylabel('Numer of Hired')
plt.title("Employers Hirings based on Hire Year",fontsize=20)
plt.show()

# Plot a pair plot
plt.figure(figsize=(15,20))
sns.pairplot(salary_cpy)
plt.show()

# Plot a heatMap
plt.figure(figsize=(10,5))
sns.heatmap(salary_cpy.corr(),annot=True)

# Create a copy od data frame
salary_master = salary_cpy.copy()

# Apply mean encoding for Job Title
mean_Job = salary_master.groupby('JobTitle')['AnnualSalary'].mean()
salary_master['JobTitle'] = salary_master['JobTitle'].map(mean_Job)

print(salary_master['JobTitle'])

# Apply mean encoding for Agency
mean_agency = salary_master.groupby('Agency')['AnnualSalary'].mean()
salary_master['Agency'] = salary_master['Agency'].map(mean_agency)

print(salary_master['Agency'])

# Apply mean encoding for AgencyID
mean_agencyID = salary_master.groupby('AgencyID')['AnnualSalary'].mean()
salary_master['AgencyID'] = salary_master['AgencyID'].map(mean_agencyID)

print(salary_master['AgencyID'])

# Check info
salary_master.info()

# Drop Name, HireDate column
salary_master = salary_master.drop(['HireDate','Name'],axis=1)

# Check Info
salary_master.info()

# Split data into train and test sets
salary_train, salary_test = train_test_split(salary_master,train_size=0.7, random_state=42)

# Shape of train set
print(salary_train.shape)

# Shape of test set
print(salary_test.shape)

# Divide tarin set into Dependent and independent variables
y_train = salary_train.pop('AnnualSalary')

X_train = salary_train


# Divide test set into Dependent and independent variables
y_test = salary_test.pop('AnnualSalary')

X_test = salary_test

# Scale the train
scaler = StandardScaler()

X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_train.describe()

# # Scale the test
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

X_test.describe()

# A sample Data Frame
example_df = pd.DataFrame({'Performance' : [1,2,3,4,5,6], 'Grade':[2,5,4,6,1,3],'Target':[100,300,200,600,200,600]})
example_df

# Create a model by creating a Linear Regression Object
example_lr = LinearRegression()

example_model = example_lr.fit(example_df[['Performance','Grade']],example_df['Target'])

# Lets have a look at coefficients as described
print(example_lr.coef_)
print(example_lr.intercept_)

# Calculate R-squared
example_model.score(example_df[['Performance','Grade']],example_df['Target'])

# Plot Distribution plot of Residuals
plt.figure(figsize=(10,5))
target_pred = example_model.predict(example_df[['Performance','Grade']])
example_res = example_df['Target'] - target_pred
sns.distplot(example_res)
plt.xlabel('example_res')
plt.title("Residual Analysis",fontsize=20)
plt.show()

sns.scatterplot(x=example_res,y=target_pred)
plt.xlabel('Residuals')
plt.title("Residual Analysis",fontsize=20)
plt.show()

# Build the model
lr = LinearRegression()

salary_reg = lr.fit(X_train,y_train)

# Verify the r2 score
salary_reg.score(X_train,y_train)

# r2 for test data
salary_reg.score(X_test,y_test)

# Plot Distribution plot of Residuals
plt.figure(figsize=(10,5))
y_train_pred = salary_reg.predict(X_train)
res = y_train - y_train_pred
sns.distplot(res)
plt.xlabel('Residuals')
plt.title("Residual Analysis",fontsize=20)
plt.show()

sns.scatterplot(x=res,y=y_train_pred)
plt.xlabel('Residuals')
plt.title("Residual Analysis",fontsize=20)
plt.show()

# Print coef
print("Coef are:",salary_reg.coef_)

#print intercept
print("Intercept is",salary_reg.intercept_)

model = str(salary_reg.intercept_)

for i in range(len(salary_reg.coef_)):
    model = model +' + '  +(str(salary_reg.coef_[i])) + ' * ' +(str(X_train.columns[i]))
print(model)
