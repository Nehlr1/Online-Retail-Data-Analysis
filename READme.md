# Online Retail Data Analysis

This project focuses on performing data analysis and building a machine learning model on TATA: Online Retail Dataset. The dataset contains information about various transactions, including the customer's country.

## Dataset Location
```
https://www.kaggle.com/datasets/ishanshrivastava28/tata-online-retail-dataset
```

## Dataset Description

The dataset consists of the following columns:
- InvoiceNo: The invoice number of the transaction.
- StockCode: The stock code of the purchased item.
- Description: The description of the purchased item.
- Quantity: The quantity of items purchased.
- InvoiceDate: The date and time of the transaction.
- UnitPrice: The unit price of the item.
- CustomerID: The unique identifier of the customer.
- Country: The country where the transaction took place.

## Data Preprocessing

The data preprocessing steps performed on the dataset include:
- Handling duplicates: Duplicated rows in the dataset are removed.
- Handling missing values: Null values in the "Description" column are replaced with "No Description". Rows with missing "CustomerID" are dropped.
- Encoding categorical variables: The "StockCode" column is encoded using dummy coding.
- Feature scaling: The numerical columns "Quantity" and "UnitPrice" are scaled using the StandardScaler.
- Dimensionality reduction: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset.

## Exploratory Data Analysis (EDA)

The EDA section includes:
- Statistical analysis: Descriptive statistics of the dataset are provided.
- Boxplots: Boxplots of the "UnitPrice" and "Quantity" variables are visualized.

## Machine Learning Model

A Random Forest Classifier (made from scratch) is implemented for predicting the "Country" based on the available features. The steps involved in the model building process are:
- Data sampling: A subset of 20,000 data points is randomly selected for model training.
- Feature encoding and scaling: Categorical variables are encoded, and numerical columns are scaled using the StandardScaler.
- Dimensionality reduction: PCA is applied to reduce the dimensionality of the feature matrix.
- Splitting the dataset: The dataset is split into training and testing sets.
- Model training and evaluation: The Random Forest Classifier is trained on the training set and evaluated using accuracy, confusion matrix, and classification report metrics.

## Results Visualization

The results of the trained model are visualized through the following plots:
- Confusion matrix: A heatmap of the confusion matrix is displayed.
- Classification report metrics: Precision, recall, and F1-score for each class are plotted.

## Requirements

The following Python libraries are required to run the code:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Install the required libraries using requirements.txt file.
```
pip install -r requirements.txt
```
2. Download the "Online Retail Data Set.csv" dataset and place it in the same directory as the code.
3. Run the code to perform data analysis and build the machine learning model.