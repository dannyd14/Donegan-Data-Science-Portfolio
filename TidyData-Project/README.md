## TidyData Project
This project will demonstrate how to tidy your data using organizing and cleaning techniques, while also demonstrating why having tidy data is necessary to evaluate and use a dataset.

## Project Overview

Ultimately, the goal of this project is to take a messy dataset and clean it into a tidy format, based closely on the principles described by Hadley Wickman in his ˆTidy Dataˆ framework. With this, we are hoping to attain the three main principles of tidy data, where each observation has its own row, each variable has its own column, and each type of observational unit forms a table. After accomplishing this and achieving this type of data, one can then begin to analyze, observe, and model the data effectively. 

## Dataset Description
For this project, I cleaned a dataset with information about the Federal Research and Development spending across different department from 1976-2019. With this dataset, there were 4 columns: the Department, Year, Spending, and GDP, and each observation included information for each department in each year that data was present. Some potential issues appeared prior to cleaning though, where the year and GDP values were originally in the column name, and there were some null values included in the data, as information for some departments did not date all the way back to 1976. This project cleans up these issues and tidys the data so that it is more applicable.

## Instructions
1. Import the pandas library in Python and assign it as pd
2. Import the pyplot functions in the matplotlib package and assign it as plt
3. Import the seaborn visualization library and assign it as sns
4. Load the CSV file of the dataset using the relative path for the CSV
5. Refer to [Data Cleaning Notebook](FedRD_DataCleaning_Project.ipynb) for the full code

## References

Here are some helpful links for tidying data
[Pandas Cheat Sheet](Pandas_Cheat_Sheet.pdf)
[Hadley Wickman Tidy Data](tidy-data.pdf)

## Cleaned Visualizations


