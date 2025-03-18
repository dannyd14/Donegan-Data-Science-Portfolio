# -*- coding: utf-8 -*-


#Cleaning and Melting cell

import pandas as pd


fed = pd.read_csv("TidyData-Project/fed_rd_year&gdp.csv")

fed.head()



# Melt the DataFrame to gather year columns
melted_fed = pd.melt(fed, id_vars=['department'], var_name='Year_GDP', value_name = 'Spending')

# Split the 'Year_GDP' column on '_' and expand into multiple columns
split_cols = melted_fed['Year_GDP'].str.split('_', expand=True)

# Assign new columns from the split data, converting to numeric so that these values are evaluated as numbers
melted_fed['Year'] = pd.to_numeric(split_cols[0])   # First part = Year
melted_fed['GDP'] = pd.to_numeric(split_cols[1].str.replace('gdp', ''))  # Second part = GDP value

# Reorder columns to place 'department ' first, followed by 'Year' and the remaining columns
melted_fed = melted_fed[['department', 'Year' , 'Spending', 'GDP']]




#Sort the values of by department and then year so that the view is more organized/easier to read

melted_fed = melted_fed.sort_values(by=['department', 'Year']).dropna()

melted_fed.isnull().sum()

# Display the cleaned DataFrame
melted_fed.head(100)

"""To start, this code cell begins by reading in the csv file as a pandas data frame, after we had imported the pandas package as pd. This data frame is named fed, and will be used until the melted data frame has been created. Next, I use the pandas melt function in order to change the set up of the initial data frame, and also pull data that was previously used as the column names for the data frame. After melting, there was a column named "Year_GDP" which needed to be split by the _, and to do this, I used the str.split() function that split this data into two different columns. After this, I made a few simple changes, where the year and GDP columns were casted as numeric variable types, so that they could be evaluated as numbers. Lastly, I rearranged the columns so that they could be sorted in a more effective way, where they were sorted by department and then year, while I also dropped all null observations to make sure that all the data had values and was tidy. To achieve this, I used changed the order of the columns by indexing, used the drop.na() function to remove null values, and then displayed the data frame in order to make sure that the proper adjustments had been made. I also used the .isnull().sum() functions to confirm that all null values had been removed. After these changes were made, I was able to confirm that each observation had its own row and each variable had its own column, two key components of tiny data. With this, I could confirm that each observational unit had its own cell, or in other words, each value is in its own cell. After I was able to ensure that I had tidy data, I could begin to create some visualizations and examine trends to make conclusions about the data.

"""

#visualization cells

import matplotlib.pyplot as plt
import seaborn as sns


#creates a pivot table organized by year, where the columns become the department, makes it easier to create a plot showing the trends in each department's funding values over time
pivot_fed = melted_fed.pivot(index='Year', columns='department', values='Spending')



#The following code creates this plot, iterating through each column and then plotting the year as the x value, and the spending as the y value, creating line graphs for each different department
plt.figure(figsize=(8, 5))
for column in pivot_fed.columns:
    plt.plot(pivot_fed.index, pivot_fed[column], label=column)

# Add labels and legend
plt.title('Department-Wise Trends')
plt.xlabel('Year')
plt.ylabel('Departmental Spending')
plt.legend(title='Department')
plt.grid(True)

# Show the plot

plt.show()
#Show the pivot table
pivot_fed.head(20)

"""This code cell shows how one of the data visualizations was created. To begin, I imported matplotlib as this would be the library that I would use. Next, since I had intended to make a line plot for the spending of each department over the years, I created a pivot table that would split each department as columns and each observation as a year to make this graph easier. Next, I took advantage of the plot function, iterating through each column and plotting each year along with its spending value as a different point. For each different department, every year was plotted as the x value while the spending for that year was plotted as the y value. I then used different plt functions in order to add labels for the x and y axes, as well as a title and a legend so that a viewer could tell which color represented which department. From this graph, one could learn which departments spend the most(DOD, NIH and HHS), as well as which departments saw the most change of the years(same). While I displayed the graph, I also displayed the pivot table as well to confirm that the data had been organized how I had intended."""



#Agg func pivot table, average yearly spending by department

#Aggregates the spending for each department across all years by their mean value, and then sorts the departmental spending values from highest to lowest

pivot_fund = melted_fed.pivot_table(index='department', values ='Spending', aggfunc='mean').sort_values(by='Spending', ascending=False)
pivot_fund

"""For the above cell, I was hoping to create a pivot table that would aggregate the spending for each department over all the years and displayed the mean spending per year, and then sorted these values from highest to lowest. To accomplish this, I used the pivot_table() function, where the department was the index, the spending was the values that were used, and the aggregate function was the mean. Next, I used the sort_values() function in order to sort them, and set ascending to false so that the values would be sorted from highest to lowest. Next, I displayed the entire pivot table in order to see the results."""





#Bar graph of average spending per year by the department
#This line groups the data by department, and then aggregates the code based on the spending values over all the years

average_spending = melted_fed.groupby('department')['Spending'].mean()
#Creates a color pallette
colors = sns.color_palette("Set2", len(average_spending))
# Plots the data
plt.figure(figsize=(8, 5))
average_spending.plot(kind='bar', color = colors)

# Add labels and title
plt.title('Average Spending by Department')
plt.xlabel('Category')
plt.ylabel('Average Spending')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()

"""The above code cell was intended to create a bar graph, that shows which department has the highest spending on average from year to year. This serves a similar function to the pivot table that was created above, but creates a visualization to go along with it. In order to create this bar graph, I first grouped the data, using a mean aggregate function. I then graphed this grouped series using the .plot() function where the type was bar graph and the colors were created as a seaborn color pallete. (Using the sns.color_pallete() function) I then used a varity of plt.() functions in order to add labels for both axes, a title, angle the x labels, and create a grid and then display the graph. Similar to the pivot table above, a viewer could learn a lot about which departments usually spend the most."""



#GDP Trends Over the Years

#Creates a scatter plot of the GDP over all years, shows growth over the years
plt.scatter(melted_fed['Year'], melted_fed['GDP'], marker='o', color = 'green')

# Set the labels and title
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Progression Over The Years')
plt.grid(True, linestyle='--', alpha=0.5)

# Display the plot
plt.show()

"""For my third visualization, I wanted to display how the GDP of the United States has grown over all the years, and to do this, I created a line plot between the year and GDP value. To accomplish this, I used a scatter plot where the x axis was the "Year" column of our melted data frame, and for the y axis I used the "GDP" column of the same data frame. Using th plt.scatter() function, I was able to display this relationship and see a gradual growth of the GDP as the years progressed, and was also able to see which stretches of years had the strongest growth vs the stretches that showed slower growth. Similar to other visualizations, I used the plt.() functions to add labels and a title."""

#Pie chart
#Sums the total spending for each group based on the department
total_spending = melted_fed.groupby('department')['Spending'].sum()
# Calculate the total spending


# Plot the pie chart
fig, ax = plt.subplots()
wedges, texts,  = ax.pie(total_spending, labels=total_spending.index, startangle=90, labeldistance = 1.1)
#Adjust the texts on the outside of the pie chart
for text in texts:
    text.set_rotation(75)
    text.set_fontsize(8)
    text.set_color('black')

#Creates a title
plt.title('Total Contributions to Departmental Spending Across All Years')
# Display the pie chart
plt.show()

"""For my final visualization, I created a pie chart that showed how each department contributed to the total sum of all spending from our data. In order to do this, I used the same groupby() function as for the bar graph, but used the sum aggregation function rather than the mean. I then plotted the pie chart in a slightly differnet way, as I needed to edit the way that the label texts were displayed so that they would be easier to read. However, this was still using the matplotlib library, just through the plt.subplots() function. And, similar to above, I created a title and displayed the visualization using the plt.() functions.

"""