## Supervised Machine Learning Streamlit Exploratory Application
This project will be an example of how to create an exploratory Machine Learning app using streamlit and its capabilities. While navigating this application, a user will be able to explore how different machine learning models operate and perform, either on their own data or with provided sample data.

## Project Overview
After choosing to either import their own data or use sample data, within this application, the user has the ability to choose a variety of machine learning models and train these models on the data. The user can choose which type of model they are looking to develop, either regression or classification. Next, the user is presented with the option to adjust the relevant hyperparamters for each type of machine learning model. At this point, they would train the models, and then are able to view the results of how the model performed at its intended task. The goal of this application is to allow a user to learn more about different types of machine learning models, and also explore how adjusting the hyperparamters of these models can change their performance.

## App Features
Within this app, users have the ability to test out a variety of differen machine learning models, either on their own dataset, or using a sample dataset. This exploration could include choosing a target variable, training a model, viewing how the model performed by examining peformance statistics and visualizations that describe how the model performs. Model types include linear regression, logistic regression, decision tree, as well as k-nearest neighbors. Also, the user has the ability to adjust the hyperparamters that drive these models, making the application even more customizable. Each model has its own set of hyperparameters, and they are shown as a slider within this app. After training the model, the user can click through different tabs of the results and visualizations, with descriptions of the different metrics given. 




## Instructions for App Usage - [Link to Deployed Version](https://dannyd14-donegan-data-scienc-mlstreamlitappstreamlit-app-v3tugj.streamlit.app/)
1. Open the either through the Streamlit Community Cloud or via a URL. If using the app locally, type 'streamlit run MLStreamlitApp/Streamlit-App.py' in the terminal of the Streamlit-App.py file. If running locally, necessary libraries include: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn
2. Choose whether you would like to insert your own data set, or view a sample
3. Considering the type of data being used, pick either Classification or Regression
4. Select your target column that will be used for this model
5. Choose your model type
6. Adjust the hyperparamters using sliding scalers
7. Train the model
8. Observe the results and evaluate which performs the best


## References

Here is some more helpful information about Machine Learning

[Machine Learning Basics](MLbasics.pdf)

## Example of Application
![](<Screen Shot 2025-04-14 at 9.56.19 PM.png>)


