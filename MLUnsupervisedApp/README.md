## Unsupervised Machine Learning Streamlit Exploratory Application
This project will be an example of an application created with the intention of giving users an opportunity to learn more about unsupervised machine learning while experimenting with a few different methods, either using sample or their own data. The methods included in this project are K-Means Clustering and Hierachial Clustering for classification models, and PCA as an example of a dimension reduction method. 

## Project Overview
After making the decision to either import their own data or take advantage of one of three sample datasets, users are then guided through a few choices that allow them to explore how a few different unsupervised machine learning models operate. Users are able to select which features from the dataset they would like to include in the model, and then they are also adjust the respective hyperparameters for each type of model. After training the model, users are able to view the results of the model. The main goal of this model is to allow a user to learn more about unsupervised machine learning, gaining information about which methods are most effective for certain datasets.


## App Features
Within this app, users have the ability to test out a variety of differen unsupervised machine learning models, either on their own dataset, or using a sample dataset. This exploration could include choosing feature variables, training a few different types of models, viewing how the model performed by examining peformance statistics and visualizations that describe how the model performs. Model types include K-Means clustering, Hierarchial Clustering, and PCA dimension reduction. Also, the user has the ability to adjust the hyperparamters that drive these models, making the application even more customizable. Each model has its own set of hyperparameters, and they are shown as a slider within this app. After training the model, the user can view different tabs of the results and visualizations, with descriptions of the different metrics given. 




## Instructions for App Usage - [Link to Deployed Version](NEED TO ADD LINK ONCE DEPLOYED)
1. Open the either through the Streamlit Community Cloud or via a URL. If using the app locally, type 'streamlit run MLStreamlitApp/Streamlit-App.py' in the terminal of the Streamlit-App.py file. If running locally, necessary libraries include: streamlit, pandas, numpy, sklearn, matplotlib, scipy
2. Choose whether you would like to insert your own data set, or view a sample
3. Select which features will be included in this model
4. Choose your model type
5. Adjust the hyperparamters using sliding scalers
6. Train the model
7. Observe the results and evaluate which performs the best


## References
Here is some other resources that could be used to learn more about unsupervised machine learning

[Unsupervised Machine Learning Basics](MLUnsupervisedApp/UnsupervisedMachineLearning.pdf)

## Example of Application
![](<Screen Shot 2025-05-05 at 8.20.15 PM.png>)
