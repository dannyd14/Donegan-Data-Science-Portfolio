import streamlit as st  #Import streamlit
import pandas as pd #Import pandas 
import numpy as np  #Import numpy 
import matplotlib.pyplot as plt   #Import matplotlib
from sklearn import datasets  #Import sklearn datasets 
from sklearn.cluster import KMeans, AgglomerativeClustering  #Import clustering methods
from sklearn.decomposition import PCA   #Import PCA (Dimension reduction)
from sklearn.metrics import silhouette_score #Silhouette score metric
from scipy.cluster.hierarchy import dendrogram, linkage  #Import dendrogram and linkage

# Page config
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")  #Set title and page configurations

# --- Sidebar: Dataset Selection ---
st.sidebar.title("🔧 Configuration")  #Create the sidebar used for dataset choices

data_source = st.sidebar.radio("Choose your data source:", ["Upload your own", "Use a sample dataset"]) #Radio button to choose between uploading a file or using a sample dataset
df = None #Initialize the dataframe

if data_source == "Upload your own":   #If the user chooses to upload their own file
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])  #File uploader to upload a CSV file
    if uploaded_file is not None:   #If a file is uploaded
        try:
            df = pd.read_csv(uploaded_file)  #Read the CSV file
            df = df.dropna() #Drop any rows with missing values
            st.sidebar.success("✅ File uploaded")  #Success message 
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}") # Error message if the file cannot be read
else:
    sample_dataset = st.sidebar.selectbox("Select a sample dataset:", ["Iris", "Wine", "Breast Cancer"]) #Select a sample dataset
    if sample_dataset == "Iris": #If the user selects the Iris dataset
        data = datasets.load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names) #Create a dataframe with the Iris dataset
    elif sample_dataset == "Wine":  #If the user selects the Wine dataset
        data = datasets.load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)  #Create a dataframe with the Wine dataset
    elif sample_dataset == "Breast Cancer": #If the user selects the Breast Cancer dataset 
        data = datasets.load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names) #Create a dataframe with the Breast Cancer dataset
    st.sidebar.success(f"✅ Loaded {sample_dataset} dataset") #Success message 
# --- Main Area ---
st.title("🧠 Unsupervised ML Explorer")  #Main title

# --- Model Descriptions ---
st.markdown("### 🤖 About the Models")  #Information about models prior to running any of them
st.info(
    """
**K-Means Clustering**: K-Means Clustering is a method of Unsupervised Machine Learning used to group the data into k pre-selected groups based on the features that are being applied. The goal of this method is to minimize the variance within each cluster, and match each data point with the group that it fits best in. This method is fast and scalable, especially with large datasets, and works best when the clusters are evenly sized and spherical in shape.

**Hierarchical Clustering**: Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy (tree) of clusters. Instead of specifying the number of clusters in advance, it shows how clusters are formed at different levels of similarity, allowing you to decide where to “cut” the tree. Most commonly, the agglomerative approach is used, where you start with each point as its own cluster, and then merge the two closest clusters at each step and repeat until all points have a cluster. There are 4 main type of linkage methods. Ward minimizes the variance within clusters, single minimizes the distance between the closest points in two clusters, complete maximizes the distance between the furthest points in two clusters, and average minimizes the average distance between all points in two clusters.


**Principal Component Analysis (PCA)**: PCA is a dimensionality reduction technique used in unsupervised learning. It transforms a dataset with potentially many correlated features into a smaller set of uncorrelated features called principal components, while retaining as much of the data's variability as possible. Typically, PCA is used to visualize high dimensional data in 2D or 3D, remove redundancy within the features, speed up learning algorithms, or identify patterns within the data.
"""
)

# --- Dataset Preview ---
if df is not None:
    st.markdown("### 📊 Dataset Preview")  #Title
    st.dataframe(df.head()) #Display the first 5 rows of the dataframe

    st.markdown("### 🧾 Dataset Info")  
    st.write(f"**Shape:** {df.shape}") #Display the shape of the dataframe
    st.write(f"**Columns:** {list(df.columns)}") #Display the columns of the dataframe
    st.write("**Missing values:**") #   Display the missing value
    st.dataframe(df.isnull().sum()) #Display the missing values in the dataframe

    # --- Sidebar: Feature Selection ---
    st.sidebar.subheader("Feature Selection")  #Feature selection
    all_cols = df.columns.tolist() #Get all columns in the dataframe
    #numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist() #Select numeric columns
    selected_features = st.sidebar.multiselect("Select features", all_cols, default= all_cols) #Multiselect to select features
    if selected_features:
        selected_data = df[selected_features] #Get the selected features from the dataframe
        selected_data = pd.get_dummies(selected_data, drop_first = True) #Convert categorical features to dummy variables
        
        X = df[selected_features].values #Convert the selected features to a numpy array
        st.session_state["X"] = X #Store the selected features in the session state
        st.session_state["feature_names"] = selected_features #Store the feature names in the session state

        # --- Sidebar: Model Selection ---
        st.sidebar.subheader("Model Selection") #Model selection
        model_type = st.sidebar.selectbox("Unsupervised Method", ["K-Means Clustering", "Hierarchical Clustering", "PCA"]) #select box for model types
        model = None #Initialize the model variable

#Use exact name values
        if model_type == "K-Means Clustering": #If the user selects K-Means Clustering
            n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3) #Slider to select the number of clusters
            random_state = st.sidebar.number_input("Random state", value=42) #Number input for random state
            model = KMeans(n_clusters=n_clusters, random_state=random_state) #Create the KMeans model
            st.session_state["model_type"] = "kmeans" #Store the model type in the session state

        elif model_type == "Hierarchical Clustering": #If the user selects Hierarchical Clustering
            n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3) #Slider to select the number of clusters
            linkage_method = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average", "single"]) #Select box for linkage method
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method) #Create the AgglomerativeClustering model
            st.session_state['model_type'] = 'hierarchial'
            st.session_state["linkage_method"] = linkage_method #Store the linkage method in the session state

        elif model_type == "PCA": #If the user selects PCA
            max_components = min(len(selected_features), 10) #Maximum number of components
            n_components = st.sidebar.slider("Number of components", 2, max_components, 2) #Slider to select the number of components
            model = PCA(n_components=n_components) #Create the PCA model
            st.session_state["model_type"] = "pca" #Store the model type in the session state

        st.session_state["model"] = model  #Store the model in the session state

        # --- Sidebar: Train Button ---
        if st.sidebar.button("🚀 Train Model"):  #Train model button
            st.subheader("🎯 Results") #header

            try:
                model_type = st.session_state["model_type"] #Get the model type from the session state
                model = st.session_state["model"] #Get the model from the session state
                X = st.session_state["X"] #Get the selected features from the session state

                if model_type == "pca":  #If the user selects PCA
                    X_transformed = model.fit_transform(X) #Transform the data using PCA
                    st.markdown("### 🎨 PCA Projection")
                    st.markdown("This scatter plot displays the data projected onto the first two principal components. It helps visualize the structure of high-dimensional data in two dimensions. Because PCA is a dimensionality reduction method and not a clustering method, the goal of this process is to make the data easier to visualize, not classify into groups. Here, we are able to visualize the data using the first two PCA components, as otherwise if the data has more than 2 or 3 dimensions, it would be difficult to visualize. Also, it is important to note that the first PCA component accounts for the most variability in the data, followed by the second and decreasing from there.")
                    fig, ax = plt.subplots() #Create a figure and axis
                    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7) #Scatter plot of the PCA transformed data
                    ax.set_xlabel("PC 1") #X-axis label
                    ax.set_ylabel("PC 2") #Y-axis label
                    ax.set_title("PCA - First Two Components") #Title
                    st.pyplot(fig) #Display the plot
                    
                    st.markdown("### 📊 Scree Plot") #title
                    st.markdown("The scree plot shows the proportion of the dataset's variance explained by each principal component. A sharp decline (elbow) in the plot suggests a good cutoff for the number of components to keep. This helps determine how many components are sufficient to retain most of the original information.")

                    explained_var = model.explained_variance_ratio_ #Get the explained variance ratio
                    fig, ax = plt.subplots() #Create a figure and axis
                    ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='-') #Plot the explained variance ratio
                    ax.set_title("Scree Plot") #Title
                    ax.set_xlabel("Principal Component") #X-axis label
                    ax.set_ylabel("Explained Variance Ratio") #Y-axis label
                    ax.grid(True) #Grid
                    st.pyplot(fig) #Display the plot

                else:
                    cluster_labels = model.fit_predict(X) #Fit the model to the data and predict the cluster labels

                    # PCA for 2D visualization
                    pca_vis = PCA(n_components=2) #Create a PCA model for visualization
                    X_2d = pca_vis.fit_transform(X) #Transform the data using PCA

                    st.markdown("### 📉 Cluster Visualization (PCA Projection)") 
                    st.markdown("This plot shows how the clusters form in 2D space after PCA reduction. Because of the potential use of multiple features, PCA reduction has been applied here to allow the scatterplot to be visualized in 2D. This graph can be used to see how effective these clustering methods were in assigning points to clusters. Points with the same color belong to the same cluster, and the distance between points indicates how similar they are to each other. The clustering methods perform well when the clusters are clearly separated from each other and all points that are close to each other are the same color.")
                    fig, ax = plt.subplots() #Create a figure and axis
                    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7) #Scatter plot of the PCA transformed data
                    ax.set_ylabel("PC 2") #Y-axis label
                    ax.set_title(f"{model_type} Clusters (2D PCA)") #Title
                    st.pyplot(fig) #Display the plot

                    st.markdown("### 📈 Clustering Metrics") 
                    if len(set(cluster_labels)) > 1: #If there is more than one cluster
                        score = silhouette_score(X, cluster_labels) #Calculate the silhouette score
                        st.markdown("**Silhouette Score**: Measures how similar each point is to its own cluster vs. other clusters. This value gives you a general estimation for how well the clustering method worked. Values that are close to 1 imply that data points are well separated into different clusters. Values around 0 imply that clusters are overlapping or ambiguous. Negative values imply that points are in the wrong clusters. The silhouette score can be used to determine which hyperparamters perform the best.")
                        st.write(f"**Silhouette Score:** {score:.3f}") #Display the silhouette score
                    else:
                        st.warning("Only one cluster found — silhouette score not applicable.") #warning

                    
                    if model_type == "kmeans":  #If the user selects K-Means Clustering
                        st.markdown("### 📐 Elbow Plot") #title
                        st.markdown("The elbow plot shows how the within-cluster sum of squares (inertia) decreases with increasing number of clusters. The 'elbow' point helps choose the optimal number, where this point is typically where the direction of the graph sees a distinct change. By applying the number of clusters at the elbow point, we are able to find a balance of minimzing the inertia and maximizing the number of clusters. At the elbow point, the intertia no longer improves rapidly with increasing number of clusters.")
                        inertia = [] #List to store inertia values
                        K = range(1, 11) #Range of clusters to test
                        for k in K:
                            kmeans = KMeans(n_clusters=k, random_state=42).fit(X) #Fit the KMeans model
                            inertia.append(kmeans.inertia_) #Append the inertia value

                        fig, ax = plt.subplots() #Create a figure and axis
                        ax.plot(K, inertia, 'bo-') #Plot the inertia values
                        ax.set_xlabel("Number of Clusters") #X-axis label
                        ax.set_ylabel("Inertia") #Y-axis label
                        ax.set_title("Elbow Method For Optimal K") #Title
                        st.pyplot(fig) #Display the plot
                    
                    elif model_type == "Hierarchical": #If the user selects Hierarchical Clustering
                        st.markdown("### 🌳 Dendrogram") 
                        st.markdown("A dendrogram illustrates the hierarchy of clusters created during agglomerative clustering. Cutting the tree at a specific level gives you your clusters. Within this graph, the x-axis represents the sample index, and the y-axis represents the distance between clusters. The height of the lines in the dendrogram indicates how far apart the clusters are. The higher the line, the more dissimilar the clusters are. The dendrogram can be used to determine how many clusters to use by cutting the tree at a certain height. Also, clusters are separated by color, though depending on the size of the idex, some of these clusters may be difficult to see visually.")
                        Z = linkage(X, method=st.session_state["linkage_method"]) #Perform hierarchical clustering
                        fig, ax = plt.subplots(figsize=(10, 5)) #Create a figure and axis
                        dendrogram(Z, ax=ax) #Dendrogram
                        ax.set_title("Hierarchical Clustering Dendrogram")
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("Distance")
                        st.pyplot(fig)

                
            except Exception as e:
                st.error(f"⚠️ Error during training or visualization: {e}") #Error message if there is an error during training or visualization
    else:
        st.warning("⚠️ Please select at least one numeric feature.") #Warning message if no features are selected
else:
    st.info("Upload or select a dataset to begin.") #Info message if no dataset is selected


##To run the app locally, type 'streamlit run MLUnsupervisedApp/UnsupervisedML-APP.py' in the terminal