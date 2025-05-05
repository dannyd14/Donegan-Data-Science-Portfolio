import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Page config
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")

# --- Sidebar: Dataset Selection ---
st.sidebar.title("🔧 Configuration")

data_source = st.sidebar.radio("Choose your data source:", ["Upload your own", "Use a sample dataset"])
df = None

if data_source == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = df.dropna()
            st.sidebar.success("✅ File uploaded")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
else:
    sample_dataset = st.sidebar.selectbox("Select a sample dataset:", ["Iris", "Wine", "Breast Cancer"])
    if sample_dataset == "Iris":
        data = datasets.load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    elif sample_dataset == "Wine":
        data = datasets.load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    elif sample_dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    st.sidebar.success(f"✅ Loaded {sample_dataset} dataset")

# --- Main Area ---
st.title("🧠 Unsupervised ML Explorer")

# --- Model Descriptions ---
st.markdown("### 🤖 About the Models")
st.info(
    """
**K-Means Clustering**: K-Means Clustering is a method of Unsupervised Machine Learning used to group the data into k pre-selected groups based on the features that are being applied. The goal of this method is to minimize the variance within each cluster, and match each data point with the group that it fits best in. This method is fast and scalable, especially with large datasets, and works best when the clusters are evenly sized and spherical in shape.

**Hierarchical Clustering**: Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy (tree) of clusters. Instead of specifying the number of clusters in advance, it shows how clusters are formed at different levels of similarity, allowing you to decide where to “cut” the tree. Most commonly, the agglomerative approach is used, where you start with each point as its own cluster, and then merge the two closest clusters at each step and repeat until all points have a cluster. There are 4 main type of linkage methods. Ward minimizes the variance within clusters, single minimizes the distance between the closest points in two clusters, complete maximizes the distance between the furthest points in two clusters, and average minimizes the average distance between all points in two clusters.


**Principal Component Analysis (PCA)**: PCA is a dimensionality reduction technique used in unsupervised learning. It transforms a dataset with potentially many correlated features into a smaller set of uncorrelated features called principal components, while retaining as much of the data's variability as possible. Typically, PCA is used to visualize high dimensional data in 2D or 3D, remove redundancy within the features, speed up learning algorithms, or identify patterns within the data.
"""
)

# --- Dataset Preview ---
if df is not None:
    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### 🧾 Dataset Info")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")
    st.write("**Missing values:**")
    st.dataframe(df.isnull().sum())

    # --- Sidebar: Feature Selection ---
    st.sidebar.subheader("Feature Selection")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    selected_features = st.sidebar.multiselect("Select features", numeric_cols, default=numeric_cols)

    if selected_features:
        X = df[selected_features].values
        st.session_state["X"] = X
        st.session_state["feature_names"] = selected_features

        # --- Sidebar: Model Selection ---
        st.sidebar.subheader("Model Selection")
        model_type = st.sidebar.selectbox("Unsupervised Method", ["K-Means Clustering", "Hierarchical Clustering", "PCA"])
        model = None

        if model_type == "K-Means Clustering":
            n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
            random_state = st.sidebar.number_input("Random state", value=42)
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
            st.session_state["model_type"] = "kmeans"

        elif model_type == "Hierarchical Clustering":
            n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
            linkage_method = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            st.session_state["model_type"] = "hierarchical"
            st.session_state["linkage_method"] = linkage_method

        elif model_type == "PCA":
            max_components = min(len(selected_features), 10)
            n_components = st.sidebar.slider("Number of components", 2, max_components, 2)
            model = PCA(n_components=n_components)
            st.session_state["model_type"] = "pca"

        st.session_state["model"] = model

        # --- Sidebar: Train Button ---
        if st.sidebar.button("🚀 Train Model"):
            st.subheader("🎯 Results")

            try:
                model_type = st.session_state["model_type"]
                model = st.session_state["model"]
                X = st.session_state["X"]

                if model_type == "pca":
                    X_transformed = model.fit_transform(X)
                    st.markdown("### 🎨 PCA Projection")
                    st.markdown("This scatter plot displays the data projected onto the first two principal components. It helps visualize the structure of high-dimensional data in two dimensions. Because PCA is a dimensionality reduction method and not a clustering method, the goal of this process is to make the data easier to visualize, not classify into groups. Here, we are able to visualize the data using the first two PCA components, as otherwise if the data has more than 2 or 3 dimensions, it would be difficult to visualize. Also, it is important to note that the first PCA component accounts for the most variability in the data, followed by the second and decreasing from there.")
                    fig, ax = plt.subplots()
                    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel("PC 2")
                    ax.set_title("PCA - First Two Components")
                    st.pyplot(fig)

                    st.markdown("### 📊 Scree Plot")
                    st.markdown("The scree plot shows the proportion of the dataset's variance explained by each principal component. A sharp decline (elbow) in the plot suggests a good cutoff for the number of components to keep. This helps determine how many components are sufficient to retain most of the original information.")

                    explained_var = model.explained_variance_ratio_
                    fig, ax = plt.subplots()
                    ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='-')
                    ax.set_title("Scree Plot")
                    ax.set_xlabel("Principal Component")
                    ax.set_ylabel("Explained Variance Ratio")
                    ax.grid(True)
                    st.pyplot(fig)

                else:
                    cluster_labels = model.fit_predict(X)

                    # PCA for 2D visualization
                    pca_vis = PCA(n_components=2)
                    X_2d = pca_vis.fit_transform(X)

                    st.markdown("### 📉 Cluster Visualization (PCA Projection)")
                    st.markdown("This plot shows how the clusters form in 2D space after PCA reduction. Because of the potential use of multiple features, PCA reduction has been applied here to allow the scatterplot to be visualized in 2D. This graph can be used to see how effective these clustering methods were in assigning points to clusters. Points with the same color belong to the same cluster, and the distance between points indicates how similar they are to each other. The clustering methods perform well when the clusters are clearly separated from each other and all points that are close to each other are the same color.")
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel("PC 2")
                    ax.set_title(f"{model_type} Clusters (2D PCA)")
                    st.pyplot(fig)

                    st.markdown("### 📈 Clustering Metrics")
                    if len(set(cluster_labels)) > 1:
                        score = silhouette_score(X, cluster_labels)
                        st.markdown("**Silhouette Score**: Measures how similar each point is to its own cluster vs. other clusters. This value gives you a general estimation for how well the clustering method worked. Values that are close to 1 imply that data points are well separated into different clusters. Values around 0 imply that clusters are overlapping or ambiguous. Negative values imply that points are in the wrong clusters. The silhouette score can be used to determine which hyperparamters perform the best.")
                        st.write(f"**Silhouette Score:** {score:.3f}")
                    else:
                        st.warning("Only one cluster found — silhouette score not applicable.")

                    if model_type == "hierarchical":
                        st.markdown("### 🌳 Dendrogram")
                        st.markdown("A dendrogram illustrates the hierarchy of clusters created during agglomerative clustering. Cutting the tree at a specific level gives you your clusters. Within this graph, the x-axis represents the sample index, and the y-axis represents the distance between clusters. The height of the lines in the dendrogram indicates how far apart the clusters are. The higher the line, the more dissimilar the clusters are. The dendrogram can be used to determine how many clusters to use by cutting the tree at a certain height. Also, clusters are separated by color, though depending on the size of the idex, some of these clusters may be difficult to see visually.")
                        Z = linkage(X, method=st.session_state["linkage_method"])
                        fig, ax = plt.subplots(figsize=(10, 5))
                        dendrogram(Z, ax=ax)
                        ax.set_title("Hierarchical Clustering Dendrogram")
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("Distance")
                        st.pyplot(fig)

                    elif model_type == "kmeans":
                        st.markdown("### 📐 Elbow Plot")
                        st.markdown("The elbow plot shows how the within-cluster sum of squares (inertia) decreases with increasing number of clusters. The 'elbow' point helps choose the optimal number, where this point is typically where the direction of the graph sees a distinct change. By applying the number of clusters at the elbow point, we are able to find a balance of minimzing the inertia and maximizing the number of clusters. At the elbow point, the intertia no longer improves rapidly with increasing number of clusters.")
                        inertia = []
                        K = range(1, 11)
                        for k in K:
                            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
                            inertia.append(kmeans.inertia_)

                        fig, ax = plt.subplots()
                        ax.plot(K, inertia, 'bo-')
                        ax.set_xlabel("Number of Clusters")
                        ax.set_ylabel("Inertia")
                        ax.set_title("Elbow Method For Optimal K")
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Error during training or visualization: {e}")
    else:
        st.warning("⚠️ Please select at least one numeric feature.")
else:
    st.info("Upload or select a dataset to begin.")
