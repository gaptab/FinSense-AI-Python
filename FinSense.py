import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate Dummy Data
# Ability to Pay Dataset
np.random.seed(42)
data = {
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.randint(30000, 150000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'outstanding_balance': np.random.randint(0, 50000, 1000),
    'payment_history': np.random.choice([0, 1], size=1000),  # 1: Good, 0: Bad
    'ability_to_pay': np.random.choice([0, 1], size=1000)     # 1: Can Pay, 0: Can't Pay
}

df = pd.DataFrame(data)

# Customer Interaction Dataset for NLP
interaction_data = {
    'customer_id': np.arange(1, 501),
    'interaction_text': [
        "I am struggling to make payments" if i % 5 == 0 else "I am satisfied with the service"
        for i in range(1, 501)
    ]
}

interaction_df = pd.DataFrame(interaction_data)

# 2. Gradient Boosting Model for Ability to Pay
X = df[['age', 'income', 'credit_score', 'outstanding_balance', 'payment_history']]
y = df['ability_to_pay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Gradient Boosting Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 3. Clustering with NLP
# Text Preprocessing using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = vectorizer.fit_transform(interaction_df['interaction_text'])

# Clustering with KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
interaction_df['cluster'] = clusters

# Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())
interaction_df['pca1'] = pca_result[:, 0]
interaction_df['pca2'] = pca_result[:, 1]

# Plot Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=interaction_df, palette='viridis')
plt.title('Customer Interaction Clusters')
plt.show()

# 4. Save Results and Reports
# Save datasets and results
df.to_csv('ability_to_pay_dataset.csv', index=False)
interaction_df.to_csv('customer_interactions_with_clusters.csv', index=False)

# Generate Model Monitoring Report
monitoring_report = {
    'model': 'Gradient Boosting Classifier',
    'accuracy': accuracy,
    'clusters': interaction_df['cluster'].value_counts().to_dict()
}

monitoring_df = pd.DataFrame([monitoring_report])
monitoring_df.to_csv('model_monitoring_report.csv', index=False)

# Summary Presentation
print("Model and Clustering Results Saved:")
print("1. ability_to_pay_dataset.csv")
print("2. customer_interactions_with_clusters.csv")
print("3. model_monitoring_report.csv")