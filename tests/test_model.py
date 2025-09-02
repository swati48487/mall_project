import unittest
import joblib
from sklearn.cluster import KMeans
import pandas as pd
import os

class TestKMeansModel(unittest.TestCase):
    def test_model_type(self):
        # Load the saved model
        model = joblib.load("model/kmeans_mall_model.pkl")
        # Check that it is a KMeans instance
        self.assertIsInstance(model, KMeans)

    def test_feature_count(self):
        # Load the dataset to confirm number of features
        data = pd.read_csv("data/Mall_Customers.csv")
        X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
        model = joblib.load("model/kmeans_mall_model.pkl")
        # KMeans doesn't store n_features_in_ until fit, but our model is already fit
        self.assertEqual(model.n_features_in_, X.shape[1])

    def test_predict_shape(self):
        # Load the dataset and model
        data = pd.read_csv("data/Mall_Customers.csv")
        X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
        model = joblib.load("model/kmeans_mall_model.pkl")
        preds = model.predict(X)
        self.assertEqual(len(preds), len(data))

if __name__ == "__main__":
    unittest.main()
