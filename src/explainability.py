import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def generate_explanations():
    # Load model and test data
    model = joblib.load('models/trained_model.pkl')
    test_df = pd.read_csv('data/processed/test.csv')
    
    X_test = test_df.drop('target', axis=1)
    
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Summary Plot (Global Importance)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance for SL Load Demand")
    plt.savefig('notebooks/shap_summary.png')
    print("SHAP Summary plot saved to notebooks folder.")

    # 2. Decision explanation for a single prediction
    # (Shows how specific inputs pushed the prediction up or down)
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
    plt.savefig('notebooks/shap_force_plot.png')

if __name__ == "__main__":
    generate_explanations()