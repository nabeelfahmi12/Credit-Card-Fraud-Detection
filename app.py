from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import json
import plotly.graph_objs as go
import plotly.express as px
import plotly.subplots as sp
import plotly.figure_factory as ff
from plotly import tools
import os
from io import StringIO

app = Flask(__name__)

url = "https://raw.githubusercontent.com/Credit-Card-Fraud-Detection/creditcard.csv"
data_df = pd.read_csv(url)

# Extract hour from the 'Time' column
# Extract hour from the 'Time' column
print("Available columns:", data_df.columns.tolist()) 
data_df['Hour'] = (data_df['Time'] // 3600).astype(int)

# Aggregation statistics for transaction Amount
agg_metrics = ['min', 'max', 'count', 'sum', 'mean', 'median', 'var']

# Group by Hour and Class, computing multiple statistics on Amount
df_hour = (data_df.groupby(['Hour', 'Class'])['Amount']
      .agg(agg_metrics)
      .rename(columns={'count': 'Transactions'})  # Rename 'count' for better clarity
      .reset_index()
     )
df_hour['Total_Amount'] = df_hour['mean'] * df_hour['Transactions']

# Load the trained model
MODEL_PATH = "model/CatBoost_fraud_detection_model.pkl"
# Load the model from the file

loaded_model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# 🚀 Introduction Page - Home
@app.route("/")
def index():
    return render_template("index.html")

# 📊 Analytics Page - Data Exploration
@app.route("/analytics")
def analytics():

    # 📊 Hourly Transactions Graph
    fig_hourly = px.bar(df_hour, x='Hour', y='Transactions', color='Class',
                        title="📊 Hourly Transactions (Fraud vs. Non-Fraud)",
                        labels={'Transactions': 'Number of Transactions', 'Hour': 'Hour of the Day'},
                        barmode='group', text='Transactions')

    fig_hourly.add_trace(go.Scatter(x=df_hour['Hour'], y=df_hour['Total_Amount'],
                                    mode='lines+markers', name="Total Transaction Amount",
                                    line=dict(color="black", width=2)))

    # 🍕 Fraud vs Non-Fraud Pie Chart
    fraud_ratio = data_df['Class'].value_counts(normalize=True) * 100
    fig_pie = px.pie(names=['Non-Fraud', 'Fraud'], values=fraud_ratio,
                     title="Fraudulent Transactions Percentage",
                     color_discrete_map={'Non-Fraud': 'blue', 'Fraud': 'red'})

    # 🔥 Feature Correlation Heatmap (Converted to Plotly)
    correlation_matrix = data_df.drop(columns=['Time']).corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='rdylbu',  # ✅ Use a valid Plotly colorscale
        hoverongaps=False
    ))
    fig_heatmap.update_layout(title="Feature Correlation Heatmap")

    # 📊 Feature Distributions
    features = data_df.columns[:-2]  # Excluding 'Time' and 'Class'
    t0 = data_df[data_df["Class"] == 0]  # Non-Fraud
    t1 = data_df[data_df["Class"] == 1]  # Fraud

    fig_dist = sp.make_subplots(rows=8, cols=4, subplot_titles=features)

    for i, feature in enumerate(features):
        row = i // 4 + 1
        col = i % 4 + 1
        fig_dist.add_trace(go.Histogram(x=t0[feature], histnorm='probability density',
                                        name=f"{feature} (Non-Fraud)", opacity=0.6, marker_color='blue'), row=row, col=col)
        fig_dist.add_trace(go.Histogram(x=t1[feature], histnorm='probability density',
                                        name=f"{feature} (Fraud)", opacity=0.6, marker_color='red'), row=row, col=col)

    fig_dist.update_layout(height=2400, width=1500, title_text="Feature Distributions for Fraud vs Non-Fraud",
                           showlegend=False)

    # 📈 Time Density Plot
    class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
    class_1 = data_df.loc[data_df['Class'] == 1]["Time"]
    hist_data = [class_0, class_1]
    group_labels = ['Not Fraud', 'Fraud']
    fig_time_density = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig_time_density.update_layout(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))

    # 💰 Fraudulent Transactions Scatter Plot
    fraud = data_df.loc[data_df['Class'] == 1]
    trace = go.Scatter(x=fraud['Time'], y=fraud['Amount'], name="Amount",
                       marker=dict(color='red', line=dict(color='red', width=1), opacity=0.5),
                       text=fraud['Amount'], mode="markers")

    fig_fraud_amount = go.Figure(data=[trace])
    fig_fraud_amount.update_layout(title='Amount of Fraudulent Transactions', xaxis_title='Time [s]', yaxis_title='Amount')

    # Convert plots to JSON
    plots = {
        "fig_hourly": fig_hourly.to_json(),
        "fig_pie": fig_pie.to_json(),
        "fig_heatmap": fig_heatmap.to_json(),
        "fig_dist": fig_dist.to_json(),
        "fig_time_density": fig_time_density.to_json(),
        "fig_fraud_amount": fig_fraud_amount.to_json(),
    }

    return render_template("analytics.html", plots=plots)


# 🔮 Prediction Page - User Inputs or File Upload
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            # 1️⃣ **Check if a file is uploaded**
            if "file" in request.files and request.files["file"].filename != "":
                file = request.files["file"]

                if file.filename.endswith(".csv"):
                    data = pd.read_csv(file)
                elif file.filename.endswith(".json"):
                    data = pd.read_json(file, lines=True )
                    print("data")
                else:
                    return render_template("predict.html", error="Unsupported file format")

                if data.empty:
                    return render_template("predict.html", error="Uploaded file is empty")
                print(" i come here")
                numeric_features = [data.columns[0], data.columns[-1]]  # First and Last column names
                pca_features = list(data.columns[1:-1])  # Middle column names
                print("pca ", pca_features)
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('robust', RobustScaler(), numeric_features),  # Scale first & last column
                        ('standard', StandardScaler(), pca_features)  # Scale PCA features
                    ])

                X_new_normalized = preprocessor.fit_transform(data)

                # 3️⃣ **Make Predictions**
                predictions = model.predict(X_new_normalized)
                data["Prediction"] = predictions

                # 4️⃣ **Save results**
                output_file = "static/predictions.csv"
                data.to_csv(output_file, index=False)

                return render_template("final_predict.html", file_link=output_file)

            else:

                # 5️⃣ **If No File, Check for Manual Input**
                input_text = request.form.get("transaction_data")
                #print("enter in this command", input_text)
                if not input_text:
                    return render_template("predict.html", error="No data provided")
                #print("passed first test")
                # Convert input string to list of floats
                data_list = list(map(float, input_text.split(",")))
                input_df = pd.DataFrame([data_list])

                if input_df.empty:
                    return render_template("predict.html", error="Invalid data format")
                print("created Data frame")
                # 6️⃣ **Normalize Input Data**
                numeric_features = [input_df.columns[0], input_df.columns[-1]]  # First and Last column names
                pca_features = list(input_df.columns[1:-1])  # Middle column names
                print("pca ", pca_features)
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('robust', RobustScaler(), numeric_features),  # Scale first & last column
                        ('standard', StandardScaler(), pca_features)  # Scale PCA features
                    ])

                X_new_normalized = preprocessor.fit_transform(input_df)
                print("Data is normalized")

                # 7️⃣ **Make Single Prediction**
                prediction = model.predict(X_new_normalized)
                if prediction[0] == 0:
                    output = "The transaction was Not Fraud"
                else:
                    output = "The transaction was Fraud"
                return render_template("final_predict.html", single_prediction=output)

        return render_template("predict.html")

    except Exception as e:
        return render_template("predict.html", error=f"Error: {str(e)}")

# 🎯 Final Prediction Page - Display or Download
@app.route("/final_predict")
def final_predict():
    return render_template("final_predict.html")

@app.route("/jupiternotebook")
def notebook():
    return render_template("jupiternotebook.html")

if __name__ == "__main__":
    app.run()
