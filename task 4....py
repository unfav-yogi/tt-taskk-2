import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Dataset Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_data(df):
    df = df.drop_duplicates()

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    for col in df.columns:
        if "date" in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    return df

def generate_report(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.describe(include='all').to_csv(os.path.join(output_folder, "summary_report.csv"))

    sns.set(style="whitegrid")
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.savefig(os.path.join(output_folder, f"{col}_distribution.png"))
        plt.close()

    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.close()

def save_clean_data(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(os.path.join(output_folder, "cleaned_data.csv"), index=False)

def run_pipeline(input_file, output_folder="output"):
    df = load_data(input_file)
    if df is not None:
        df = clean_data(df)
        save_clean_data(df, output_folder)
        generate_report(df, output_folder)
        print("Automation Completed Successfully!")

if __name__ == "__main__":
    input_file = r"C:\Users\yoeshwar\OneDrive\Pictures\Desktop\internships\cleaned_superstore.csv"
    run_pipeline(input_file)