import streamlit as st
import pandas as pd
from openai import OpenAI
import os

client = OpenAI()

# Function to load the CSV file
def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to summarize the CSV data using GPT-4 mini
def generate_summary(data, user_prompt):
    # Extract basic statistics to include in the prompt
    description = data.describe().to_string()
    column_names = ", ".join(data.columns)
    prompt = f"""
    I have a dataset with the following columns: {column_names}.
    Here is a summary of the statistics:\n{description}.
    {user_prompt}
    Please provide a high-level summary of this dataset, noting any key insights or interesting patterns and write a brief report .
    """

    # Use OpenAI API to generate a summary with GPT-4 mini
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ensure "gpt-4-mini" is supported in your environment
            messages=[
                {"role": "system", "content": "Given a CSV file, analyze the data and provide a summary."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=15000,
            temperature=0.2
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Main function to execute the upload and summarize workflow
def main():
    st.title("CSV Data Summary with GPT-4 Mini")
    
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    # Text input for additional user prompt
    user_prompt = st.text_area("Additional Prompt (optional)", 
                                 "What specific insights are you looking for?")

    if uploaded_file is not None:
        data = load_csv(uploaded_file)
        if data is not None:
            st.subheader("Data Preview")
            st.dataframe(data.head())  # Show the first few rows of the data
            
            if st.button("Generate Summary"):
                summary = generate_summary(data, user_prompt)
                if summary:
                    st.subheader("Summary of the Data")
                    st.write(summary)

if __name__ == "__main__":
    main()