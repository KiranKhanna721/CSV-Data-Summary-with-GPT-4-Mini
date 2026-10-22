# CSV Data Summary with GPT-4o mini

A Streamlit tool that writes a short report about any CSV file. You upload a file and can add an extra instruction. The app sends the column names and `describe()` statistics to OpenAI's `gpt-4o-mini` and shows a summary of key insights and patterns.

## Features

- Upload a CSV and preview it
- Add an optional custom prompt to steer the analysis
- Get an AI-written summary and report

## Configuration

Set your OpenAI key as an environment variable:

```bash
export OPENAI_API_KEY="your-key"
```

## Getting started

```bash
pip install -r requirements.txt
streamlit run summary.py
```
