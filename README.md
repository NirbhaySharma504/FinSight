# FinSight AI

FinSight AI is an AI-powered equity research tool designed to provide users with insights and analytics on stocks using Langchain and Yahoo Finance API. This application integrates sentiment analysis and offers a user-friendly interface for stock market analysis.
The site link is : [Finsight](https://finsight-2024.streamlit.app/)
The tutorial link is : [Finsight_Tut](https://youtu.be/Cz4qGNCV9Ys?si=_oDi-BYlrOaff4Z4)


## Features

- **Stock Summaries**: Generate summaries and general information about stocks.
- **Sentiment Analysis**: Analyze financial news sentiment using NLTK and Langchain.
- **Interactive Chatbot**: Get stock-related Q&A assistance through an integrated chatbot.
- **Customizable News Feed**: Change date ranges and limits for financial news articles.

## Technologies Used

- **Langchain**: For building the AI-driven features and integrating models.
- **Streamlit**: For creating the frontend interface.
- **Yahoo Finance API**: For retrieving stock market data and news.
- **EODHD API**: For sentiment analysis of financial news articles.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/yourusername/finsight-ai.git
cd finsight-ai 
```
### Install Dependencies
```bash
pip install streamlit yfinance requests langchain_groq eodhd sentence-transformers matplotlib pandas annoy transformers pillow
```
### Environment Variables

To run the FinSight AI application, you need to configure the following environment variables in a `.env` file or your system environment. These variables store sensitive API keys and configuration settings.

| Variable Name         | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `STREAMLIT_APP_PORT`   | The port number where the Streamlit app will run (default: 8501). |
| `YAHOO_FINANCE_API_KEY`| The API key for accessing Yahoo Finance API data.               |
| `LANGCHAIN_API_KEY`    | The API key for accessing Langchain models (e.g., ChatGroq).    |
| `EODHD_API_KEY`     | The API key for sentiment analysis models.        |

### Example `.env` File:

```bash
STREAMLIT_APP_PORT=8501
YAHOO_FINANCE_API_KEY=your_yahoo_finance_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
EODHD_API_KEY=your_eodhd_api_key
```
Ensure that you replace the placeholder values with your actual API keys.

### Loading Environment Variables

Load the environment variables in your application using the `dotenv` library:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

streamlit_port = os.getenv('STREAMLIT_APP_PORT')
yahoo_api_key = os.getenv('YAHOO_FINANCE_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
nltk_corpus_path = os.getenv('NLTK_CORPUS_PATH')
```


## Running the Application

To run the FinSight AI application, use the following command:
```bash
streamlit run app.py
```
Access the application in your web browser at `http://localhost:8501`

## Contributions

Kunal Mittal  `Equity Research Report`
- **LinkedIn**: [Kunal Mittal](https://www.linkedin.com/in/kunal-mittal-749a1a27b/)
- **Email**: [kunal.mittal@iiitb.ac.in](mailto:kunal.mittal@iiitb.ac.in)

Nirbhay Sharma `Stock Analysis with QnA`
- **LinkedIn**: [Nirbhay Sharma](https://www.linkedin.com/in/nirbhay-sharma-575639280/)
- **Email**: [nirbhay.sharma@iiitb.ac.in](mailto:nirbhay.sharma@iiitb.ac.in)

Kanav Bhardwaj `PPT and UI-UX`

-  **LinkedIn**: [Kanav Bhardwaj](https://www.linkedin.com/in/kanav-bhardwaj-a25940281/)
- **Email**: [kanav.bhardwaj@iiitb.ac.in](mailto:kanav.bhardwaj@iiitb.ac.in)

## Acknowledgements
We would like to express our gratitude to the following:

- **Groq**: For providing the powerful AI capabilities that enhance our analysis and predictions.
- **Yahoo Finance API**: For offering comprehensive financial data and news that enhance our application.
- **Langchain**: For enabling efficient chaining of LLMs and tools for building intelligent applications.
- **Streamlit**: For providing an intuitive framework for developing data-driven web applications.
- **EODHD API**: For offering extensive financial data and insights that support our application's functionality.
- **The open-source community**: For continuous contributions and innovations that inspire our development.

## PowerPoint Presentation

- [Link to PPT](https://1drv.ms/p/c/fc78c453f4580c22/ETsH6EgWr2NDgQaHxw2yow0BD_bjcdZbGC0ae6fmo30cLA?e=TFs7Oe)

