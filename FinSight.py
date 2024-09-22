import requests
import json
import os
import yfinance as yf
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
#from saved import open_api_key
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt
from eodhd import APIClient
from sentence_transformers import SentenceTransformer

import numpy as np
from annoy import AnnoyIndex
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_groq import ChatGroq
import os
from langchain.prompts import PromptTemplate
# Set your Groq API key as an environment variable
os.environ['GROQ_API_KEY'] = "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"

# Initialize ChatGroq
llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), temperature=0.6)

# Initialize EODHD API
api_key_eodhd = '66ef487f742278.78938106'  # Your EODHD API key
api = APIClient(api_key_eodhd)

# Streamlit App Layout
st.set_page_config(page_title="FinSight AI - Equity Research", layout="wide")

# Sidebar
st.sidebar.title("FinSight AI")
logo = Image.open(r"/Users/nirbhay/Desktop/img.jpeg")
st.sidebar.image(logo, use_column_width=True)
st.sidebar.write("""
Welcome to **FinSight AI**. Enter a stock symbol to generate a real-time AI-driven equity research report and queries section.
""")

# Sidebar Links for Scrolling
st.sidebar.header("Sections")
st.sidebar.subheader("Equity Research Report")
st.sidebar.markdown("[Stock Summary](#stock-summary)", unsafe_allow_html=True)
st.sidebar.markdown("[Historical Stock Prices](#historical-stock-prices)", unsafe_allow_html=True)
st.sidebar.markdown("[Sustainability](#sustainability)", unsafe_allow_html=True)
st.sidebar.markdown("[Analyst Recommendations](#analyst-recommendations)", unsafe_allow_html=True)
st.sidebar.markdown("[AI-Driven Stock Analysis](#ai-driven-stock-analysis)", unsafe_allow_html=True)
st.sidebar.markdown("[Sentiment Analysis](#sentiment-analysis)", unsafe_allow_html=True)

st.sidebar.subheader("Stock Market Analysis With QnA")
st.sidebar.markdown("[QnA Section](#qna-section)",unsafe_allow_html=True)
# Main Header
st.title("FinSight AI - AI-Powered Equity Research")

# Stock Symbol Input
st.header("Equity Research Report")
stock_symbol = st.text_input("Stock Symbol (e.g., AAPL, TSLA)", "")

# Function to fetch stock data from Yahoo Finance
def get_stock_summary(symbol):
    try:
        stock = yf.Ticker(symbol)
        summary = stock.info
        history = stock.history(period="1mo")
        actions = stock.actions
        dividends = stock.dividends
        splits = stock.splits
        sustainability = stock.sustainability
        calendar = stock.calendar
        recommendations = stock.recommendations
        
        return summary, history, actions, dividends, splits, sustainability, calendar, recommendations
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None, None, None, None, None, None, None, None

# Function to generate AI-driven stock analysis
def generate_stock_analysis(symbol):
    prompt_template = PromptTemplate(
        input_variables=["symbol"], 
        template="Generate a detailed analysis report for the stock symbol {symbol}, including company overview, market trends, and future outlook."
    )
    
    analysis_prompt = prompt_template.format(symbol=symbol)
    response = llm.invoke(analysis_prompt)
    
    if response:
        return response.content
    else:
        return "No response generated."

# Sentiment Analysis Functionality
def sentiment_analysis(ticker, from_date, to_date, article_limit):
    try:
        resp = api.financial_news(s=ticker, from_date=from_date, to_date=to_date, limit=article_limit)
        df = pd.DataFrame(resp)

        if not df.empty:
            df['content'] = df['content'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
            sentiments = []

            for content in df['content']:
                prompt = f"Identify the sentiment towards the {ticker} stocks from the news article: {content}"
                sentiment = llm.invoke(prompt)
                sentiments.append(sentiment.content)

            df['sentiment'] = sentiments

            # Categorize sentiments
            def categorize_sentiment(sentiment):
                try:
                    score = float(sentiment)
                    if score > 0:
                        return 'Positive'
                    elif score < 0:
                        return 'Negative'
                    else:
                        return 'Neutral'
                except ValueError:
                    return 'Neutral'
            df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)  
            
            sentiment_counts = df['sentiment_category'].value_counts()

            # Check if sentiment counts are empty
            if sentiment_counts.empty:
                st.warning("No sentiments found.")
                return

            # Plotting the sentiment distribution as a pie chart
            fig, ax = plt.subplots(figsize=(48,50))
            ax.pie(sentiment_counts, labels=None, autopct=None, startangle=140)
            ax.set_title(f'Pie Chart of {ticker} Sentiment', fontsize=50)
            ax.axis('equal')
            st.pyplot(fig)

            # Summary Sentiment Analysis
            total_articles = len(df)
            positive_count = sentiment_counts.get('Positive', 0)
            negative_count = sentiment_counts.get('Negative', 0)
            neutral_count = sentiment_counts.get('Neutral', 0)

            st.write(f"### Summary of Sentiment Analysis")
            st.write(f"Total Articles Analyzed: {total_articles}")
            st.write(f"Positive Sentiments: {positive_count}")
            st.write(f"Negative Sentiments: {negative_count}")
            st.write(f"Neutral Sentiments: {neutral_count}")

            overall_sentiment_score = positive_count - negative_count
            if overall_sentiment_score > 0:
                st.success("Overall Sentiment: Positive")
            elif overall_sentiment_score < 0:
                st.error("Overall Sentiment: Negative")
            else:
                st.info("Overall Sentiment: Neutral")
        else:
            st.warning("No articles found for sentiment analysis.")
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")

# Streamlit Title


# Alpha Vantage API key
# Alpha Vantage API key


def fetch_financial_data(symbol):
    url1 = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
    url2 = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}'
    url3 = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}'
    
    response = requests.get(url1)
    data = response.json()
    report = eval(json.dumps(data))
    income_statement = report["quarterlyReports"]

    response = requests.get(url2)
    data = response.json()
    report = eval(json.dumps(data))
    balance_sheet = report["quarterlyReports"]
    response = requests.get(url3)
    data = response.json()
    report = eval(json.dumps(data))
    cash_flow = report["quarterlyReports"]

    return income_statement, balance_sheet, cash_flow

# Stock Summary Section
def stock_summary(symbol):
    st.subheader(f"Stock Summary for {symbol}")
    income_statement, balance_sheet, cash_flow = fetch_financial_data(symbol)

    # Display the first few entries from the financial statements
    '''if income_statement:
        st.write("Income Statement (Quarterly):")
        st.json(income_statement[:2])  # Show top 2 reports
    else:
        st.write("Income statement data not available.")

    if balance_sheet:
        st.write("Balance Sheet (Quarterly):")
        st.json(balance_sheet[:2])  # Show top 2 reports
    else:
        st.write("Balance sheet data not available.")

    if cash_flow:
        st.write("Cash Flow Statement (Quarterly):")
        st.json(cash_flow[:2])  # Show top 2 reports
    else:
        st.write("Cash flow data not available.")'''

# QnA Section
def qna_section(symbol, question):
    # Fetch financial data
    income_statement, balance_sheet, cash_flow = fetch_financial_data(symbol)

    # Encoder setup
    encoder = SentenceTransformer("all-mpnet-base-v2")
    
    # Convert statements to strings for encoding
    statement = income_statement[:8]
    sheet = balance_sheet[:8]
    flow = cash_flow[:8]
    
    income_statement_strings = [str(item) for item in statement]
    sheet_strings = [str(item) for item in sheet]
    flow_strings = [str(item) for item in flow]
    
    # Build Annoy index
    annoy_index = AnnoyIndex(768, 'angular')
    
    vectors = np.ascontiguousarray(encoder.encode(income_statement_strings))
    for i, vector in enumerate(vectors):
        annoy_index.add_item(i, vector)
        
    j = len(vectors)
    vectors = np.ascontiguousarray(encoder.encode(sheet_strings))
    for i, vector in enumerate(vectors):
        annoy_index.add_item(i + j, vector)
        
    j += len(vectors)
    vectors = np.ascontiguousarray(encoder.encode(flow_strings))
    for i, vector in enumerate(vectors):
        annoy_index.add_item(i + j, vector)
    
    # Build the index
    annoy_index.build(10)

    # Function to generate answers based on the query
    def get_relevant_statements(query, n_neighbors=5):
        query_vector = np.ascontiguousarray(encoder.encode(query))
        indices = annoy_index.get_nns_by_vector(query_vector, n_neighbors, include_distances=True)
        relevant_statements = [income_statement[i] for i in indices[0]]
        return relevant_statements

    # Generate LLM prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Based on the following statements:\n{context}\n\nAnswer the question: {question}"
    )

    # Main function to handle the question and retrieve answers
    def main(question):
        relevant_statements = get_relevant_statements(question)
        context = "\n"
        for i in relevant_statements:
            for key in i.keys():
                context += f"{key}: {i[key]}, "
            context += "\n"
        
        prompt = prompt_template.format(context=context, question=question)
        
        # Initialize the model for response
        os.environ['GROQ_API_KEY'] = "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"
        llm = ChatGroq(api_key=os.environ['GROQ_API_KEY'])
        response = llm.invoke(prompt)
        
        return response.content

    # Fetch and return the answer
    answer = main(question)
    return answer


# Generate Report Button
if stock_symbol:
    # Fetch and Display Stock Summary
    st.subheader(f"Stock Summary for {stock_symbol.upper()}", anchor="stock-summary")
    stock_summary, stock_history, stock_actions, stock_dividends, stock_splits, stock_sustainability, stock_calendar, stock_recommendations = get_stock_summary(stock_symbol)

    if stock_summary:
        # Display stock summary
        st.write(f"**Company Name**: {stock_summary.get('longName', 'N/A')}")
        st.write(f"**Sector**: {stock_summary.get('sector', 'N/A')}")
        st.write(f"**Industry**: {stock_summary.get('industry', 'N/A')}")

        market_cap = stock_summary.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap_in_billions = market_cap / 1_000_000_000
            st.write(f"**Market Cap**: ${market_cap_in_billions:.2f} billion")
        else:
            st.write(f"**Market Cap**: N/A")

        # Display historical data
        st.subheader("Historical Stock Prices (Last Month)", anchor="historical-stock-prices")
        if stock_history is not None:
            st.line_chart(stock_history['Close'])

        # Display sustainability data (ESG scores)
        st.subheader("Sustainability (ESG Scores)", anchor="sustainability")
        if stock_sustainability is not None and not stock_sustainability.empty:
            st.write(stock_sustainability)
        else:
            st.write("No sustainability data available.")

        # Display recommendations
        st.subheader("Analyst Recommendations", anchor="analyst-recommendations")
        if stock_recommendations is not None and not stock_recommendations.empty:
            st.write(stock_recommendations)
        else:
            st.write("No recommendations available.")

        # Generate AI-driven stock analysis
        st.subheader("AI-Driven Stock Analysis", anchor="ai-driven-stock-analysis")
        analysis_report = generate_stock_analysis(stock_symbol)
        st.write(analysis_report)

        # Sentiment Analysis Section
        st.subheader("Financial News Sentiment Analysis", anchor="sentiment-analysis")

        # Add input options for articles and date above sentiment analysis
        with st.form(key='sentiment_form'):
            from_date = st.date_input("From Date", pd.to_datetime("2024-01-01"))
            to_date = st.date_input("To Date", pd.to_datetime("2024-01-30"))
            article_limit = st.number_input("Number of Articles to Fetch", min_value=1, max_value=100, value=5)
            submit_button = st.form_submit_button("Update Sentiment Analysis")

        if submit_button:
            sentiment_analysis(stock_symbol, from_date, to_date, article_limit)

# Streamlit Input for Stock Symbol and Stock Summary Display
st.header("Stock Market Analysis with QnA")
symbol_input = st.text_input("Enter Stock Symbol", value="")

if symbol_input:
    # Display Stock Summary

    api_key = '0IUZ7GSURVA7SNC1'

    # Company symbol (e.g., 'IBM', 'AAPL')
    symbol = symbol_input

    # Alpha Vantage URL for fetching income statement
    url1 = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
    url2 = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}'
    url3 = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}'

    # Fetch income statement data
    response = requests.get(url1)
    data = response.json()
    report = eval(json.dumps(data))
    income_statement = report["quarterlyReports"]

    response = requests.get(url2)
    data = response.json()
    report = eval(json.dumps(data))
    balance_sheet = report["quarterlyReports"]

    response = requests.get(url3)
    data = response.json()
    report = eval(json.dumps(data))
    cash_flow = report["quarterlyReports"]

    encoder = SentenceTransformer("all-mpnet-base-v2")
    os.environ['GROQ_API_KEY'] = "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"
    llm = ChatGroq(temperature=0.6, model="mixtral-8x7b-32768")
    # Convert income statements to strings
    statement = income_statement[:8]
    sheet = balance_sheet[:8]
    flow = cash_flow[:8]
    income_statement_strings = [str(item) for item in statement]
    sheet_strings = [str(item) for item in sheet]
    flow_strings = [str(item) for item in flow]
    # Encode the income statements
    vectors = np.ascontiguousarray(encoder.encode(income_statement_strings))

    # Create Annoy index
    dim = vectors.shape[1]
    annoy_index = AnnoyIndex(dim, 'angular')

    # Add vectors to the Annoy index
    j=0
    for i in range(len(vectors)):
        annoy_index.add_item(j+i, vectors[i])

    j=len(vectors)
    # Encode the income statements
    vectors = np.ascontiguousarray(encoder.encode(sheet_strings))

    # Create Annoy index
    dim = vectors.shape[1]
    annoy_index = AnnoyIndex(dim, 'angular')

    # Add vectors to the Annoy index
    for i in range(len(vectors)):
        annoy_index.add_item(i+j, vectors[i])

    j+=len(vectors)
    # Encode the income statements
    vectors = np.ascontiguousarray(encoder.encode(flow_strings))

    # Create Annoy index
    dim = vectors.shape[1]
    annoy_index = AnnoyIndex(dim, 'angular')

    # Add vectors to the Annoy index
    for i in range(len(vectors)):
        annoy_index.add_item(i+j, vectors[i])

    # Build the index with a specified number of trees
    num_trees = 10
    annoy_index.build(num_trees)

    # Function to generate an answer based on a question
    def get_relevant_statements(query, n_neighbors=4):
        # Encode the query
        query_vector = np.ascontiguousarray(encoder.encode(query))

        # Find the nearest neighbors
        indices = annoy_index.get_nns_by_vector(query_vector, n_neighbors, include_distances=True)

        # Retrieve the relevant income statements
        relevant_statements = [income_statement[i] for i in indices[0]]
        return relevant_statements


    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Based on the following statements:\n{context}\n\nAnswer the question along with some numbers: {question}"
    )

    # Main function to handle the question and retrieve answers
    def main(question):
        # Get relevant income statements
        relevant_statements = get_relevant_statements(question)
        context = "\n"
        for i in relevant_statements:
            for j in i.keys():
                context+=f"{j}:{i[j]},"
            context+="\n"

        prompt = prompt_template.format(context=context, question=question)

        response=llm.invoke(prompt)

        return response.content

    # Example usage
    #question = "tell about the companys growth over the few months and other details?"
    #answer = main(question)
    #print("Question:", question)
    #print("Answer:", answer)








    # Streamlit QnA Input
    st.subheader("QnA Section")
    question_input = st.text_input("Ask a question about the stock", value="")
    if st.button("Get Answer"):
        answer = main(question_input)
        st.write("Answer:", answer)

# Add JavaScript for smooth scrolling
st.markdown("""
    <script>
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
""", unsafe_allow_html=True)


