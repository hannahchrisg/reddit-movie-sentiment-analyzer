# Reddit Comment Sentiment Analyzer Bot

## Overview
This project is a Python-based bot that collects comments from Reddit posts and analyzes their sentiment using a machine learning model. The bot helps Reddit users and moderators quickly understand the overall tone of discussions in a thread.

The application retrieves comments from Reddit posts using the Reddit API, processes the text, and predicts whether the sentiment is positive, negative, or neutral.

The model is trained once using a dataset of labeled comments so predictions during runtime are fast.

## Purpose
The goal of this project is to help Reddit users understand the sentiment of discussions in comment sections.

The bot provides insights such as:
- Overall sentiment of comment threads
- Distribution of positive and negative comments
- Understanding community reactions to posts

The bot **does not spam, vote, or manipulate Reddit content**. It only reads publicly available comments and analyzes them.

## Features
- Fetch comments from Reddit posts
- Clean and preprocess text
- Predict sentiment using a trained machine learning model
- Export results to CSV
- Fast predictions after model training

## Technology Stack
- Python
- PRAW (Python Reddit API Wrapper)
- Scikit-learn
- Pandas
- NLTK

## Project Structure

reddit-sentiment-bot

main.py – main script  
train_model.py – trains the sentiment model  
combine.py – prepares dataset  
dataset/ – training data  
output/ – sentiment results  
README.md – project documentation  


Install dependencies

Download nltk stopwords

## Reddit API Setup

Create a Reddit app and get:

- client_id
- client_secret
- user_agent

Add them in `main.py`.

Example:
import praw

reddit = praw.Reddit(
client_id="YOUR_CLIENT_ID",
client_secret="YOUR_CLIENT_SECRET",
user_agent="reddit-sentiment-bot by u/YOUR_USERNAME"
)
### Installation

Clone the repository:

git clone https://github.com/hannahcrhisg/reddit-movie-sentiment-analyzer.git
cd reddit-movie-sentiment-analyzer

Install the required dependencies:

pip install praw pandas scikit-learn nltk

Download NLTK stopwords:

python
import nltk
nltk.download('stopwords')


## Running the Program

Run the main script:

python main.py

The program will:
- Load the movie dataset
- Train the sentiment analysis model
- Analyze movie-related text data
- Output sentiment results


## Project Structure

reddit-movie-sentiment-analyzer/

cleaned_movies.csv   # movie dataset  
combine.py           # dataset preparation script  
main.py              # sentiment analysis model and main program  
README.md            # project documentation  


## Subreddits Used

The bot analyzes discussions from public movie-related subreddits such as:

- r/movies
- r/MovieSuggestions
- r/TrueFilm


## Responsible Usage

This project follows the policies of Reddit:

- Only reads publicly available posts and comments
- Does not spam Reddit
- Does not post comments
- Does not vote or manipulate content
- Does not store personal user data

This project is intended for educational and research purposes only.


## License

Educational use.
