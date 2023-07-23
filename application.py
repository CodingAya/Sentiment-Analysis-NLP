import streamlit as st
import pandas as pd
import altair as alt
import transformers
from transformers import pipeline
from PIL import Image


# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")

# Fxn
def convert_to_df(sentiment):
#   sentiment_dict = {'metric': ['polarity', 'subjectivity'], 'value': [sentiment.polarity, sentiment.subjectivity]}
    sentiment_dict = {'metric': [], 'value': []}
    for item in sentiment:
        label = item['label'].lower()
        score = item['score']
        sentiment_dict['metric'].append(label)
        sentiment_dict['value'].append(score)

###########
    sentiment_df = pd.DataFrame(sentiment_dict)
    return sentiment_df

def analyze_token_sentiment(docx):
    tokens = docx.split()
    results = sentiment_analyzer(tokens)
    sentiment_dict = {'positives': [], 'negatives': [], 'neutral': []}

    for result in results:
        if result['label'] == 'POSITIVE' and result['score'] > 0.1:
            sentiment_dict['positives'].append((result['label']))
        elif result['label'] == 'NEGATIVE' and result['score'] > 0.1:
            sentiment_dict['negatives'].append((result['label']))
        else:
            sentiment_dict['neutral'].append(result['label'])

    return sentiment_dict


def main():
    st.title("Sentiment Analysis NLP App")
    
    logo_image = Image.open("sentimentanalysislogo.png") 
    st.image(logo_image, use_column_width=True)
    
    st.empty()

    st.subheader("Streamlit Projects")
    
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                #sentiment = sentiment_analyzer(raw_text)[0]
                sentiment = [sentiment_analyzer(raw_text)[0]]
                ###
                st.write(sentiment)

                # Emoji
                if sentiment[0]['label'] == 'POSITIVE':
                    st.markdown("Sentiment: Positive :smiley: ")
                elif sentiment[0]['label'] == 'NEGATIVE':
                    st.markdown("Sentiment: Negative :angry: ")
                else:
                    st.markdown("Sentiment: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Token Sentiment")

                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    else:
        st.subheader("About")
        st.write("This Sentiment Analysis NLP App aims to unlock sentiment insights from your text data. By leveraging the DistilBERT model, the app can accurately determine the sentiment of any given text, whether it's positive, negative, or neutral.")

if __name__ == '__main__':
    main()
