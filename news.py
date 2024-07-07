import streamlit as st
from newspaper import Article
from textblob import TextBlob
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from textblob.classifiers import NaiveBayesClassifier

text_classified = [
    
    ('Scientists discover new species in the Amazon', 'Science'),
    ('Global warming impacts on Arctic ice observed', 'Science'),
    ('Tech giant launches new AI-powered device', 'Technology'),
    ('Stock market reaches all-time high', 'Finance'),
    ('New health regulations to be implemented next year', 'Health'),
    ('Astronauts return safely from the International Space Station', 'Science'),
    ('Hollywood actor accused of tax evasion', 'Entertainment'),
    ('Government announces new infrastructure projects', 'Politics'),
    ('Sports team wins championship after 10 years', 'Sports'),
    ('Breakthrough in cancer research reported', 'Health'),
    ('New software update improves user experience', 'Technology'),
    ('Artificial intelligence reshaping industries worldwide', 'Technology'),
    ('Large-scale protests erupt in major city over new law', 'Politics'),
    ('Upcoming elections to determine future leadership', 'Politics'),
    ('Serial killer apprehended after string of murders', 'Crime'),
    ('Investigation underway in brutal murder case', 'Crime'),
    ('Government officials implicated in corruption scandal', 'Corruption'),
    ('High-profile corruption case trial begins', 'Corruption'),
    ('Public outcry over handling of rape case in city', 'Crime'),
    ('Allegations of rape against public figure surface', 'Crime'),
    ('Local BSP chief killed: 8 suspects detained; party workers block city road', 'Crime'),
    ('Corruption charges against state officials spark public outrage', 'Corruption'),
    ('Political leaders clash over controversial new policy', 'Politics'),
    ('Actor accused of sexual assault faces public backlash', 'Entertainment'),
    ('Climate change summit concludes with new global agreements', 'Environment'),
    ('New breakthrough in renewable energy technology announced', 'Technology'),
    ('SpaceX successfully launches mission to Mars', 'Science'),
    ('Educational reforms proposed to improve student outcomes', 'Education'),
    ('Healthcare system faces challenges amid COVID-19 pandemic', 'Health'),
    ('Art exhibit featuring renowned artists opens in major city', 'Arts'),
    ('Cybersecurity threats on the rise, experts warn', 'Technology'),
    ('Local community celebrates annual cultural festival', 'Culture'),
    ('New archaeological discovery sheds light on ancient civilization', 'History'),
    ('Startup company disrupts traditional industry with innovative approach', 'Business'),
    ('International trade negotiations reach a critical stage', 'Economics'),
    ('Social media platform introduces new privacy features', 'Technology'),
    ('New President aims for improved relations with neighboring countries', 'Politics'),
    ('President\'s policies and challenges in governance system', 'Politics'),
    ('Impact of diplomatic efforts and regional stability', 'Politics'),
    ('Reforms in national laws and societal implications', 'Politics'),
    ('Advocacy for minority rights and cultural diversity', 'Politics'),
    ('Economic strategies under new administration', 'Economics'),
    ('Diplomatic efforts and regional stability', 'Politics'),
    ('Cultural reforms and artistic freedoms in contemporary society', 'Culture'),
    ('Technological advancements in defense capabilities', 'Technology'),
    ('Celebrity scandal rocks industry as new allegations surface', 'Entertainment'),
    ('Major earthquake strikes Pacific region, causing widespread damage', 'Environment'),
    ('New breakthrough drug shows promise in treating rare disease', 'Health'),
    ('Artificial intelligence surpasses human capabilities in study', 'Technology'),
    ('Summit focuses on cybersecurity threats and solutions', 'Technology'),
    ('Political unrest escalates amid economic downturn', 'Politics'),
    ('Blockchain technology revolutionizes finance industry', 'Technology'),
]

classifier = NaiveBayesClassifier(text_classified)

custom_css = """
    <style>
        @font-face {
            font-family: 'Roboto';
            src: url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
        }
        body {
            font-family: 'Roboto', sans-serif;
        }
    </style>
"""

def summarize(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    
    if article.text.strip():  
        try:
            language = detect(article.text)
            return article.title, article.authors, article.publish_date, language, article.keywords, article.summary, article.text
        except LangDetectException as e:
            print(f"Language detection error: {e}")
            return article.title, article.authors, article.publish_date, "Unknown", article.keywords, article.summary, article.text
    else:
        print("No valid text extracted.")
        return None, None, None, None, None, None, None

def main():
    st.title("News Article Summarization")
    st.markdown(custom_css, unsafe_allow_html=True)
    url = st.text_input("Enter the URL")
    
    if st.button("Summarize"):
        if url:
            title, authors, publish_date, lang, keywords, summary, text = summarize(url)
            
            st.subheader("Article Information")
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Authors:** {', '.join(authors) if authors else 'Not available'}")
            st.markdown(f"**Publish Date:** {publish_date}")
            st.markdown(f"**Language:** {lang}")
            
            st.subheader("Summary")
            st.markdown(f"{summary}")
            
            st.subheader("Keywords")
            st.markdown(f"{', '.join(keywords)}")
            
            st.subheader("Sentiment")
            analysis = TextBlob(summary)
            sentiment = "Positive" if analysis.polarity > 0 else "Negative" if analysis.polarity < 0 else "Neutral"
            st.markdown(f"{sentiment}")
            
            st.subheader("Classification")
            classification = classifier.classify(text)
            st.markdown(f"{classification}")
            
        else:
            st.warning("Please enter a valid URL.")
            
if __name__ == '__main__':
    main()
