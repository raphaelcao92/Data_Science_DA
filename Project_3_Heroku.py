#Import
#EDA
import os
from os import path
import json
import pkgutil
import re
# from tkinter import N
import numpy as np
import pandas as pd
import sqlite3 as sql
import difflib
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import io
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import wordcloud
#NLP & ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from underthesea import word_tokenize,pos_tag,sent_tokenize
import warnings
import string
from wordcloud import WordCloud
import gensim
import jieba
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, classification_report

# %matplotlib inline

#Project

data = pd.read_csv('data_Foody.csv', encoding = 'utf-8', index_col = 0)

##### EDA & Cleaning #####

buffer = io.StringIO()

profile = ProfileReport(data, title="Pandas Profiling Report")

# 10 restaurant có rating cao nhất
rating_hi = data.groupby('restaurant') \
                    .agg({'restaurant':'count', 'review_score':'mean'}) \
                    .rename(columns={'restaurant':'count_restaurant', 'review_score':'mean_review_score'}) \
                    .reset_index() \
                    .sort_values(by='mean_review_score', ascending=False)
                                        
# 10 restaurant số lượt rating nhiều nhất:
review_times = data.groupby('restaurant') \
                    .agg({'restaurant':'count', 'review_score':'mean'}) \
                    .rename(columns={'restaurant':'count_restaurant', 'review_score':'mean_review_score'}) \
                    .reset_index() \
                    .sort_values(by='count_restaurant', ascending=False)

# Creating review_score_level column with level from 0-10
data.loc[ (data['review_score'] >= 0) & (data['review_score'] <= 1.4), 'review_score_level'] = 1
data.loc[ (data['review_score'] > 1.4) & (data['review_score'] <= 2.4), 'review_score_level'] = 2
data.loc[ (data['review_score'] > 2.4) & (data['review_score'] <= 3.4), 'review_score_level'] = 3
data.loc[ (data['review_score'] > 3.4) & (data['review_score'] <= 4.4), 'review_score_level'] = 4
data.loc[ (data['review_score'] > 4.4) & (data['review_score'] <= 5.4), 'review_score_level'] = 5
data.loc[ (data['review_score'] > 5.4) & (data['review_score'] <= 6.4), 'review_score_level'] = 6
data.loc[ (data['review_score'] > 6.4) & (data['review_score'] <= 7.4), 'review_score_level'] = 7
data.loc[ (data['review_score'] > 7.4) & (data['review_score'] <= 8.4), 'review_score_level'] = 8
data.loc[ (data['review_score'] > 8.4) & (data['review_score'] <= 9.4), 'review_score_level'] = 9
data.loc[ (data['review_score'] > 9.4) & (data['review_score'] <= 10), 'review_score_level'] = 10

review_scoreItem_level = data.groupby('review_score_level') \
                            .agg({'review_score_level':'count', 'review_score':'mean'}) \
                            .rename(columns={'review_score_level':'count_review_score_level', 'review_score':'mean_review_score'}) \
                            .reset_index() \
                            .sort_values(by='count_review_score_level', ascending=False)

# Visualization
ax1 = sns.set_style(style=None, rc=None )
fig, ax1 = plt.subplots(figsize=(6,4))
sns.barplot(x='review_score_level', y='count_review_score_level', data=review_scoreItem_level, palette='cubehelix', ax=ax1)
ax2 = ax1.twinx()
fig = sns.lineplot(data=review_scoreItem_level['mean_review_score'], ax=ax2)

# Creating review_score_level column with level with 0 & 1
data.loc[ (data['review_score'] >= 0) & (data['review_score'] <= 6.4), 'review_score_level'] = 0 # Not recommended
data.loc[ data['review_score'] >= 6.5, 'review_score_level'] = 1 # Recommended

#Word Cloud

def print_word_cloud(df_text):
    #Combine all the reviews into one massive string
    review_text = np.array(df_text)
    review_text_combined = " ".join(review for review in review_text)
    # Create stopword list:
    stopwords = set(STOPWORDS)
    #For now let's only remove the
    # stopwords.update(["the"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, 
                            background_color="white", 
                            width= 2000, height = 1000, 
                            max_words=50).generate(review_text_combined)

    # Display the generated image:

    image = wordcloud.to_image()
    return image
    
data_0 = data.loc[data['review_score_level'] == 0]
data_1 = data.loc[data['review_score_level'] == 1]

c1 = print_word_cloud(data_1['review_text'])
c0 = print_word_cloud(data_0['review_text'])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

df = data

df_new = pd.read_csv('df_final.zip', encoding = 'utf-8', index_col = 0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NLP

# Chúng ta sẽ chuẩn bị các stopwords tiếng Việt, emoji, teencode, từ sai, tiếng Anh để xử lý các văn bản hiệu quả.

#VietNamese Stop Words

STOP_WORD_FILE = 'files/vietnamese-stopwords.txt'

with open(STOP_WORD_FILE,'r',encoding='utf-8') as file:
  stop_words=file.read()

stop_words = stop_words.split('\n')

#EMOJI

EMOJI_CON_FILE = 'files/emojicon.txt'

with open(EMOJI_CON_FILE,'r',encoding='utf-8') as file:
  emoji=file.read()

emoji = emoji.split('\n')

emoji_dict = {}

for line in emoji:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
    
#TEENCODE

TEEN_CODE_FILE = 'files/teencode.txt'

with open(TEEN_CODE_FILE,'r',encoding='utf-8') as file:
  teencode=file.read()

teencode = teencode.split('\n')

teencode_dict = {}

for line in teencode:
    key, value = line.split('\t')
    teencode_dict[key] = str(value)

#WRONG WORD

WRONG_WORDS_FILE = 'files/wrong-word.txt'

with open(WRONG_WORDS_FILE,'r',encoding='utf-8') as file:
  wrongwords=file.read()

wrongwords = wrongwords.split('\n')

#ENG TO VN

EV_FILE = 'files/english-vnmese.txt'

with open(EV_FILE,'r',encoding='utf-8') as file:
  e2v=file.read()

e2v = e2v.split('\n')

e2v_dict = {}

for line in e2v:
    key, value = line.split('\t')
    e2v_dict[key] = str(value)

def process_text(text, emoji_dict, teencode_dict, e2v_dict, wrongwords):
    document = text.lower()
    document = document.replace("'","")
    document = regex.sub(r'\.+','.',document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        #EMOJI
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        #TEENCODE
        sentence = ' '.join(teencode_dict[word] if word in teencode_dict else word for word in sentence.split())
        #ENGLISH
        sentence = ' '.join(e2v_dict[word] if word in e2v_dict else word for word in sentence.split())
        #Wrong words
        sentence = ' '.join('' if word in wrongwords else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    document = new_sentence
    #print(doc)
    #DELETE exceed blank space
    document = regex.sub(r'\s+',' ',document).strip()
    return document
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

cr = classification_report(df_new.review_score_level, df_new.preds)

pkl_filename = 'final_model.pkl'

with open(pkl_filename, 'rb') as file:  
    lr_model = pickle.load(file)

pkl_count = "count_model.pkl" 

with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#GUI
#Title
st.markdown("<h1 style='text-align: center; color: #339966; '>Data Science Capstone projects </h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #339966; '>Project 3: Sentiment Analysis </h2>", unsafe_allow_html=True)

st.markdown("""<h4 style='text-align: left; color: ;'> Business Objective/Problem 
            <p>- Foody.vn là một kênh phối hợp với các nhà hàng/quán ăn bán thực phẩm online.</p>
            <p>- Chúng ta có thể lên đây để xem các đánh giá, nhận xét cũng như đặt mua thực phẩm.</p>
            <p>- Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/sản phẩm.</p></h4>""", unsafe_allow_html=True)

menu = ["EDA & Cleaning","Natural Language Processing & Machine learning", "New Prediction"]

choice = st.selectbox('Select one of the options', menu)

if choice == "EDA & Cleaning" :
    st.title("EDA & Cleaning")
    #EDA & Cleaning
    st.code('Dataframe') # Dataframe
    st.dataframe(data.head(10))
    st.code('Pandas profiling') # Pandas profiling
    st_profile_report(profile)
    st.code('10 restaurant có rating cao nhất')
    st.table(rating_hi.head(10))
    st.code('10 restaurant số lượt review nhiều nhất')
    st.table(review_times.head(10))
    st.markdown("<h4 style='text-align: left; color: #339966; '>Nhận xét:</h4>", unsafe_allow_html=True)
    st.markdown(""" <p>- Giá trị chấm điểm trải đều từ 0-10, các điểm số đều là số thực => Quy về số nguyên để có thể thực hiện thuật toán Classification.</p>
                <p>- Có đến hơn 12k nhà hàng khác nhau trong 39k review => trung bình vào 3 review có chữ cho 1 nhà hàng.</p>
                <p>- Đa phần điểm số trung bình từ các bài review đều mức khá (7 điểm) trở lên. [~75%]</p>
                <p>- Chúng sẽ không thực hiện loại trùng tại bài toán này tránh trường hợp các nhận xét ngắn gọn tương tự nhau.</p>
                <p>- Các nhà hàng có điểm số cao nhất chưa chắc hẳn sẽ có số lượt review cao nhất, nhưng các nhà hàng có lượng review cao lại đa phần nằm tại khoảng mean. => Popularity sẽ quan trọng hơn rating.</p>""", unsafe_allow_html=True)
    st.code('Chuyển đổi các điểm số từ số thực thành số nguyên')
    ax1 = sns.set_style(style=None, rc=None )
    fig, ax1 = plt.subplots(figsize=(6,4))
    sns.barplot(x='review_score_level', y='count_review_score_level', data=review_scoreItem_level, palette='cubehelix', ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=review_scoreItem_level['mean_review_score'], ax=ax2)
    st.pyplot(fig=fig, showPyplotGlobalUse = False)
    st.markdown("<h4 style='text-align: left; color: #339966; '>Nhận xét:</h4>", unsafe_allow_html=True)
    st.markdown("""<p>- Sau khi chuyển đổi 1 lần chúng ta sẽ dễ dàng thấy rằng các nhà hàng phân hóa thành 2 loại rõ rệt:</p>
                <p>- Điểm >= 7 : Tốt</p>
                <p>- Điêm < 7: Không tốt</p>
                <p>=> Chúng ta sẽ chuyển đổi lại 1 lần nữa.</p>""", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left; color: #339966; '>Recommended Group</h4>", unsafe_allow_html=True)    
    st.image(c1)
    st.markdown("<h4 style='text-align: left; color: #339966; '>Not Recommended Group</h4>", unsafe_allow_html=True)    
    st.image(c0)
    st.markdown("<h4 style='text-align: left; color: #339966; '>Nhận xét:</h4>", unsafe_allow_html=True)
    st.markdown(""" <p>- Nhóm được yêu thích chủ yếu sẽ có ngợi khen ở 3 điểm chính: Không gian, nhân viên, phục vụ. Cùng với từ như: sạch sẽ, thân thiện, ngon.</p>
                    <p>- Nhóm các quán không được yêu thích cũng có 3 điểm chính tương tự nhưng các từ đi cùng lại rất khác: thất vọng, bình thường, hết, nhưng.</p>
                    <p>=>> Chúng ta hãy chờ sau khi có kết quả dự đoán để đưa ra các đề nghị cho các chủ doanh nghiệp.</p>""", unsafe_allow_html=True)
if choice == "Natural Language Processing & Machine learning" :    
    st.title("Natural Language Processing & Machine learning")
    st.markdown("<h4 style='text-align: left; color: #339966; '>Dữ liệu trước khi xử lý</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(10))
    st.markdown("<h4 style='text-align: left; color: #339966; '>Model dự báo được sử dụng là Logistic Regression, với dữ liệu cung cấp như trên, model có độ chính xác như bên dưới:</h4>", unsafe_allow_html=True)
    st.code(cr)
    st.markdown("<h4 style='text-align: left; color: #339966; '>Dữ liệu sau khi được xử lý và đưa ra dự báo</h4>", unsafe_allow_html=True)
    st.dataframe(df_new.head(10))
    st.markdown("<h4 style='text-align: left; color: #339966; '>Recommended Group</h4>", unsafe_allow_html=True)    
    st.image('df_1_img.PNG')
    st.markdown("<h4 style='text-align: left; color: #339966; '>Not Recommended Group</h4>", unsafe_allow_html=True)
    st.image('df_0_img.PNG')
    st.markdown("<h4 style='text-align: left; color: #339966; '>Nhận xét:</h4>", unsafe_allow_html=True)
    st.markdown(""" <p>- Có thể thấy rằng kết quả dự báo có các key word gần sát so với khi kiểm tra Woud Cloud ban đầu =>>> Các chủ nhà hàng quán để ý đến 2 điểm chính: Nhân viên và phục vụ.</p>""", unsafe_allow_html=True)


elif choice == "New Prediction":
    st.title("New Prediction")
    st.subheader('Select data')
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options = ('Upload', 'Input'))
    if type == "Upload":
        #Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type = ['txt','csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(line.columns)
            lines = lines['text']
            flag = True
    if type == "Input":
        review = st.text_area(label="Input your content:")
        if review!="":
            lines = np.array([review])
            # lines = st.dataframe({'text':[review]})
            flag = True
        
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            d = {'text':lines}
            lines = pd.DataFrame(data=d)
            lines['text'] = lines['text'].str.lower()
            lines['text'] = lines['text'].str.replace('[\d+]',' ')
            lines['text'] = lines['text'].str.replace('[{}]'.format(string.punctuation), ' ')
            lines['text'] = lines['text'].str.replace("['•','\n','-','≥','±','–','…','_']",' ') 
            lines['text'] = lines['text'].str.replace('(\s[a-z]\s)',' ')
            lines['text'] = lines['text'].str.replace('(\s+)',' ')
            lines['text'] = lines['text'].apply(lambda x: process_text(str(x), emoji_dict, teencode_dict, e2v_dict, wrongwords))
            lines['text'] = lines['text'].apply(lambda x:word_tokenize(x,format = 'text'))
            text_data = np.array(lines['text'])
            # count = CountVectorizer()
            # count.fit(text_data)
            bag_of_words = count_model.transform(text_data)
            x_new = bag_of_words.toarray()
            y_pred_new = lr_model.predict(x_new)
            st.code("New predictions (0: Nhà hàng bạn không được yêu thích, 1: Nhà hàng bạn được yêu thích): " + str(y_pred_new))
