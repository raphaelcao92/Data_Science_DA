#Import
#EDA
import os
from os import path
import json
import pkgutil
import re
from tkinter import N
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
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
import lightgbm
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

def print_word_cloud(group, df_text):
    print(group)
    #Combine all the reviews into one massive string
    review_text = np.array(df_text)
    review_text_combined = " ".join(review for review in review_text)
    # Import image to np.array
    if group == 'group 0':
        mask = np.array(Image.open('icons/dislike.png'))
    else:
        mask = np.array(Image.open('icons/like.png'))
    # Create stopword list:
    stopwords = set(STOPWORDS)
    #For now let's only remove the
    # stopwords.update(["the"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, 
                            background_color="white", 
                            width= 2000, height = 1000, 
                            max_words=50, mask=mask).generate(review_text_combined)

    # Display the generated image:

    image = wordcloud.to_image()
    return image
    
data_0 = data.loc[data['review_score_level'] == 0]
data_1 = data.loc[data['review_score_level'] == 1]

c1 = print_word_cloud('Group Recommended', data_1['review_text'])
c0 = print_word_cloud('Group Not Recommended', data_0['review_text'])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

df = data

df_new = pd.read_csv('df_final.csv', encoding = 'utf-8', index_col = 0)

cr = classification_report(df_new.review_score_level, df_new.preds)

pkl_filename = 'finalized_model'

with open(pkl_filename, 'rb') as file:
    lr_model = pickle.load(file)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#GUI
#Title
st.markdown("<h1 style='text-align: center; color: #339966; '>Data Science Capstone projects </h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #339966; '>Project 3: Sentiment Analysis </h2>", unsafe_allow_html=True)

st.markdown("""<h4 style='text-align: left; color: ;'> Business Objective/Problem 
            <p>- Foody.vn là một kênh phối hợp với các nhà hàng/quán ăn bán thực phẩm online.</p>
            <p>- Chúng ta có thể lên đây để xem các đánh giá, nhận xét cũng như đặt mua thực phẩm.</p>
            <p>- Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/sản phẩm.</p></h4>""", unsafe_allow_html=True)

menu = ["Build Project", "New Prediction"]

choice = st.selectbox('Select one of the options', menu)

if choice == "Build Project" :
    st.title("Build Project")
    st.subheader("EDA & Cleaning")
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
    st.subheader("Natural Language Processing & Machine learning")
    st.markdown("<h4 style='text-align: left; color: #339966; '>Dữ liệu trước khi xử lý</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(10))
    st.markdown("<h4 style='text-align: left; color: #339966; '>Model dự báo được sử dụng là Logistic Regression, với dữ liệu cung cấp như trên, model có độ chính xác như bên dưới:</h4>", unsafe_allow_html=True)
    st.code(cr)
    st.markdown("<h4 style='text-align: left; color: #339966; '>Dữ liệu sau khi được xử lý và đưa ra dự báo</h4>", unsafe_allow_html=True)
    st.dataframe(df_new.head(10))
    st.markdown("<h4 style='text-align: left; color: #339966; '>Recommended Group</h4>", unsafe_allow_html=True)    
    st.image('df_1_img.png')
    st.markdown("<h4 style='text-align: left; color: #339966; '>Not Recommended Group</h4>", unsafe_allow_html=True)
    st.image('df_0_img.png')
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
            lines = lines[0]
            flag = True
        if type == "Input":
            review = st.text_area(label="Input your content:")
            if review!="":
                lines = np.array([review])
                flag = True
            
        if flag:
            st.write("Content:")
            if len(lines)>0:
                st.code(lines)
                x_new = lines
                y_pred_new = lr_model.predict(x_new)
                st.code("New predictions (0: Nhà hàng bạn không được yêu thích, 1: Nhà hàng bạn được yêu thích): " + str(y_pred_new))