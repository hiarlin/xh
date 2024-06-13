import streamlit as st
import pandas as pd
import random
import torch
from fastai.collab import load_learner

# 加载模型
learn = load_learner('joke.pkl')

# 加载笑话数据
jokes_df = pd.read_excel('Dataset4JokeSet.xlsx')
jokes_df.columns = ['joke']
jokes_df = jokes_df.rename_axis('joke_id').reset_index()

# 生成随机笑话函数
def get_random_jokes(n=3):
    return jokes_df.sample(n).reset_index(drop=True)

# 获取推荐笑话函数
def recommend_jokes(user_ratings, n=5):
    user_id = max(user_ratings['user_id']) + 1  
    new_data = pd.DataFrame({'user_id': [user_id]*len(jokes_df), 'joke_id': jokes_df['joke_id'], 'joke': jokes_df['joke']})
    dls = learn.dls.test_dl(new_data)
    preds, _ = learn.get_preds(dl=dls)
    new_data['rating'] = preds
    new_data = new_data.sort_values(by='rating', ascending=False).head(n)
    return new_data.merge(jokes_df, on='joke_id')

# 初始界面
st.title('想要点开心果吗')
st.header("笑话推荐")

if 'user_ratings' not in st.session_state:
    st.session_state['user_ratings'] = pd.DataFrame(columns=['user_id', 'joke_id', 'rating'])

if 'random_jokes' not in st.session_state:
    st.session_state['random_jokes'] = get_random_jokes()

st.subheader('请对以下笑话进行评分（1-5分）')
user_id = 1  # 假设当前用户ID为1

for i, joke in st.session_state['random_jokes'].iterrows():
    rating = st.slider(f'笑话{i+1}: {joke["joke"]}', 1, 5, key=f'rating_{i}')
    if st.button(f'提交评分 笑话{i+1}', key=f'button_{i}'):
        new_rating = pd.DataFrame({'user_id': [user_id], 'joke_id': [joke['joke_id']], 'rating': [rating]})
        st.session_state['user_ratings'] = pd.concat([st.session_state['user_ratings'], new_rating], ignore_index=True)
        st.write('评分已提交！')

if st.button('推荐笑话'):
    recommended_jokes = recommend_jokes(st.session_state['user_ratings'])
    st.session_state['recommended_jokes'] = recommended_jokes

if 'recommended_jokes' in st.session_state:
    st.subheader('推荐的笑话')
    total_rating = 0
    for i, joke in st.session_state['recommended_jokes'].iterrows():
        rating = st.slider(f'推荐笑话{i+1}: {joke["joke"]}', 1, 5, key=f'rec_rating_{i}')
        total_rating += rating
    if len(st.session_state['recommended_jokes']) > 0:
        satisfaction = total_rating / len(st.session_state['recommended_jokes'])
        st.write(f'用户满意度： {satisfaction:.2f} / 5')