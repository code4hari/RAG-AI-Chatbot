import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import openai
import psycopg2
from datetime import datetime
import hashlib
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("quickstart")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
openai.api_key = OPENAI_API_KEY

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def query_database(query):
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_list = query_embedding.numpy()[0].tolist()
    results = index.query(vector=query_embedding_list, top_k=15, include_metadata=True)
    relevant_content = [result['metadata']['content'] for result in results['matches']]
    return relevant_content

def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT NOT NULL
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        "user" TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )''')
    conn.commit()
    cursor.close()
    conn.close()

def authenticate_user(username, password):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, hash_password(password)))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user is not None

def save_chat_history(user, query, response):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE user = %s", (user,))
    count = cursor.fetchone()[0]
    if count >= 5:
        cursor.execute("DELETE FROM chat_history WHERE id = (SELECT id FROM chat_history WHERE user = %s ORDER BY timestamp ASC LIMIT 1)", (user,))
    cursor.execute("INSERT INTO chat_history (\"user\", query, response) VALUES (%s, %s, %s)", (user, query, response))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_chat_history(user):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT query, response FROM chat_history WHERE user = %s ORDER BY timestamp ASC", (user,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def answer_query(user, query):
    chat_history = fetch_chat_history(user)
    relevant_content = query_database(query)
    context = '\n'.join(relevant_content)
    for q, a in chat_history:
        context += f"\nUser: {q}\nAssistant: {a}"
    system_prompt = "You are a helpful assistant, providing accurate and concise answers based on the given context. In every answer include the page number, the document name your solution came from, and the link of the document your solution came from and should be in the following format, (document names, page numbers, links). If there are multiple documents being used make sure to provide the links of all the documents."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Given the following context:\n\n{context}\n\nAnswer the following question:\n\n{query}"}
    ]
    max_tokens = 500
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1.0,
    )
    answer = response.choices[0].message['content'].strip()
    save_chat_history(user, query, answer)
    return answer

st.set_page_config(page_title="Chatbot", layout="wide")

init_db()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'query_input' not in st.session_state:
    st.session_state['query_input'] = ''

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'user' not in st.session_state:
    st.session_state['user'] = ''

def login():
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['user'] = username
            st.success("Login successful")
            st.rerun()  
        else:
            st.error("Invalid username or password")

def ask_question():
    if st.session_state.query_input:
        answer = answer_query(st.session_state['user'], st.session_state.query_input)
        st.session_state.chat_history.insert(0, (st.session_state.query_input, answer))
        st.session_state.query_input = ""

def logout():
    st.session_state['logged_in'] = False
    st.session_state['user'] = ''
    st.session_state['chat_history'] = []

if not st.session_state['logged_in']:
    st.subheader("Login")
    login()
else:
    st.sidebar.button("Logout", on_click=logout)
    query = st.text_input("Enter your question", key="query_input")
    if st.button("Ask", on_click=ask_question):
        st.rerun()
    if st.session_state.chat_history:
        for q, a in (st.session_state.chat_history):
            st.markdown(f"**User:** {q}")
            st.markdown(f"**Assistant:** {a}")
