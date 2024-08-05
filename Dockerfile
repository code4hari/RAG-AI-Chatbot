FROM python:3.12
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8201
ENTRYPOINT ["streamlit", "run"]
CMD ["chatbot.py"]