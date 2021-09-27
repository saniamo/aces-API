FROM tiangolo/uvicorn-gunicorn:python3.6

# Make directories suited to your application
RUN mkdir -p /home/project/app
WORKDIR /home/project/app

# Copy and install requirements
COPY requirements.txt ..
RUN pip install --no-cache-dir -r ../requirements.txt

# Copy contents from your local to your docker container
COPY ./app .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
