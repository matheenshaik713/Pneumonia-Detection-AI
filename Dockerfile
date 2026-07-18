FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "10000"]