FROM python:3.12-slim

WORKDIR /CreditCardFraud

COPY . /CreditCardFraud

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir catboost==1.2.3 --find-links https://pypi.org/simple/
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "./app.py"]
