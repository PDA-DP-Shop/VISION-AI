FROM python:3.9

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /code

# Copy requirements first for caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . /code

# Create necessary directories for file uploads and model saving
# 'static' is used for uploads in app.py
RUN mkdir -p /code/static

# Set permissions so the app can write to these directories
RUN chmod -R 777 /code/static
# Also ensure the application root is writable if the model needs to be saved/updated (app.py lines 39-43)
RUN chmod -R 777 /code

# Expose port 80 for Azure App Service
EXPOSE 80
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]
