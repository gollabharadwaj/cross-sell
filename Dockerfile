FROM python:3.8.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the Python dependencies
RUN pip install -r requirements.txt

# Expose the desired port
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "model_app:app", "--host", "0.0.0.0", "--port", "80"]
