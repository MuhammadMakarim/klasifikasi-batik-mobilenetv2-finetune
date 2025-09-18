# Use a slim Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "app.py"]