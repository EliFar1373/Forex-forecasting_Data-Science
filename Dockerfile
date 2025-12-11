# Use official Python image
FROM python:3.12-slim

# Install system dependencies needed by yfinance + networking tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    tzdata \
    curl \
    iputils-ping \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# Set timezone (important for Yahoo date ranges)
ENV TZ=UTC


# Set working directory inside container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .



# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose port if using streamlit web app
EXPOSE 8504

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8504", "--server.address=0.0.0.0"]









