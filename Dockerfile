# Use Python 3.10 slim image to match project requirements
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure model file is reachable at the path expected by the app
RUN if [ -f /app/models/densenet_unweighted.pth ]; then \
    ln -sf /app/models/densenet_unweighted.pth /app/densenet_unweighted.pth; \
    fi

# Expose the port Gradio runs on
EXPOSE 7860

# Run the application
CMD ["python", "-m", "chest_x_ray_classification.app"]