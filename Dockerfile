# Use a Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment.yml file into the container
COPY environment.yml /app/environment.yml

# Create the conda environment from the environment.yml file
RUN conda env create -f /app/environment.yml

# Activate the environment and make it the default
RUN echo "source activate coffeavenv" > ~/.bashrc
ENV PATH /opt/conda/envs/coffeavenv/bin:$PATH

# Set the environment variables
ENV COFFEAHOME=/app
ENV COFFEADATA=/app/data
ENV PATH=$HOME/bin:$PATH

# Copy the rest of the application code into the container
COPY . /app

# Command to run the container
CMD ["bash"]
