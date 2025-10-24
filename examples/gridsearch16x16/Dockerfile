# Use the official PyTorch image as the base image
FROM python:3.12

# Set the working directory and importing requirements
WORKDIR /main

# Copy everything from the current directory to the /main dir in the container
COPY . /main

# Change wd to the cloned DeconomiX repo
WORKDIR /main/DeconomiX

# Install package from cloned repo
RUN pip install .

# Set the wd back to main
WORKDIR /main
