ARG GRADIO_ACOUSTIC
FROM python:3.9.6

LABEL maintainer="Rida Saghir ridasagheer94@gmail.com"

# Set the working directory
WORKDIR /gradio_app

RUN chmod 777 /gradio_app
RUN mkdir /demo
RUN chmod 777 /demo

# Copy the current directory contents into the container
COPY . /gradio_app
ADD requirements.txt /gradio_app/requirements.txt


# Copy the Linux version of frpc into the container
#COPY frpc_linux_aarch64_v0.2 /usr/local/lib/python3.9/site-packages/gradio/frpc_linux_aarch64_v0.2

# Install dependencies
RUN pip install -r requirements.txt

#RUN apt-get update && apt-get install -y libsndfile1

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user

# Set the working directory to the user's home directory
WORKDIR $HOME/app


# Make port available to the world outside this container
EXPOSE 7860/tcp

# Define environment variable
ENV NAME Acoustic

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Run your application
CMD ["python3", "app.py"]



