## Goal
To demonstrate a simple way to deploy a trained ML model as a REST api within a Docker container. It uses
* FAST api
* Docker for deployment in a Docker container
* pyenv and virtualenv
* Python 3.11.0


<br/>

To build the Docker image (from within the project dir)  
`docker build -t lang-detector-mlapi .`  
To run a container from the above image  
`docker run -p 8000:8000 lang-detector-mlapi`