# API
You can test this model on your file via this API. To install the API, you need to install the dependencies first. You can do this by running the following commands:
```bash
git clone https://github.com/OussamaElhamdani/FaceSimNet.git

cd FaceSimNet/api
pip install -r requirements.txt
```
You can then run the API using:
```bash
uvicorn main:app --reload
```
The API will be running on: `http://127.0.0.1:8000` 

FastAPI provides a user friendly environement where you can test requests on: `http://127.0.0.1:8000/docs`
