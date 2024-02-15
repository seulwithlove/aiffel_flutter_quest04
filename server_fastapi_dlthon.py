# server_fastapi_dlthon.py
import uvicorn 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 예측 모듈 가져오기
import dlthon_prediction_model

# Create the FastAPI application
app = FastAPI()

# cors issue
origins = ["*"]  # 무슨 역할?

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# A simple example of a GET request
@app.get("/")
async def read_root():
    print("url was requested")
    return "Make an API that uses DLthon model"

@app.get('/sample')
async def sample_prediction():
    result = await dlthon_prediction_model.prediction_model()
    print("predictoin was requrested and done")
    return result




# Run the server
if __name__ == "__main__":
    uvicorn.run("server_fastapi_dlthon:app",
                reload=True,        # Reload the server when cod changes
                host="127.0.0.1",   # Listen on localhost
                port=5000,          # Listen on port 5000
                log_level="info"    # Log level
               )