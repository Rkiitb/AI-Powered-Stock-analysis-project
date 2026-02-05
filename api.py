import fastapi
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
from typing import Annotated
import pickle
import uuid
from graph import workflow 

app=FastAPI()


class val_input(BaseModel):
    query:Annotated[str,Field(...,description="write your query.",min_length=2,max_length=100,examples=['Todays business news'])]



@app.get('/')
def status():
    return {'Health':'ok'}


@app.post('/predict')
def predicts(query:val_input):
    thread_id = str(uuid.uuid4())
    CONFIG = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    response=workflow.invoke({'user_query':query},config=CONFIG)
    return JSONResponse(status_code=200,content={'content': response['final_result']})


