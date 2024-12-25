from fastapi import FastAPI, WebSocket
import uvicorn
import json
from server.move import move


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)




# expecting: json in the format of `{"pressed" or "released": "forward" or "backward" or "left" or "right"}`
@app.websocket("/move")
async def move_robot(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        move(json.loads(data))
