import io

from fastapi import FastAPI, Body
from .crawler import *
from .ner import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Welcome to Web Title Tagger Application"}


@app.post("/tagger")
def predict(url: str = Body(..., embed=True)):
    crawler = Crawler()
    title = crawler.get_title(url)
    tags = predict_tags(title)
    return tags