import uvicorn
from models.semantic_cache import SemanticCache
from fastapi import FastAPI
from pydantic import BaseModel, Field
from config.config import config
from utils.utils import LogUtils

# Create a FastAPI application instance
app = FastAPI()
logger = LogUtils().get_logging()

# default configs
logger.info("storage type {0}".format(config["storage_type"]))
semantic_cache = SemanticCache()


# Define a GET endpoint at the root URL ("/")
@app.get("/")
async def health_check():
    return {"message": "SUCCESS"}


class RequestParams(BaseModel):
    query: str
    forceRefresh: bool


@app.get("/query/")
async def api_data(r: RequestParams):
    logger.info("paramets query {0}".format(r.query))
    logger.info("paramets forceRefresh {0}".format(r.forceRefresh))
    response = semantic_cache.check(r.query, r.forceRefresh)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
