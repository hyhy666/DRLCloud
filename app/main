# app/main.py

from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="Cloud Resource Allocation API",
    description="基于 DQN 的云计算资源分配平台接口",
    version="1.0.0"
)

# 挂载 API 路由
app.include_router(router, prefix="/api")

