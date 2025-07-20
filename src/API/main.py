from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import api

description = """
This is a REST API for the OAMK Certificate Verification System.
"""

app = FastAPI(
    title="API",
    description=description,
    version="0.1.0",
)
tags_metadata = [
    {
        "name": "student",
        "description": "API for the student.",
    },
    {
        "name": "reviewer",
        "description": "API for the reviewer.",
    },
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)
