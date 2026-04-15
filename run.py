import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.routes import router
from app.app_logging import setup_logging


def create_app():
    setup_logging()

    application = FastAPI(
        title="Face Verification API",
    )
    application.mount(
        "/outputs",
        StaticFiles(directory=str(settings.OUTPUT_DIR)),
        name="outputs",
    )
    application.include_router(router)

    return application


app = create_app()


if __name__ == "__main__":
    print(f"Starting Face Verification API on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
