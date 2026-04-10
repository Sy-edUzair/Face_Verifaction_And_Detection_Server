from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.routes import router
from app.app_logging import setup_logging
import uvicorn


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
    application.include_router(router, prefix="/api/v1")

    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
