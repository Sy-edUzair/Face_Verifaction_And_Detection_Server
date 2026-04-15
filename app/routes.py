import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from .config import settings
from .schemas import HealthResponse, VerificationResponse
from .pipeline import VerificationOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check():
    """Returns server health status and version."""
    return HealthResponse(
        status="ok",
        message="Face Verification API is running.",
    )


@router.post(
    "/verify",
    response_model=VerificationResponse,
    tags=["Verification"],
)
async def verify_faces(
    videos: Annotated[
        list[UploadFile],
        File(
            description="Two or more video files. The first video is used as the reference."
        ),
    ],
):
    """
    - **videos**: The first file is treated as the **reference** video.
      All subsequent files are compared against it.

    Returns a JSON response indicating whether the same person appears
    across all videos, along with per-video details and saved face crops.
    """
    if not videos:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one video file must be provided.",
        )

    if len(videos) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least two video files are required: one reference and one comparison.",
        )

    logger.info("Received %d video(s) for verification.", len(videos))

    orchestrator = VerificationOrchestrator()
    try:
        result = await orchestrator.run(videos)
    except Exception as exc:
        logger.exception("Unhandled error during verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(exc)}",
        )

    return result
