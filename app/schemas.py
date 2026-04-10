from typing import Optional
from pydantic import BaseModel, Field


class FaceInfo(BaseModel):
    face_id: str = Field()  # Unique identifier for this face crop
    video_filename: str = Field()  # Source video file name
    frame_number: int = Field()  # Frame number where the face was detected
    saved_path: str = Field()  # Relative path to the saved face image
    confidence: Optional[float] = Field(None)


class VideoAnalysisResult(BaseModel):
    filename: str
    faces_detected: int
    is_reference: bool = False
    match_result: Optional[bool] = Field(
        None
    )  # True = same person as reference, False = different, None = reference video
    similarity_score: Optional[float] = Field(None)
    faces: list[FaceInfo] = []


class VerificationResponse(BaseModel):  # endpoint response schema

    same_person_across_videos: bool = (
        Field()
    )  # true if all non-reference videos contain the same person as the reference
    total_videos_processed: int
    total_faces_detected: int
    reference_video: str = Field()  # Filename of the video used as reference
    summary: str = Field()  # Human-readable summary of findings
    video_results: list[VideoAnalysisResult]


class HealthResponse(BaseModel):
    status: str
    version: str
    message: str
