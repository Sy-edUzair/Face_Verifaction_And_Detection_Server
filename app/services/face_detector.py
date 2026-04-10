import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)

FACE_DETECTION_BACKEND = "opencv"
MIN_FACE_CONFIDENCE = 0.5
FACE_VERIFICATION_MODEL: str = "Facenet"  # VGG-Face, Facenet, ArcFace, etc.
VERIFICATION_THRESHOLD: float = 0.6  # Cosine similarity threshold
# Quality thresholds
MIN_FACE_SIZE_PX = 60  # Each side must be at least this many pixels
MIN_BRIGHTNESS = 40.0  # Mean pixel value (0–255); below = too dark
MAX_BRIGHTNESS = 220.0  # Above = overexposed
MIN_SHARPNESS = 50.0  # Laplacian variance; below = too blurry


@dataclass
class DetectedFace:
    face_id: str = field(default_factory=lambda: str(int(time.time_ns())))
    frame_number: int = 0
    cropped_image: np.ndarray = field(default_factory=lambda: np.array([]))
    embedding: list[float] = field(default_factory=list)
    confidence: float = 0.0
    source_video: str = ""
    face_area: int = 0  # bounding box area (w * h)


class FaceDetector:
    def get_embedding(self, face_rgb: np.ndarray):
        """
        Generate a face embedding using the configured model.
        Falls back to VGG-Face if the primary model is unavailable.
        """
        models_to_try = [FACE_VERIFICATION_MODEL]
        # Ensure VGG-Face is always in the fallback chain (bundled with deepface)
        if "VGG-Face" not in models_to_try:
            models_to_try.append("VGG-Face")

        for model_name in models_to_try:
            try:
                result = DeepFace.represent(
                    img_path=face_rgb,
                    model_name=model_name,
                    detector_backend="skip",  # already cropped
                    enforce_detection=False,
                )
                if result:
                    embedding = result[0].get("embedding", [])
                    if embedding:
                        if model_name != FACE_VERIFICATION_MODEL:
                            logger.info(
                                "Using fallback model '%s' for embedding.", model_name
                            )
                        return embedding
            except Exception as exc:
                logger.warning(
                    "Embedding with model '%s' failed: %s. Trying next...",
                    model_name,
                    exc,
                )
        logger.error("All embedding models failed for this face crop.")
        return []

    def detect_faces_in_frame(self, frame, source_video):
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

        try:
            results = DeepFace.extract_faces(
                img_path=rgb,
                detector_backend=FACE_DETECTION_BACKEND,
                enforce_detection=False,
                align=True,
            )
        except Exception as exc:
            logger.warning(
                "Face detection error on frame %d: %s", frame.frame_number, exc
            )
            return []

        valid_results = [
            r
            for r in results
            if r.get("confidence", 0.0) >= MIN_FACE_CONFIDENCE
            and r.get("face") is not None
            and r["face"].size > 0
        ]

        if not valid_results:
            return []

        best = max(
            valid_results,
            key=lambda r: r.get("facial_area", {}).get("w", 0)
            * r.get("facial_area", {}).get("h", 0),
        )

        face_arr = best["face"]
        if face_arr.dtype != np.uint8:
            face_arr = (face_arr * 255).astype(np.uint8)

        face_bgr = (
            cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
            if face_arr.ndim == 3 and face_arr.shape[2] == 3
            else face_arr
        )

        # quality_issues = self._check_quality(face_bgr)
        # if quality_issues:
        #     logger.warning(
        #         "Face discarded on frame %d due to quality issues: %s",
        #         frame.frame_number,
        #         ", ".join(quality_issues),
        #     )
        #     return []

        embedding = self.get_embedding(face_arr)

        # ADDED: extract face area from bounding box
        facial_area = best.get("facial_area", {})
        face_area = facial_area.get("w", 0) * facial_area.get("h", 0)

        return [
            DetectedFace(
                frame_number=frame.frame_number,
                cropped_image=face_bgr,
                embedding=embedding,
                confidence=float(best.get("confidence", 0.0)),
                source_video=source_video,
                face_area=face_area,  # ← ADDED
            )
        ]

    # Run detection across all frames and return all found faces.
    def detect_faces_in_video_frames(self, frames, source_video):
        all_faces = []
        for frame in frames:
            faces = self.detect_faces_in_frame(frame, source_video)
            all_faces.extend(faces)
        logger.info(
            "Detected %d face(s) in '%s' across %d frames",
            len(all_faces),
            source_video,
            len(frames),
        )
        return all_faces

    @staticmethod
    def _check_quality(face_bgr: np.ndarray) -> list[str]:
        """Return a list of quality failure reasons (empty = passed)."""
        issues: list[str] = []
        h, w = face_bgr.shape[:2]

        if h < MIN_FACE_SIZE_PX or w < MIN_FACE_SIZE_PX:
            issues.append(f"too small ({w}x{h}px, min {MIN_FACE_SIZE_PX}px)")

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < MIN_BRIGHTNESS:
            issues.append(f"too dark (brightness={brightness:.1f})")
        elif brightness > MAX_BRIGHTNESS:
            issues.append(f"overexposed (brightness={brightness:.1f})")

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < MIN_SHARPNESS:
            issues.append(f"too blurry (sharpness={sharpness:.1f})")

        return issues


# if __name__ == "__main__":
#     from processor import VideoProcessor

#     detector = FaceDetector()
#     processor = VideoProcessor()

#     test_video_path = Path(
#         r"/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Face_Verifacation/face_video/face_video2.webm"
#     )

#     if test_video_path.exists():
#         try:
#             print(f"\nTesting Face Detector with: {test_video_path.name}")
#             processor.validate_video(test_video_path)
#             print("Video validation passed")
#             frames = processor.extract_frames(test_video_path)
#             print(f"Extracted {len(frames)} frames")

#             detected_faces = detector.detect_faces_in_video_frames(
#                 frames, str(test_video_path.name)
#             )

#             for i, frame in enumerate(detected_faces):
#                 if frame.frame_number >= 50:
#                     cv2.imshow(f"Frame {i+1}", frame.cropped_image)
#                     print(f"Showing frame {i+1}. Press any key to continue...")
#                     cv2.waitKey(0)

#             cv2.destroyAllWindows()

#             for face in detected_faces:
#                 print(f"\nFace detected:")
#                 print(f"Face ID: {face.face_id}")
#                 print(f"Frame: {face.frame_number}")
#                 print(f"Confidence: {face.confidence:.2f}")
#                 print(f"Face Area: {face.face_area}")
#                 print(f"Embedding dim: {len(face.embedding)}")
#             print(f"Detected {len(detected_faces)} face(s) in total")
#         except Exception as e:
#             print(f"Error: {e}")
#     else:
#         print(f"\nNo test video found at {test_video_path}")
#         print("To test, provide a video file and update the test_video_path variable")
