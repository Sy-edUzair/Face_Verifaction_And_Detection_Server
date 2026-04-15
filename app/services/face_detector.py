import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

logger = logging.getLogger(__name__)

MIN_FACE_CONFIDENCE = 0.5

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
    def __init__(self):
        """Initialize FaceNet model and MTCNN detector."""
        try:
            self.facenet_model = FaceNet()
            logger.info("FaceNet model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load FaceNet model: %s", exc)
            self.facenet_model = None

        try:
            self.mtcnn_detector = MTCNN()
            logger.info("MTCNN detector loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load MTCNN detector: %s", exc)
            self.mtcnn_detector = None

    def get_embedding(self, face_rgb: np.ndarray):
        """
        Generate a face embedding using FaceNet model.
        Input: RGB image (already cropped to face)
        Returns: 128-dimensional FaceNet embedding as list
        """
        if self.facenet_model is None:
            logger.error("FaceNet model not initialized.")
            return []

        try:
            # Ensure image is uint8
            if face_rgb.dtype != np.uint8:
                if face_rgb.max() <= 1:
                    face_rgb = (face_rgb * 255).astype(np.uint8)
                else:
                    face_rgb = face_rgb.astype(np.uint8)

            # Check shape and resize if needed
            if len(face_rgb.shape) == 3 and face_rgb.shape[2] == 3:
                # Resize to 160x160 (FaceNet standard input)
                if face_rgb.shape[:2] != (160, 160):
                    face_rgb = cv2.resize(face_rgb, (160, 160))
                
                # Add batch dimension: (160, 160, 3) -> (1, 160, 160, 3)
                face_batch = np.expand_dims(face_rgb, axis=0)
                
                # Get embedding - returns numpy array of shape (1, 128)
                embeddings = self.facenet_model.embeddings(face_batch)
                
                # Extract first embedding and convert to list
                if embeddings is not None and len(embeddings) > 0:
                    return embeddings[0].tolist()
                else:
                    logger.warning("No embeddings returned from FaceNet model.")
                    return []
            else:
                logger.warning("Face image has unexpected shape: %s", face_rgb.shape)
                return []
        except Exception as exc:
            logger.warning("Failed to generate embedding: %s", exc)
            return []

    @staticmethod
    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        # CLAHE brightness normalization
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Simple white-balance
        result = bgr.astype(np.float32)
        mean_b = np.mean(result[:, :, 0])
        mean_g = np.mean(result[:, :, 1])
        mean_r = np.mean(result[:, :, 2])
        overall_mean = (mean_b + mean_g + mean_r) / 3.0
        if mean_b > 0:
            result[:, :, 0] *= overall_mean / mean_b  # reduce blue
        if mean_g > 0:
            result[:, :, 1] *= overall_mean / mean_g
        if mean_r > 0:
            result[:, :, 2] *= overall_mean / mean_r
        bgr = np.clip(result, 0, 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2)
        bgr = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)

        return bgr

    def detect_faces_in_frame(self, frame, source_video):
        clean = self._preprocess(frame.image)
        rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        if self.mtcnn_detector is None:
            logger.warning(
                "MTCNN detector not initialized; skipping detection on frame %d",
                frame.frame_number,
            )
            return []

        try:
            # MTCNN.detect_faces returns list of dicts with 'box' and 'confidence'
            results = self.mtcnn_detector.detect_faces(rgb)
        except Exception as exc:
            logger.warning(
                "Face detection error on frame %d: %s", frame.frame_number, exc
            )
            return []

        valid_results = [
            r
            for r in results
            if r.get("confidence", 0.0) >= MIN_FACE_CONFIDENCE
            and r.get("box") is not None
        ]

        if not valid_results:
            return []

        # Select best face by box area
        best = max(
            valid_results,
            key=lambda r: r.get("box")[2] * r.get("box")[3],  # width * height
        )

        # Extract face crop from bounding box
        x, y, w, h = best.get("box", [0, 0, 0, 0])
        x, y = max(0, x), max(0, y)
        x_end, y_end = min(rgb.shape[1], x + w), min(rgb.shape[0], y + h)

        face_rgb = rgb[y:y_end, x:x_end]
        if face_rgb.size == 0:
            logger.warning("Empty face crop on frame %d", frame.frame_number)
            return []

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        embedding = self.get_embedding(face_rgb)

        face_area = w * h

        return [
            DetectedFace(
                frame_number=frame.frame_number,
                cropped_image=face_bgr,
                embedding=embedding,
                confidence=float(best.get("confidence", 0.0)),
                source_video=source_video,
                face_area=face_area,
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
