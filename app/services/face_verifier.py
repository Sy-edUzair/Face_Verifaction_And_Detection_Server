import logging
from dataclasses import dataclass
import cv2
import numpy as np
from .face_detector import DetectedFace

logger = logging.getLogger(__name__)

VERIFICATION_THRESHOLD = 0.72
REFERENCE_THRESHOLD = 0.85


@dataclass
class VerificationResult:
    is_match: bool
    similarity_score: float
    best_face: DetectedFace | None


class FaceVerifier:
    # sharpness scoring using Laplacian variance
    @staticmethod
    def blur_score(image: np.ndarray):
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        a = np.array(vec_a, dtype=np.float64)
        b = np.array(vec_b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def select_reference_face(self, faces: list[DetectedFace]):
        if not faces:
            return None

        # Find the face from the earliest frame (lowest frame number)
        earliest_face = min(faces, key=lambda f: f.frame_number)

        # Among all faces similar to that earliest face,pick the sharpest one as the actual reference
        candidates = []
        for face in faces:
            if not face.embedding or not earliest_face.embedding:
                continue
            score = self.cosine_similarity(earliest_face.embedding, face.embedding)
            if score >= REFERENCE_THRESHOLD:
                candidates.append(face)

        if not candidates:
            return earliest_face

        # From matching candidates, return the sharpest
        return max(candidates, key=lambda f: self.blur_score(f.cropped_image))

    def verify_against_reference(self, reference, candidate_faces: list[DetectedFace]):
        if not reference.embedding:
            logger.warning("Reference face has no embedding; cannot verify.")
            return VerificationResult(
                is_match=False, similarity_score=0.0, best_face=None
            )

        if not candidate_faces:
            return VerificationResult(
                is_match=False, similarity_score=0.0, best_face=None
            )

        best_score = -1.0
        best_face = None

        for face in candidate_faces:
            if not face.embedding:
                continue
            score = self.cosine_similarity(reference.embedding, face.embedding)
            if score > best_score:
                best_score = score
                best_face = face

        is_match = best_score >= VERIFICATION_THRESHOLD
        logger.debug(
            "Best similarity for '%s': %.4f (threshold=%.2f) → %s | All scores: %s",
            candidate_faces[0].source_video if candidate_faces else "?",
            best_score,
            VERIFICATION_THRESHOLD,
            "MATCH" if is_match else "NO MATCH",
            [
                round(self.cosine_similarity(reference.embedding, f.embedding), 4)
                for f in candidate_faces
                if f.embedding
            ],
        )
        return VerificationResult(
            is_match=is_match,
            similarity_score=round(best_score, 4),
            best_face=best_face,
        )


# if __name__ == "__main__":
#     from pathlib import Path
#     from processor import VideoProcessor
#     from face_detector import FaceDetector

#     verifier = FaceVerifier()
#     detector = FaceDetector()
#     processor = VideoProcessor()

#     test_video_path = Path(
#         r"/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Face_Verifacation/face_video/face_video1.webm"
#     )

#     if test_video_path.exists():
#         try:
#             print(f"\nTesting Face Verifier with: {test_video_path.name}")
#             processor.validate_video(test_video_path)
#             print("Video validation passed")
#             frames = processor.extract_frames(test_video_path)
#             print(f"Extracted {len(frames)} frames")
#             detected_faces = detector.detect_faces_in_video_frames(
#                 frames, str(test_video_path.name)
#             )
#             print(f"Detected {len(detected_faces)} face(s)")

#             if len(detected_faces) >= 2:
#                 reference_face = verifier.select_reference_face(detected_faces)
#                 print(
#                     f"\nSelected reference face (ID: {reference_face.face_id}, "
#                     f"Confidence: {reference_face.confidence:.2f}, "
#                     f"Area: {reference_face.face_area})"
#                 )

#                 print(f"\nVerification Results:")
#                 for i, face in enumerate(detected_faces):
#                     if face.face_id == reference_face.face_id:
#                         cv2.imshow(
#                             f"Reference Face (ID: {face.face_id})", face.cropped_image
#                         )
#                         cv2.waitKey(0)
#                         print(f"Face {i+1}: Reference face (skipped)")
#                         continue

#                     result = verifier.verify_against_reference(reference_face, [face])
#                     status = "MATCH" if result.is_match else "NO MATCH"
#                     print(
#                         f"Face {i+1}: {status} (Similarity: {result.similarity_score:.4f})"
#                     )
#             else:
#                 print(
#                     f"\nNeed at least 2 faces for verification. Found: {len(detected_faces)}"
#                 )

#         except Exception as e:
#             print(f"Error: {e}")
#     else:
#         print(f"\nNo test video found at {test_video_path}")
#         print("To test, provide a video file and update the test_video_path variable")
