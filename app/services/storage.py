import logging
from pathlib import Path
import cv2

from config import settings
from services.face_detector import DetectedFace

logger = logging.getLogger(__name__)


class Storage:
    def _save_face(self, face: DetectedFace, directory: Path, label):
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{label}_{face.face_id}_frame{face.frame_number}.jpg"
        save_path = directory / filename

        if face.cropped_image is not None and face.cropped_image.size > 0:
            cv2.imwrite(str(save_path), face.cropped_image)
            logger.debug("Saved face crop: %s", save_path)
        else:
            logger.warning("Empty image for face %s; skipping save.", face.face_id)

        return str(save_path)

    def save_reference_face(self, face: DetectedFace):
        return self._save_face(face, settings.REFERENCE_DIR, label="ref")

    def save_matched_face(self, face: DetectedFace):
        return self._save_face(face, settings.MATCHED_DIR, label="match")

    def save_unmatched_face(self, face: DetectedFace):
        return self._save_face(face, settings.UNMATCHED_DIR, label="nomatch")

    def save_u_and_m_face(self, face: DetectedFace, is_match: bool):
        if is_match:
            return self.save_matched_face(face)
        return self.save_unmatched_face(face)
