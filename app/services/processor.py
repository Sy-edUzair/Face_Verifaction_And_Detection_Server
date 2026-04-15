import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_FPS_OUTPUT = 5.0
MAX_VIDEO_SIZE_MB = 100
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".mkv"]


@dataclass
class ExtractedFrame:
    frame_number: int
    image: np.ndarray
    frame_id: str = field(
        default_factory=lambda: f"{int(time.time_ns())}_{np.random.randint(0, 9999):04d}"
    )


class VideoProcessor:
    def validate_video(self, file_path):
        suffix = file_path.suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{suffix}'. "
                f"Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
            )
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            raise ValueError(
                f"Video '{file_path.name}' is {size_mb:.1f} MB, "
                f"exceeding the {MAX_VIDEO_SIZE_MB} MB limit."
            )

    def extract_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path.name}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps:
            fps = 30.0

        logger.info(
            "Video '%s': %d frames @ %.1f fps (reading all)",
            video_path.name,
            total_frames,
            fps,
        )

        frames = []
        frame_idx = 0
        calculated_interval = int(fps / TARGET_FPS_OUTPUT)
        interval = max(1, calculated_interval)

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(ExtractedFrame(frame_number=frame_idx, image=frame))
            frame_idx += interval
            if frame_idx >= total_frames:
                break

        cap.release()
        logger.info("Extracted %d frames from '%s'", len(frames), video_path.name)
        return frames


if __name__ == "__main__":
    processor = VideoProcessor()
    test_video_path = Path(
        "/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Face_Verifacation/face_video/face_video.webm"
    )

    if test_video_path.exists():
        try:
            print(f"\nTesting with: {test_video_path}")
            processor.validate_video(test_video_path)
            print("Video validation passed")
            frames = processor.extract_frames(test_video_path)
            print(f"Successfully extracted {len(frames)} frames")

            for i, frame in enumerate(frames):
                if i <= 30:
                    print(f"Frame {frame.frame_number}: {frame.image.shape}")
                    cv2.imshow(f"Frame {i+1}", frame.image)
                    cv2.waitKey(0)
                else:
                    break

            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"No test video found at {test_video_path}")
        print("To test, provide a video file and update the test_video_path variable")
