import tempfile
from pathlib import Path
from fastapi import UploadFile
from .schemas import FaceInfo, VideoAnalysisResult, VerificationResponse
from .services.face_detector import DetectedFace, FaceDetector
from .services.face_verifier import FaceVerifier
from .services.storage import Storage
from .services.processor import VideoProcessor
from .utils import cleanup_temp_files, save_upload_to_temp


ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".mkv"]


class VerificationOrchestrator:

    def __init__(self):
        self.video_processor = VideoProcessor()
        self.face_detector = FaceDetector()
        self.face_verifier = FaceVerifier()
        self.storage = Storage()

    @staticmethod
    def allowed_ext(filename: str) -> bool:
        return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS

    @staticmethod
    def error_response(message: str, warnings: list[str]) -> VerificationResponse:
        warnings.append(message)
        return VerificationResponse(
            same_person_across_videos=False,
            total_videos_processed=0,
            total_faces_detected=0,
            reference_video="",
            summary=message,
            video_results=[],
            warnings=warnings,
        )

    @staticmethod
    def build_summary(
        all_match: bool,
        reference_filename: str,
        results: list[VideoAnalysisResult],
        warnings: list[str],
    ):
        non_ref = [r for r in results if not r.is_reference]
        match_count = sum(1 for r in non_ref if r.match_result)
        total = len(non_ref)

        if all_match:
            return (
                f"The same person was identified across all {total + 1} video(s). "
                f"Reference: '{reference_filename}'. All {total} comparison video(s) matched."
            )
        return (
            f"Identity mismatch detected. "
            f"{match_count}/{total} comparison video(s) matched the reference '{reference_filename}'. "
            + (f"Warnings: {'; '.join(warnings)}" if warnings else "")
        )

    async def run(self, video_files: list[UploadFile]) -> VerificationResponse:
        """
        Full pipeline:
        1. Save uploads to temp
        2. Extract frames per video
        3. Detect faces per video
        4. Select reference from first video
        5. Verify remaining videos against reference
        6. Persist face crops
        7. Build and return structured response
        """
        warnings: list[str] = []
        temp_dir = Path(tempfile.mkdtemp(prefix="fv_upload"))
        temp_paths: list[Path] = []

        try:
            saved_uploads: list[tuple[str, Path]] = []
            for upload in video_files:
                filename = upload.filename
                if not self.allowed_ext(filename):
                    warnings.append(
                        f"'{filename}' has an unsupported extension and was skipped."
                    )
                    continue

                temp_path = await save_upload_to_temp(upload, temp_dir)
                temp_paths.append(temp_path)
                saved_uploads.append((filename, temp_path))

            if not saved_uploads:
                return self._error_response(
                    "No valid video files were provided.", warnings
                )

            # Extract frames + detect faces per video
            video_faces = {}
            video_results = []

            for filename, path in saved_uploads:
                try:
                    self.video_processor.validate_video(path)
                    frames = self.video_processor.extract_frames(path)
                    faces = self.face_detector.detect_faces_in_video_frames(
                        frames, filename
                    )
                except (ValueError, RuntimeError) as exc:
                    warnings.append(f"'{filename}': {exc}")
                    faces = []

                video_faces[filename] = faces
                if not faces:
                    warnings.append(f"No faces detected in '{filename}'.")

            # Select reference
            reference_filename = saved_uploads[0][0]
            reference_faces = video_faces.get(reference_filename, [])
            reference_face = self.face_verifier.select_reference_face(reference_faces)

            if reference_face is None:
                return self._error_response(
                    f"No face detected in the reference video '{reference_filename}'.",
                    warnings,
                )

            # Save reference face crop
            ref_path = self.storage.save_reference_face(reference_face)
            ref_face_info = FaceInfo(
                face_id=reference_face.face_id,
                video_filename=reference_filename,
                frame_number=reference_face.frame_number,
                saved_path=ref_path,
                confidence=reference_face.confidence,
            )

            # Build reference video result
            ref_video_result = VideoAnalysisResult(
                filename=reference_filename,
                faces_detected=len(reference_faces),
                is_reference=True,
                match_result=None,
                similarity_score=None,
                faces=[ref_face_info],
            )
            video_results.append(ref_video_result)

            # Verify remaining videos
            all_match = True
            total_faces = len(reference_faces)

            for filename, _ in saved_uploads[1:]:
                candidate_faces = video_faces.get(filename, [])
                total_faces += len(candidate_faces)

                if not candidate_faces:
                    video_results.append(
                        VideoAnalysisResult(
                            filename=filename,
                            faces_detected=0,
                            is_reference=False,
                            match_result=False,
                            similarity_score=0.0,
                            faces=[],
                        )
                    )
                    all_match = False
                    continue

                face_infos: list[FaceInfo] = []
                video_has_reference = False
                best_matched_face = None
                best_matched_score = -1.0
                best_unmatched_face = None
                best_unmatched_confidence = -1.0

                # Verify each face individually - reference person may appear multiple times
                for candidate_face in candidate_faces:
                    result = self.face_verifier.verify_against_reference(
                        reference_face, [candidate_face]
                    )

                    if result.is_match:
                        # Track best matched face (highest similarity score)
                        video_has_reference = True
                        if result.similarity_score > best_matched_score:
                            best_matched_score = result.similarity_score
                            best_matched_face = candidate_face
                    else:
                        # Track best unmatched face (highest confidence)
                        if candidate_face.confidence > best_unmatched_confidence:
                            best_unmatched_confidence = candidate_face.confidence
                            best_unmatched_face = candidate_face

                # Save only the best matched and best unmatched faces
                if best_matched_face is not None:
                    saved_path = self.storage.save_matched_face(best_matched_face)
                    face_infos.append(
                        FaceInfo(
                            face_id=best_matched_face.face_id,
                            video_filename=filename,
                            frame_number=best_matched_face.frame_number,
                            saved_path=saved_path,
                            confidence=best_matched_face.confidence,
                        )
                    )

                if best_unmatched_face is not None:
                    saved_path = self.storage.save_unmatched_face(best_unmatched_face)
                    face_infos.append(
                        FaceInfo(
                            face_id=best_unmatched_face.face_id,
                            video_filename=filename,
                            frame_number=best_unmatched_face.frame_number,
                            saved_path=saved_path,
                            confidence=best_unmatched_face.confidence,
                        )
                    )

                if not video_has_reference:
                    all_match = False

                video_results.append(
                    VideoAnalysisResult(
                        filename=filename,
                        faces_detected=len(candidate_faces),
                        is_reference=False,
                        match_result=video_has_reference,
                        similarity_score=None,
                        faces=face_infos,
                    )
                )

            # Compose response
            summary = self.build_summary(
                all_match, reference_filename, video_results, warnings
            )

            return VerificationResponse(
                same_person_across_videos=all_match,
                total_videos_processed=len(saved_uploads),
                total_faces_detected=total_faces,
                reference_video=reference_filename,
                summary=summary,
                video_results=video_results,
                warnings=warnings,
            )

        finally:
            cleanup_temp_files(temp_paths)


# async def test_pipeline():
#     print("\nTesting VerificationOrchestrator Pipeline Components")

#     orchestrator = VerificationOrchestrator()
#     test_video_paths = [
#         Path(
#             r"/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Face_Verifacation/face_video/face_video.webm"
#         ),
#         Path(
#             r"/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Face_Verifacation/face_video/face_video1.webm"
#         ),
#         Path(
#             r"/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Face_Verifacation/face_video/face_video2.webm"
#         ),
#     ]

#     # Check if all videos exist
#     missing_videos = [p for p in test_video_paths if not p.exists()]
#     if missing_videos:
#         print(f"Missing videos:")
#         for video in missing_videos:
#             print(f"  - {video}")
#         return

#     try:
#         print(f"Testing with {len(test_video_paths)} videos:\n")
#         for i, video_path in enumerate(test_video_paths, 1):
#             print(f"  {i}. {video_path.name}")

#         print("\nProcessing videos...")

#         # Process each video directly
#         video_faces = {}
#         for filename, path in [(p.name, p) for p in test_video_paths]:
#             print(f"\nProcessing: {filename}")
#             try:
#                 orchestrator.video_processor.validate_video(path)
#                 frames = orchestrator.video_processor.extract_frames(path)
#                 faces = orchestrator.face_detector.detect_faces_in_video_frames(
#                     frames, filename
#                 )
#                 video_faces[filename] = faces
#                 print(f"--Detected {len(faces)} face(s)")
#             except Exception as exc:
#                 print(f"--Error: {exc}")
#                 video_faces[filename] = []

#         # Select reference
#         reference_filename = test_video_paths[0].name
#         reference_faces = video_faces.get(reference_filename, [])
#         reference_face = orchestrator.face_verifier.select_reference_face(
#             reference_faces
#         )

#         if reference_face is None:
#             print(f"\n--No face detected in reference video '{reference_filename}'")
#             return

#         print(f"\n--- Verification Results ---")
#         print(f"Reference video: {reference_filename}")
#         print(f"Reference face ID: {reference_face.face_id}")
#         print(f"Reference confidence: {reference_face.confidence:.2f}")

#         # Save reference face
#         ref_path = orchestrator.storage.save_reference_face(reference_face)
#         print(f"Reference face saved to: {ref_path}")

#         # Verify other videos and save faces
#         all_match = True
#         total_faces = len(reference_faces)

#         print(f"\nVerifying and saving faces:")
#         for filename, _ in [(p.name, p) for p in test_video_paths[1:]]:
#             candidate_faces = video_faces.get(filename, [])
#             total_faces += len(candidate_faces)

#             if not candidate_faces:
#                 print(f"  - {filename}:NO FACES (no match)")
#                 all_match = False
#                 continue

#             # Verify each face individually and track best matched/unmatched
#             video_has_reference = False
#             best_matched_face = None
#             best_matched_score = -1.0
#             best_unmatched_face = None
#             best_unmatched_confidence = -1.0

#             for candidate_face in candidate_faces:
#                 result = orchestrator.face_verifier.verify_against_reference(
#                     reference_face, [candidate_face]
#                 )
#                 status = "MATCH" if result.is_match else "NO MATCH"
#                 print(
#                     f"    - Face ID {candidate_face.face_id} (frame {candidate_face.frame_number}): {status}"
#                 )

#                 if result.is_match:
#                     video_has_reference = True
#                     if result.similarity_score > best_matched_score:
#                         best_matched_score = result.similarity_score
#                         best_matched_face = candidate_face
#                 else:
#                     if candidate_face.confidence > best_unmatched_confidence:
#                         best_unmatched_confidence = candidate_face.confidence
#                         best_unmatched_face = candidate_face

#             # Save only best matched and best unmatched
#             if best_matched_face is not None:
#                 saved_path = orchestrator.storage.save_matched_face(best_matched_face)
#                 print(f"      Saved best matched: {saved_path}")

#             if best_unmatched_face is not None:
#                 saved_path = orchestrator.storage.save_unmatched_face(best_unmatched_face)
#                 print(f"      Saved best unmatched: {saved_path}")

#             if not video_has_reference:
#                 all_match = False

#         print(f"\n--- Final Summary ---")
#         print(f"Videos processed: {len(test_video_paths)}")
#         print(f"Total faces detected: {total_faces}")
#         print(f"Same person confirmed: {all_match}")
#         print(f"Check your outputs folder for saved face images")

#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback

#         traceback.print_exc()


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(test_pipeline())
