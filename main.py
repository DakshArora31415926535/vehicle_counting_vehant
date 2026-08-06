import subprocess
import sys
import os

class Solution:
    def __init__(self):
        pass

    def _run_worker(self, video_path, angle_frames):
        result = subprocess.check_output(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "worker.py"),
                str(angle_frames),
                video_path
            ],
            text=True
        )
        return int(result.strip())

    def forward(self, video_path: str) -> int:
        """
        Args:
            video_path (str): Path to input traffic video

        Returns:
            int: Total vehicle count
        """

        c100 = self._run_worker(video_path, 100)
        c500 = self._run_worker(video_path, 500)

        # relative difference
        diff_ratio = abs(c100 - c500) / max(c100, c500)

        # 🔑 updated rule
        if diff_ratio > 0.50:
            final_count = max(c100, c500)
        else:
            final_count = min(c100, c500)

        return final_count
