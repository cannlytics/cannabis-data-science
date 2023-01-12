"""
Remove Silence from a Video
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 1/7/2022
Updated: 1/7/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Vivek Haldar <https://github.com/vivekhaldar>
    Donald Feury  <https://gitlab.com/dak425/scripts/-/blob/master/trim_silenceV2>

Command-line Usage:

    python video_cut D:\\videos\\video.mp4 D:\\videos\\edited-video.mp4
"""
# Standard imports:
import math
from typing import Optional

# External imports:
from moviepy.editor import VideoFileClip, concatenate_videoclips


def find_speaking(
        audio_clip,
        window_size: Optional[float] = 0.1,
        volume_threshold: Optional[float] = 0.01,
        ease_in: Optional[float] = 0.25,
    ) -> list:
    """
    Iterate over a video's audio to find the non-silent parts. Outputs a
    list of (speaking_start, speaking_end) intervals.
    Args:
        audio_clip: The audio being used.
        window_size: (in seconds) hunt for silence in windows of this size.
        volume_threshold: volume below this threshold is considered to be silence.
        ease_in: (in seconds) add this much silence around speaking intervals.
    """
    # First, iterate over audio to find all silent windows.
    num_windows = math.floor(audio_clip.end/window_size)
    window_is_silent = []
    for i in range(num_windows):
        s = audio_clip.subclip(i * window_size, (i + 1) * window_size)
        v = s.max_volume()
        window_is_silent.append(v < volume_threshold)

    # Find speaking intervals.
    speaking_start = 0
    speaking_end = 0
    speaking_intervals = []
    for i in range(1, len(window_is_silent)):
        e1 = window_is_silent[i - 1]
        e2 = window_is_silent[i]
        # silence -> speaking
        if e1 and not e2:
            speaking_start = i * window_size
        # speaking -> silence, now have a speaking interval
        if not e1 and e2:
            speaking_end = i * window_size
            new_speaking_interval = [speaking_start - ease_in, speaking_end + ease_in]
            # With tiny windows, this can sometimes overlap the previous window, so merge.
            need_to_merge = len(speaking_intervals) > 0 and speaking_intervals[-1][1] > new_speaking_interval[0]
            if need_to_merge:
                merged_interval = [speaking_intervals[-1][0], new_speaking_interval[1]]
                speaking_intervals[-1] = merged_interval
            else:
                speaking_intervals.append(new_speaking_interval)

    # Return the intervals with speaking.
    return speaking_intervals


def video_cut(
        video_file: str,
        edited_file: str,
        temp_audiofile: Optional[str] = 'temp-audio.m4a',
    ) -> None:
    """Edit silence out of a video."""

    # Find intervals to keep.
    vid = VideoFileClip(video_file)
    intervals = find_speaking(vid.audio)
    keep_clips = [
        vid.subclip(max(start, 0), end) for [start, end] in intervals
    ]

    # Edit the video.
    edited_video = concatenate_videoclips(keep_clips)
    edited_video.write_videofile(
        edited_file,
        fps=60,
        preset='ultrafast',
        codec='mpeg4',
        temp_audiofile=temp_audiofile,
        remove_temp=True,
        audio_codec='aac',
        threads=6,
    )
    vid.close()

# === Usage ===
if __name__ == '__main__':

    # Command-line usage:
    # import sys
    # video_cut(sys.argv[1], sys.argv[2])

    # Example:
    # import os
    # data_dir = 'D:\\cannabis-data-science\\videos'
    # video_file = 'Cannabis Data Science (2022-12-28 13_22 GMT-8).mp4'
    # edited_file = 'cannabis-data-science-2022-12-28-cut.mp4'
    # video_cut(os.path.join(data_dir, video_file), os.path.join(data_dir, edited_file))

    # Example:
    import os
    from cannlytics.utils import kebab_case
    data_dir = 'D:\\cannabis-data-science\\videos'
    output_dir = os.path.join(data_dir, 'cut')
    video_files = os.listdir(os.path.join(data_dir, 'pending'))
    video_files.reverse()
    for video_file in video_files:
        edited_file = kebab_case(video_file).replace('-mp-4', '.mp4')
        audio_file = edited_file.replace('mp4', 'm4a')
        video_cut(
            os.path.join(data_dir, 'pending', video_file),
            os.path.join(data_dir, 'cut', edited_file),
            os.path.join(data_dir, 'audio', audio_file),
        )
