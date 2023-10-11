"""
Remove Image Background
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 9/25/2023
Updated: 9/25/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Command-line example:

    ```
    python tools/remove_bg.py image_path output_path
    ```

Note: Ensure you have the `rembg` libraries installed.
"""

# Standard imports:
import sys

# External imports:
from rembg import remove
from PIL import Image


def remove_bg(input_path: str, output_path: str) -> None:
    """Convert a video file to another video with a transparent background.
    
    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to save the output image with a transparent background.
    """
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)


# === Test ===
if __name__ == '__main__':

    # Get the input video path and output video path
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_bg(input_file, output_file)

    # Example: Cutout all images in a directory.
    # import os
    # image_dir = r''
    # images = os.listdir(image_dir)
    # for image in images:
    #     input_path = os.path.join(image_dir, image)
    #     output_path = os.path.join(image_dir, image.split('.')[0] + '-cutout.png')
    #     remove_bg(input_path, output_path)
    #     print('Cutout:', output_path)
