"""
FlowerArt | Cannlytics AI
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/29/2022
Updated: 8/10/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Programmatically create flower art given effects and aromas.
    E.g. skunk -> green heat wave coming off of flower

Resources:

    - NFT Image Generator
    URL: https://github.com/benyaminahmed/nft-image-generator/blob/main/generate.ipynb

    - Turn Photos into Cartoons Using Python
    URL: <https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e>

"""

# Standard imports.
from typing import Any, Optional

# External imports.
from bs4 import BeautifulSoup
import cv2
import numpy as np
import requests
import urllib.parse


class FlowerArt():
    """Create cannabis art and cannabis strain NFTs."""

    def __init__(self,
        line_size = 7,
        blur_value = 7,
        number_of_filters = 10,
        total_colors = 9,
        sigmaColor = 200,
        sigmaSpace = 200,
    ) -> None:
        """Initialize the FlowerArt client.
        Args:
            line_size (int): The width of the lines to draw.
            blur_value (int): The degree to which to blur the image.
            number_of_filers (int): The number of filters to apply.
                Generally use 5 for fast to 10 for slow rendering.
            total_colors (int): The maximum number of colors.
            sigmaColor (int): The distortion of color, > 200 for cartoon.
            sigmaSpace (int): The distortion of space, > 200 for cartoon.
        """
        self.line_size = line_size
        self.blur_value = blur_value
        self.number_of_filters = number_of_filters
        self.total_colors = total_colors
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
    
    def cartoonize_image(
            self,
            filename: str,
            outfile: str,
            grayscale: Optional[bool] = False,
            convert_colors: Optional[bool] = False,
            show: Optional[bool] = False,
        ) -> Any:
        """Create a NFT for a given strain given a representative image.
        Combine edge mask with the colored image.
        Apply bilateral filter to reduce the noise in the image.
        This blurs and reduces the sharpness of the image.
        Args:
            filename (str): The image file to use as a model.
            outfile (str): The image file to create.
            grayscale (bool): Whether to convert to grayscale, False by default.
            convert_colors (bool): Whether to convert the colors, False by default.
            show (bool): Whether or not to show the image, False by default.
        Returns:
            (Mat): The image matrix data.
        """
        img = cv2.imread(filename)
        edges = self.edge_mask(img, self.line_size, self.blur_value)
        if grayscale:
            img = self.color_quantization(img, self.total_colors)
        if convert_colors:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blurred = cv2.bilateralFilter(
            img,
            d=self.number_of_filters,
            sigmaColor=self.sigmaColor,
            sigmaSpace=self.sigmaSpace,
        )
        cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
        cv2.imwrite(outfile, cartoon)
        if show:
            cv2.imshow('image', cartoon)
            cv2.waitKey()
        return img
    
    def color_quantization(self, img, k: int):
        """Reduce the color palette, because a drawing has fewer colors
        than a photo. Color quantization is performed by the K-Means
        clustering algorithm of OpenCV.
        Args:
            img (Mat): The image matrix data.
            k (int): The degree of the K-means clustering algorithm.
        Returns:
            (Mat): The image matrix data.
        """
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result
    
    def edge_mask(self, img, line_size, blur_value):
        """Create an edge mask, emphasizing the thickness of the edges
        to give a cartoon-style to the image.
        Args:
            img (Mat): The image matrix data.
            line_size (int): The width for the image lines.
            blur_value (int): The degree to which to blur the image.
        Returns:
            (Mat): The matrix data for the edges of the image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
        return edges

    def get_color_association(self, string: str) -> str:
        """Get a color associated with a given word or phrase.
        The algorithm uses Colorize, a tool that uses a search engine to
        find image results for a word or phrase, and then calculates the
        average color across approximately 25 image results.
        Credit: Alex Beals
        URL: https://alexbeals.com/projects/colorize/
        Args:
            string (str): Text to create an association with.
        Returns:
            (str): A color hex code.
        """
        base = 'https://alexbeals.com/projects/colorize/search.php?q='
        url = base + urllib.parse.quote_plus(string)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features='html.parser')
        a = soup.find_all('span', {'class': 'hex'})
        return a[0].text
