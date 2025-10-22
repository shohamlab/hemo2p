# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-10-25 00:18:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-10-25 11:36:34

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_font_filepath():
    ''' 
    Get the path to the font file depending on the OS
    
    :return: path to the font file
    '''
    if platform.system() == 'Darwin':
        font = '/Library/Fonts/Arial Unicode.ttf'
    elif platform.system() == 'Linux':
        font = '/usr/share/fonts/liberation/LiberationMono-Bold.ttf'
    else:
        raise ValueError('Unsupported OS')
    return font


def is_hex_color(color):
    '''
    Check if a string is a valid HEX color value

    :param color: the string to check
    :return: True if the string is a valid HEX color value, False otherwise
    '''
    if color[0] != '#':
        return False
    if len(color) != 7:
        return False
    for c in color[1:].lower():
        if c not in '0123456789abcdef':
            return False
    return True


def parse_color(color):
    ''' 
    Parse input color to a valid RGB tuple

    :param color: the input color, specified either as a CSS4 color name,
        a HEX value or an RGB tuple
    :return: the color as an RGB tuple
    '''
    # If color provided as RGB tuple, check that it contains 3 integers between 0 and 255
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError('Color tuple must have 3 elements')
        for c in color:
            if not isinstance(c, int):
                raise ValueError('Color tuple must contain integers')
            if c < 0 or c > 255:
                raise ValueError('Color tuple elements must be between 0 and 255')

    # If color provided as string, check that it is either a CSS4 color name or a valid HEX value,
    # and convert it to an RGB tuple
    elif isinstance(color, str):
        if color in mcolors.CSS4_COLORS.keys():
            color = mcolors.CSS4_COLORS[color]
        if not is_hex_color(color):
            raise ValueError('String color must be either a CSS4 color name or a valid HEX value')
        color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    
    # If color is neither a tuple nor a string, raise an error
    else:
        raise ValueError(f'Invalid color type: {type(color)}')

    # Return the color as an RGB tuple
    return color


def add_label_to_image(image, label, size=25, color='black', position='top-right'):
    '''
    Add a text label to an image

    :param image: the input image as a numpy array
    :param label: the text label to add
    :param size (optional): the font size (default = 36)
    :param color (optional): the font color, specified either as a CSS4 color name, 
        a HEX value or an RGB tuple (default = 'red')
    :param position (optional): the position of the label (default = 'top-right')
    :return: the modified image as a numpy array
    '''
    # Convert the NumPy array to a Pillow Image
    image_pil = Image.fromarray(image)

    # Create a drawing context
    draw = ImageDraw.Draw(image_pil)

    # Define the text label and its parameters
    font = ImageFont.truetype(get_font_filepath(), size)

    # Connvert color string to HEX value and RGB tuple
    rgb_color = parse_color(color)

    # Get anchor type from position specifier
    pos_y, pos_x = position.split('-')
    try:
        anchor_x = {'left': 'l', 'center': 'm', 'right': 'r'}[pos_x]
    except KeyError:
        raise ValueError(f'Invalid horizontal position specifier: {pos_x}')
    try:
        anchor_y = {'top': 't', 'center': 'm', 'bottom': 'b'}[pos_y]
    except KeyError:
        raise ValueError(f'Invalid vertical position specifier: {pos_y}')
    anchor = anchor_x + anchor_y

    # Calculate text image position from anchor type and image size
    width, height = image_pil.size
    xy_margin = width // 20
    x = {'l': xy_margin, 'm': width / 2, 'r': width - xy_margin}[anchor_x]
    y = {'t': xy_margin, 'm': height / 2, 'b': height - xy_margin}[anchor_y]
    xy = (x, y)
    
    # Add the text label to the image
    draw.text(xy, label, fill=rgb_color, font=font, anchor=anchor)

    # Convert the image back to a NumPy array
    image_with_label = np.array(image_pil)

    # Return the modified image
    return image_with_label


# Main function
if __name__ == '__main__':

    # Generate uniform gray image
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128

    # Add label to the image
    img = add_label_to_image(img, 'STIM')

    # Show the result
    plt.imshow(img)
    plt.show()
