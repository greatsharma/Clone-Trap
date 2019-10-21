import cv2
import argparse
from utils.vision import draw_image
from utils.vision import image_to_text
from utils.vision import get_4_corner_points, four_point_transformer

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False,
                help='Path to the image')
args = vars(ap.parse_args())

if args['image']:
    img_path = args['image']
else:
    img_path = input('Enter path to the image : ')

img_orginal = cv2.imread(img_path)
draw_image(img_orginal, 'original')

img = img_orginal.copy()

in_coords = get_4_corner_points(img)
warped_img = four_point_transformer(img, in_coords)
draw_image(warped_img, 'bird eye view')
