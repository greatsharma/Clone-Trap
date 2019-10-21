import cv2
import numpy
import PyPDF2
import imutils
import pytesseract


def draw_image(img, title=''):
    """ Draw images

    Parameters
    ----------
    img : numpy ndarray
        Image to display

    title : str, optional
        Title of the output window (default is '')
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def order_coords(coords: numpy.array) -> numpy.array:
    """ Order the coordinates in the order tl, tr, br, bl

    Parameters
    ----------
    coords : numpy array
        numpy array of size (4,2) consisting coordinates in any order

    Returns
    -------
    ordered_coords : numpy array
        A numpy array of size (4,2) consisting coordinates in the order tl, tr, br, bl
    """
    ordered_coords = numpy.empty(shape=(4, 2), dtype='float32')

    coord_sum = coords.sum(axis=1)  # sum along rows
    ordered_coords[0] = coords[numpy.argmin(coord_sum)]
    ordered_coords[2] = coords[numpy.argmax(coord_sum)]

    coord_diff = numpy.diff(coords, axis=1)  # diff along rows
    ordered_coords[3] = coords[numpy.argmax(coord_diff)]
    ordered_coords[1] = coords[numpy.argmin(coord_diff)]

    return ordered_coords


def four_point_transformer(image, coords):
    """Takes image and return a top-down bird eye view of the image

    Parameters
    ---------
    image : image
        An image of any valid format like png, jpg etc.

    coords : numpy array
        A numpy array of the coordinates of four corners of the image 

    Returns
    -------
    warped_img : numpy ndarray
        A warped image in top-down bird eye view
    """
    input_coords = order_coords(coords)
    tl, tr, br, bl = input_coords

    width_top = numpy.sqrt(((tl[0] - tr[0]) ** 2) + ((tl[1] - tr[1]) ** 2))
    width_bottom = numpy.sqrt(((bl[0] - br[0]) ** 2) + ((bl[1] - br[1]) ** 2))

    out_width = max(int(width_top), int(width_bottom))

    height_left = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    out_height = max(int(height_left), int(height_right))

    out_coords = numpy.array([[0, 0], [out_width-1, 0], [out_width-1, out_height-1],
                              [0, out_height-1]], dtype='float32')

    transform_matrix = cv2.getPerspectiveTransform(input_coords, out_coords)
    warped_img = cv2.warpPerspective(image, transform_matrix,
                                     dsize=(out_width, out_height))

    return warped_img


def edge_detector(image, w_ratio=0.75, h_ratio=0.75, kernel=(5, 5)):
    """Edge detector using canny kernel

    Parameters
    ----------
    image : image
        An image of any valid format like png, jpg etc

    w_ratio : float, optional
        Width ratio of input to output image (default is 0.75)

    h_ratio : float, optional
        Height ratio of input to output image (default is 0.75)

    kernel : tuple, optional
        kernel size for (default is (5,5))

    Returns
    -------
    edged_img : numpy ndarray
        A canny edged image
    """
    copied_img = image.copy()
    copied_img = cv2.resize(copied_img, None, fx=w_ratio, fy=h_ratio)

    gray_img = cv2.cvtColor(copied_img, cv2.COLOR_BGR2GRAY)
    # gaussian bluring to remove high frequency noise
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edged_img = cv2.Canny(gray_img, threshold1=70, threshold2=200)

    return edged_img


def get_contour(edged_img):
    """Returns the contour of the paper point object in the image

    Parameters
    ----------
    edged_img : numpy ndarray
        A canny edged image

    Returns
    -------
    approx : numpy ndarray
        Approximated contour of the 4 point object in the image 
    """
    contours = cv2.findContours(edged_img, mode=cv2.RETR_LIST,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, (0.05 * peri), True)

        if len(approx) == 4:
            return approx

    return None


def get_4_corner_points(img):
    """Returns the 4 corner points of the object in the image

    Parameters
    ----------
    img : numpy ndarray
        Input image

    Returns
    -------
    corner_points : numpy ndarray
        4 corner points of the object in the image
    """
    edged_img = edge_detector(img)
    screen_cnt = get_contour(edged_img).reshape(4, 2)
    corner_points = screen_cnt / 0.75

    return corner_points.astype(int)


def image_to_text(img):
    """Extracts text from the given image

    Parameters
    ----------
    img : numpy.ndarray
        Input image

    Returns
    -------
    img_c : numpy.ndarray
        Final transformed image
    text : str
        Text in the image
    """
    # copy image
    img_c = img.copy()
    # convert to gray scale
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    # removes noise from grayscale image
    img_c = cv2.fastNlMeansDenoising(img_c)
    # gaussian bluring to remove high frequency noise
    img_c = cv2.GaussianBlur(img_c, (3, 3), 0)
    # sharp the image
    kernel_sharp_3 = numpy.array([[-2, -2, -2],
                                  [-2, 17, -2],
                                  [-2, -2, -2]])
    img_c = cv2.filter2D(img_c, -1, kernel_sharp_3)
    # threshold image to two values 0(black) and 255(white)
    _, img_c = cv2.threshold(img_c, thresh=120, maxval=255,
                             type=cv2.THRESH_BINARY)

    config = ('-l eng + equ --oem 1 --psm 3')
    text = pytesseract.image_to_string(img_c, config=config)

    return img_c, text


def pdf_to_text(pdf_path: str) -> str:
    """Extracts text from the pdf

    Parameters
    ----------
    pdf_path : str
        Path to the pdf file

    Returns
    -------
    text : str
        Text in the pdf
    """
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)

    page_count = pdf_reader.getNumPages()

    text = ''
    for i in range(page_count):
        page = pdf_reader.getPage(i)
        text += page.extractText()

    pdf_file.close()

    return text
