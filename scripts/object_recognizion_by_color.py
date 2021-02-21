import numpy as np
import cv2

cap = cv2.VideoCapture(0)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 100)

cv2.createTrackbar('Hue Minimo', 'image', 9, 255, lambda: None)
cv2.createTrackbar('Hue Maximo', 'image', 255, 255, lambda: None)

cv2.createTrackbar('Saturation Minimo', 'image', 99, 255, lambda: None)
cv2.createTrackbar('Saturation Maximo', 'image', 255, 255, lambda: None)

cv2.createTrackbar('Value Minimo', 'image', 98, 255, lambda: None)
cv2.createTrackbar('Value Maximo', 'image', 255, 255, lambda: None)


def printObj(c, img_, title):
    x, y, w, h = cv2.boundingRect(c)
    # rectagle
    cv2.rectangle(img_, (x+5, y-5), (x+w+10, y+h-10), (229, 0, 255), 3)
    # title
    coordX = x+round(w/2)
    coordY = y+round(h/2)
    cv2.putText(img_, f'{title}', (x, y-10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (250, 250, 255), 2, cv2.LINE_AA)
    # Countour
    cv2.drawContours(img_, [c], 0, [0, 255, 0], 2, cv2.LINE_AA)


def get_obj(img, obj_color_range):
    lower = np.array(obj_color_range['lower'])
    upper = np.array(obj_color_range['upper'])
    # Gray color
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('########################')
    print(f'{lower}')
    print(f'{upper}')

    # cretae mask for desired object
    mask = cv2.inRange(hsv, lower, upper)
    # Adjust image for remove noise
    kernel = np.ones((11, 11), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Get countour
    countours, _ = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours = sorted(
        countours, key=lambda c: cv2.contourArea(c), reverse=True)

    if len(countours) > 0:
        printObj(countours[0], img, 'Pollito')

    cv2.imshow('original', img)


def adjust_colors_range():
    # Adjust values to get the desired object
    hMin = cv2.getTrackbarPos('Hue Minimo', 'image')
    hMax = cv2.getTrackbarPos('Hue Maximo', 'image')
    vMin = cv2.getTrackbarPos('Saturation Minimo', 'image')
    vMax = cv2.getTrackbarPos('Saturation Maximo', 'image')
    sMin = cv2.getTrackbarPos('Value Minimo', 'image')
    sMax = cv2.getTrackbarPos('Value Maximo', 'image')
    return {
        'lower': [hMin, vMin, sMin],
        'upper': [hMax, vMax, sMax]
    }


def nothing():
    pass


def run(objects):

    while True:
        _, frame = cap.read()

        obj_color_range = adjust_colors_range()
        get_obj(frame, obj_color_range)
        # cv2.imwrite('./../imgs/frame.jpg', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    objects = {
        'pollito': {'lower': [9, 99, 122], 'upper': [255, 255, 255]}
    }
    run(objects)
