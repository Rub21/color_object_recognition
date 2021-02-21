import numpy as np
import cv2

cap = cv2.VideoCapture(0)


def printObj(img, c, title, area):
    x, y, w, h = cv2.boundingRect(c)
    # rectagle
    cv2.rectangle(img, (x + 5, y - 5), (x + w + 10, y + h - 10), (0, 255, 0), 3)
    # title
    coordX = x + round(w / 2)
    coordY = y + round(h / 2)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(
        img,
        f"{title}",
        (x + round(w / 2), y - 10),
        font,
        0.5,
        (250, 250, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"({x},{y})",
        (x - 50, y - 10),
        font,
        0.5,
        (229, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"({x},{y+h})",
        (x - 50, y + h),
        font,
        0.5,
        (229, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"({x+w},{y})",
        (x + w, y),
        font,
        0.5,
        (229, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"({x+w},{y+h})",
        (x + w, y + h),
        font,
        0.5,
        (229, 0, 255),
        1,
        cv2.LINE_AA,
    )

    # Countour
    cv2.drawContours(img, [c], 0, [0, 255, 0], 2, cv2.LINE_AA)


#


def get_obj(
    img,
    title,
    obj_color_range_lower,
    obj_color_range_upper,
    obj_area,
    obj_kernel,
):
    lower = np.array(obj_color_range_lower)
    upper = np.array(obj_color_range_upper)

    # Gray color
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # cretae mask for desired object
    mask = cv2.inRange(hsv, lower, upper)

    # Adjust image for remove noise
    kernel = np.ones(obj_kernel, np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Get countour
    countours, _ = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    countours = sorted(
        countours, key=lambda c: cv2.contourArea(c), reverse=True
    )

    for c in countours:
        area = cv2.contourArea(c)
        if area > obj_area[0] and area < obj_area[1]:
            printObj(img, c, title, area)

def adjust_colors_range():
    # Adjust values to get the desired object
    hMin = cv2.getTrackbarPos("Hue Minimo", "image")
    hMax = cv2.getTrackbarPos("Hue Maximo", "image")
    vMin = cv2.getTrackbarPos("Saturation Minimo", "image")
    vMax = cv2.getTrackbarPos("Saturation Maximo", "image")
    sMin = cv2.getTrackbarPos("Value Minimo", "image")
    sMax = cv2.getTrackbarPos("Value Maximo", "image")
    # print('########################')
    print(f"{hMin, vMin, sMin}")
    print(f"{hMax, vMax, sMax}")
    return {"lower": [hMin, vMin, sMin], "upper": [hMax, vMax, sMax]}


def run(objects):
    while True:
        _, frame = cap.read()
        obj_color_range = adjust_colors_range()
        for key, value in objects.items():
            get_obj(
                frame,
                key,
                value["lower"],
                value["upper"],
                value["area"],
                value["kernel"],
            )
            # get_obj(
            #     frame,
            #     key,
            #     obj_color_range["lower"],
            #     obj_color_range["upper"],
            #     value["area"],
            #     value["kernel"],
            # )

        cv2.imshow("original", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 600, 100)


cv2.createTrackbar("Hue Minimo", "image", 9, 255, lambda: None)
cv2.createTrackbar("Hue Maximo", "image", 255, 255, lambda: None)

cv2.createTrackbar("Value Minimo", "image", 55, 255, lambda: None)
cv2.createTrackbar("Value Maximo", "image", 255, 255, lambda: None)

cv2.createTrackbar("Saturation Minimo", "image", 47, 255, lambda: None)
cv2.createTrackbar("Saturation Maximo", "image", 255, 255, lambda: None)


if __name__ == "__main__":
    objects = {
        "pollito": {
            "lower": [9, 99, 122],
            "upper": [255, 255, 255],
            "area": [5000, 10000],
            "kernel": (15, 15),
        },
        "hipo": {
            "lower": [31, 35, 8],
            "upper": [138, 145, 113],
            "area": [4000, 6000],
            "kernel": (21, 21),
        },
        "hand": {
            "lower": [9, 47, 55],
            "upper": [255, 255, 255],
            "area": [20000, 500000],
            "kernel": (15, 15),
        },
    }
    run(objects)
