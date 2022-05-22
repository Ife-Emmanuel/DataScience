from email.policy import default
import cv2
import numpy as np
from decouple import config

def nothing(x):
    pass

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype= np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=4)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img



# image = cv2.imread(r'images\ela_original.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
def process(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (15.3, 238.7),
        (227.0, 238.7),
        (301.7, 59.5),
        (249.1, 38.1)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image,
                   np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image,
                            rho= 6,
                            theta= np.pi/60,
                            threshold= 50,
                            lines= np.array([]),
                            minLineLength= 40,
                            maxLineGap= 100
                            )
    print(lines)
    if lines is not None:
        image_with_lines = draw_the_lines(image, lines)
        return image_with_lines
    else:
        return image

#cap = cv2.VideoCapture('videos\cars.avi')
capture_path = config('capture_path', default=0)
cap = cv2.VideoCapture(capture_path)
# cap = cv2.VideoCapture(0)
_, frame = cap.read()
cv2.imshow('moving cars', frame)
# plt.imshow(frame)
# plt.show()
cv2.waitKey(3) & 0xFF
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r'videos\Background_Subtraction_Tutorial_frame.mp4')

while cap.isOpened():
    _, frame = cap.read()
    #print(frame, ' \n   \n THE END OF THIS FRAME !!!!! \n')
    if frame is None:
        continue
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
else:
    print('No video captured.')

cap.release()
cv2.destroyAllWindows()