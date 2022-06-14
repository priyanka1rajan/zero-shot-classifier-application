
import cv2
from dataclasses import dataclass

@dataclass
class OpencvUtils:
    
    image_threshold = 75
    image_maxval = 255
    gaussian_ksize = (5,5)

    def get_max_contour_area(self, frame1, frame2):

        # Process image frame to get contours corresponding to perceived movements relative to frame1
        # You may have to play with the threshold value based on your usecase
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.gaussian_ksize, cv2.BORDER_DEFAULT)
        _, thresh = cv2.threshold(blur, self.image_threshold, self.image_maxval, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the areas for each contour and return max contour size
        contour_areas = list(map(lambda x: cv2.contourArea(x), contours))

        max_contour_area = 0
        if len(contour_areas) > 0:
            max_contour_area = max(contour_areas)

        return max_contour_area

    def draw_label(self, img, text, pos, bg_color):

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.8
        color = (255, 255, 255)
        thickness = cv2.FILLED
        margin = 20

        if len(text) > 30:
            scale = 1.0
        else:
            scale = 1.0

        txt_size = cv2.getTextSize(text, font_face, scale, thickness)

        end_x = int(pos[0] + txt_size[0][0] + margin/2)
        end_y = int(pos[1] - txt_size[0][1] - margin/2)

        pos_x = int(pos[0] - margin/2)
        pos_y = int(pos[1] + margin/2)

        #cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
        cv2.rectangle(img, (pos_x, pos_y), (end_x, end_y), bg_color, thickness)

        cv2.putText(img, text, pos, font_face, scale, color, 4, cv2.LINE_AA)


# In[ ]:




