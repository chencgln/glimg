import cv2
import sys
sys.path.append(".")

from glimg import visualizer as glvis

image_path = "example/images/img1.jpg"

def test_draw_bbox():
    img = cv2.imread(image_path)
    bbox1 = [100, 100, 200, 200]
    bbox2 = [150, 150, 25, 25]
    img = glvis.draw_bbox(img, bbox1, width=1)
    img = glvis.draw_bbox(img, bbox2, width=2, xywh=True)
    cv2.imshow('image', img)
    cv2.waitKey()

if __name__=="__main__":
    test_draw_bbox()