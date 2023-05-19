import sys
import cv2
import pytesseract
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PIL import Image


# running pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()

        # title
        title = "Haar Cascade license plate detector"
        self.setWindowTitle(title)

        # set fixed size of the windows
        self.setFixedSize(640, 480)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.import_button = QPushButton('Import Image', self)
        self.import_button.clicked.connect(self.import_image)
        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_image)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.import_button)
        layout.addWidget(self.submit_button)

    def import_image(self):
        global file_name
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg)')
        if file_name:
            self.image = QPixmap(file_name)
            self.image_label.setPixmap(self.image)
        print(file_name)

    def submit_image(self):

        # Read input image
        img = cv2.imread(file_name)

        # convert input image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # read haarcascade for number plate detection
        cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

        # Detect license number plates
        plates = cascade.detectMultiScale(gray, 1.2, 5)

        if (len(plates) == '0'):
            print('Failed to detect license plate')
        else:
            print('Number of detected license plates:', len(plates))

        # loop over all plates
        for (x, y, w, h) in plates:

            # draw bounding rectangle around the license number plate
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cropped = gray[y:y+h, x:x+w]
            color_plates = img[y:y+h, x:x+w]

            # Try to read the license plate region with different config
            psm7 = pytesseract.image_to_string(color_plates, config='--psm 7')
            psm3 = pytesseract.image_to_string(
                color_plates, config='-l eng --oem 1 --psm 3')
            psm11 = pytesseract.image_to_string(
                color_plates, config='--psm 11')
            noConfig = pytesseract.image_to_string(color_plates).strip()

            if psm11 == "":
                psm11 = "Fail"
            if psm3 == "":
                psm3 = "Fail"
            if psm7 == "":
                psm7 = "Fail"
            if noConfig == "":
                noConfig = "Fail"

            print("Car Plate prediction:")
            print("PSM3 : ", psm3.strip())
            print("PSM7 : ", psm7.strip())
            print("PSM11 : ", psm11)
            print("0Config : ", noConfig)

            # small window (Show the cropped number plate)
            croppedImage = cv2.resize(cropped, (200, 200))

            cv2.imshow('Number Plate', croppedImage)
            cv2.imshow('Number Plate Image', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
