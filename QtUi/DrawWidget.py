import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap
from PyQt5.QtWidgets import QWidget
from PyQt5.uic.properties import QtGui

from support_vector_classifier import SupportVectorClassifier


class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.path = []
        self.pressing=False

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(10)
        pen.setColor(QColor(0, 0, 0))
        painter.setPen(pen)

        for i in range(len(self.path) - 1):
            painter.drawLine(self.path[i], self.path[i + 1])

    def mousePressEvent(self, event):
        self.pressing=True
        self.path.append(event.pos())

    def mouseReleaseEvent(self, event) -> None:
        self.pressing=False

    def mouseMoveEvent(self, event):
        if self.pressing:
            self.path.append(event.pos())
            self.update()

    def get_drawing_image(self):
        # Get the size of the drawing widget
        size = self.size()

        # Create a grayscale matrix with the same size as the drawing widget
        image_matrix = np.zeros((size.height(), size.width()), dtype=np.uint8)

        # Draw the path onto the matrix
        image = QImage(image_matrix.data, size.width(), size.height(), QImage.Format_Grayscale8)
        painter = QPainter(image)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(255, 255, 255))  # White color for drawing
        painter.setPen(pen)
        for i in range(len(self.path) - 1):
            painter.drawLine(self.path[i], self.path[i + 1])
        painter.end()

        pixmap = QPixmap.fromImage(image)

        # Resize QPixmap to 28x28
        pixmap = pixmap.scaled(28, 28)

        # Convert QPixmap to grayscale QImage
        gray_image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)

        # Convert grayscale QImage to numpy array
        width = gray_image.width()
        height = gray_image.height()
        buffer = gray_image.bits().asarray(width * height)
        image_matrix = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width))

        return image_matrix

    def classify(self):
        # Implement your classification logic here
        image = np.array([self.get_drawing_image()]).reshape(1,-1)

        svc = SupportVectorClassifier.load_model('models/svcmodel')
        print(svc.predict(image))