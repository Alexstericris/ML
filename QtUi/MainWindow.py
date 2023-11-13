from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QAction
from QtUi.DrawWidget import DrawingWidget
from support_vector_classifier import SupportVectorClassifier


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Free Draw Example')
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.drawing_widget = DrawingWidget()
        self.layout.addWidget(self.drawing_widget)
        self.setCentralWidget(self.central_widget)
        self.create_menu()

    def create_menu(self):
        classify_action = QAction("Classify", self)
        classify_action.triggered.connect(self.drawing_widget.classify)

        menu_bar = self.menuBar()
        classify_menu = menu_bar.addMenu("Options")
        classify_menu.addAction(classify_action)


