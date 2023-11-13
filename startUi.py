import sys

from PyQt5.QtWidgets import QApplication

from QtUi.MainWindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 256, 256)
    window.show()
    sys.exit(app.exec_())