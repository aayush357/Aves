from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import time
from PyQt5 import QtCore
from PyQt5.uic import *
import requests
from multithread import MainScreen


class SplashScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('UI/LoadingScreen.ui', self)
        self.ProgressBar.setValue(0)
        ## REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)


        ## DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.frame.setGraphicsEffect(self.shadow)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)

        self.timer.start(35)

        self.show()

    def progress(self):
        self.Loadinglabel.setText(
            "<strong>Loading</strong> Bird prediction api")
        for i in range(50):
            # slowing down the loop
            time.sleep(0.01)
            # setting value to progress bar
            self.ProgressBar.setValue(i)
        res = requests.get(
            'https://bird-species-class-prediction.herokuapp.com/test')
        print(res.text)
        self.Loadinglabel.setText("<strong>Loading</strong> Bird Details api")
        for i in range(51, 101):
            # slowing down the loop
            time.sleep(0.01)
            # setting value to progress bar
            self.ProgressBar.setValue(i)
        res = requests.get('https://aves-detail.herokuapp.com/test')
        print(res.text)
        self.timer.stop()
        self.main = MainScreen()
        self.main.show()
        self.close()


def main():
    app = QApplication(sys.argv)

    splashScreen = SplashScreen()
    app.exec_()


if __name__ == '__main__':
    main()
