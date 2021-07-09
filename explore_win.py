from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUi
from PyQt5 import QtCore
global_State=0
class Explore(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('UI/explore_window.ui',self)
        # Removing windows title bar
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # adding function to close button
        self.closeButton.clicked.connect(lambda: self.close())
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        self.titleBarFrame.mouseMoveEvent = self.moveWindow
        self.maximizeButton.clicked.connect(lambda: self.maximize_restore())

    def maximize_restore(self):
        global global_State
        status=global_State

        if status==0:
            self.showMaximized()
            global_State=1
            self.maximizeButton.setToolTip('Restore')

        else:
            global_State=0
            self.showNormal()
            self.resize(self.width()+1, self.height()+1)
            self.maximizeButton.setToolTip('Maximize')

    def return_status(self):
        return global_State

    def moveWindow(self, event):
        if self.return_status()==1:
            self.maximize_restore()

        # IF LEFT CLICK MOVE WINDOW
        if event.buttons() == QtCore.Qt.LeftButton:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


def main():
    app=QApplication(sys.argv)
    exp=Explore()
    exp.show()
    app.exec()

if __name__ == '__main__' : main()