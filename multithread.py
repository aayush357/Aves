from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.uic import loadUi
import requests, urllib
import json
from BirdDetailsGuiLoad import BirdDetailsGuiLoader
from list_making import *
from random import sample
from explore_win import *

global_State = 0


class Worker(QObject):
    finished = pyqtSignal(QPixmap, dict, int)

    @pyqtSlot(str, int)
    def run(self, name, num):
        img_dict = {}
        print('worker{} running'.format(num))
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 163)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage
        print('generating images')
        self.finished.emit(img_dict[0], data, num)


class MainScreen(QMainWindow):
    img_request1 = pyqtSignal(str, int)
    img_request2 = pyqtSignal(str, int)
    img_request3 = pyqtSignal(str, int)
    img_request4 = pyqtSignal(str, int)
    img_request5 = pyqtSignal(str, int)
    img_request6 = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        loadUi('UI/MainScreen.ui', self)
        # Removing windows title bar
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # adding function to close button
        self.closeButton.clicked.connect(lambda: self.close())
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        self.titleBarFrame.mouseMoveEvent = self.moveWindow
        # self.maximizeButton.clicked.connect(lambda: self.showMaximized())
        self.SelectImageButton.clicked.connect(self.selectImage)
        self.explore_btn.clicked.connect(self.explore)

        self.worker1 = Worker()
        self.worker1_thread = QThread()
        self.worker1.moveToThread(self.worker1_thread)
        self.worker1_thread.start()
        self.img_request1.connect(self.worker1.run)
        self.worker1.finished.connect(self.set_Img)

        self.worker2 = Worker()
        self.worker2_thread = QThread()
        self.worker2.moveToThread(self.worker2_thread)
        self.worker2_thread.start()
        self.img_request2.connect(self.worker2.run)
        self.worker2.finished.connect(self.set_Img)

        self.worker3 = Worker()
        self.worker3_thread = QThread()
        self.worker3.moveToThread(self.worker3_thread)
        self.worker3_thread.start()
        self.worker3.finished.connect(self.set_Img)
        self.img_request3.connect(self.worker3.run)

        self.worker4 = Worker()
        self.worker4_thread = QThread()
        self.worker4.moveToThread(self.worker4_thread)
        self.worker4_thread.start()
        self.worker4.finished.connect(self.set_Img)
        self.img_request4.connect(self.worker4.run)

        self.worker5 = Worker()
        self.worker5_thread = QThread()
        self.worker5.moveToThread(self.worker5_thread)
        self.worker5_thread.start()
        self.worker5.finished.connect(self.set_Img)
        self.img_request5.connect(self.worker5.run)

        self.worker6 = Worker()
        self.worker6_thread = QThread()
        self.worker6.moveToThread(self.worker6_thread)
        self.worker6_thread.start()
        self.worker6.finished.connect(self.set_Img)
        self.img_request6.connect(self.worker6.run)

        # self.run_long_task()

        self.fname = ""
        self.bname = ""
        self.MoreInfoButton.clicked.connect(self.showDetails)
        self.maximizeButton.clicked.connect(lambda: self.maximize_restore())
        pixmapImage = QPixmap("Resource/image.png")
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 250)
        width = height * aspectRatio
        self.appIcon.setPixmap(pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio))
        # self.BirdNameValue.setText("<small>Select Image First</small>")

        self.ResultFrame.hide()

    def randomBirds(self):
        self.names = []
        self.names_without = []
        randomlist = sample(range(1, 200), 6)
        all_names = list_birds()
        for i in randomlist:
            name = (all_names[i])[4:]
            self.names.append(name)
            self.names_without.append(name.replace('_', ' '))

    def maximize_restore(self):
        global global_State
        status = global_State

        if status == 0:
            self.showMaximized()
            global_State = 1
            self.maximizeButton.setToolTip('Restore')

        else:
            global_State = 0
            self.showNormal()
            self.resize(self.width() + 1, self.height() + 1)
            self.maximizeButton.setToolTip('Maximize')

    def loadImage(self, fname):
        pixmapImage = QPixmap(fname)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 250)
        width = round(height * aspectRatio)
        self.BirdImage.setPixmap(
            pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def explore(self):
        self.randomBirds()
        self.run_long_task()
        self.Expl.show()

    def predictClass(self, fname):
        res = requests.post("https://bird-species-class-prediction.herokuapp.com/predict",
                            files={'file': open(fname, 'rb')})
        resJson = json.loads(res.text)
        birdName = resJson['birdName']
        birdName = birdName.replace("_", " ")
        self.BirdNameValue.setText(birdName)
        self.bname = resJson['birdName']
        return str(resJson['birdName'])

    def run_long_task(self):
        self.Expl = Explore()
        self.Expl.name1.setEnabled(False)
        self.Expl.name2.setEnabled(False)
        self.Expl.name3.setEnabled(False)
        self.Expl.name4.setEnabled(False)
        self.Expl.name5.setEnabled(False)
        self.Expl.name6.setEnabled(False)

        print(self.names[0])
        self.img_request1.emit(self.names[0], 1)
        self.img_request2.emit(self.names[1], 2)
        self.img_request3.emit(self.names[2], 3)
        self.img_request4.emit(self.names[3], 4)
        self.img_request5.emit(self.names[4], 5)
        self.img_request6.emit(self.names[5], 6)

    def set_Img(self, img, api_data, n):
        print('num', n)
        image = getattr(self.Expl, 'image{}'.format(n))
        lname = getattr(self.Expl, 'lname{}'.format(n))
        name = getattr(self.Expl, 'name{}'.format(n))
        image.setPixmap(img)
        print(2)
        lname.setText(self.names_without[n - 1])
        name.setEnabled(True)
        name.clicked.connect(lambda: self.load_details_exp(api_data))

    def selectImage(self):
        self.SelectImageButton.setText("Loading")
        self.SelectImageButton.setEnabled(False)
        self.MoreInfoButton.setEnabled(False)
        self.explore_btn.setEnabled(False)

        filepath = QFileDialog.getOpenFileName(
            self, 'Open file', 'c:\\', "Image files (*.jpg *.gif *.jpeg)")
        self.fname = filepath[0]
        if (self.fname != ''):
            self.loadImage(self.fname)
            self.predictClass(self.fname)
            self.loadDetails()
            self.ResultFrame.show()
        self.SelectImageButton.setText("Select Image")
        self.SelectImageButton.setEnabled(True)
        self.MoreInfoButton.setEnabled(True)
        self.explore_btn.setEnabled(True)

    def loadDetails(self):
        # API url to fetch Bird Details
        apiUrl = 'https://aves-detail.herokuapp.com/getDetails/'

        # Bird Name to be searched
        birdName = self.bname
        # birdName = "abc"
        # Fetch the Bird Details
        response = requests.get(apiUrl + birdName)
        data = json.loads(response.text)
        self.birdDetailsGuiLoader = BirdDetailsGuiLoader(data)

    def load_details_exp(self, apiData):
        print(apiData['birdName'])
        self.det = BirdDetailsGuiLoader(apiData)
        self.det.show()

    def showDetails(self):
        if (self.bname != ''):
            self.birdDetailsGuiLoader.show()

    def moveWindow(self, event):
        if self.return_status():
            self.maximize_restore()

        # IF LEFT CLICK MOVE WINDOW
        if event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def return_status(self):
        return global_State


def main():
    app = QApplication(sys.argv)
    mainScreen = MainScreen()
    mainScreen.show()
    app.exec_()


if __name__ == '__main__':
    main()
