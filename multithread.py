from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from PyQt5.QtCore import Qt,QObject,QThread,pyqtSignal, pyqtSlot
from PyQt5.uic import loadUi
import requests,urllib
import json
from BirdDetailsGuiLoad import BirdDetailsGuiLoader
from list_making import *
from random import sample
from explore_win import *


class Worker1(QObject):
    finished=pyqtSignal(dict)

    @pyqtSlot(str)
    def run(self, name):
        img_dict={}
        print('worker1 running')
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 170)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage
        print('generating images')
        self.finished.emit(img_dict)

class Worker2(QObject):
    finished=pyqtSignal(dict)

    @pyqtSlot(str)
    def run(self, name):
        img_dict={}
        print('worker2 running')
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 170)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage
        self.finished.emit(img_dict)

class Worker3(QObject):
    finished=pyqtSignal(dict)

    @pyqtSlot(str)
    def run(self, name):
        img_dict={}
        print('worker3 running')
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 170)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage

        self.finished.emit(img_dict)

class Worker4(QObject):
    finished=pyqtSignal(dict)

    @pyqtSlot(str)
    def run(self, name):
        img_dict={}
        print('worker4 running')
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 170)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage

        self.finished.emit(img_dict)

class Worker5(QObject):
    finished=pyqtSignal(dict)

    @pyqtSlot(str)
    def run(self, name):
        img_dict={}
        print('worker5 running')
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 170)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage

        self.finished.emit(img_dict)

class Worker6(QObject):
    finished=pyqtSignal(dict)

    @pyqtSlot(str)
    def run(self, name):
        img_dict={}
        print('worker6 running')
        apiurl = 'https://aves-detail.herokuapp.com/getDetails/'
        response = requests.get(apiurl + name)
        data = json.loads(response.text)
        imageurl = data['birdImageUrl']
        imageurl = (imageurl.split("/220")[0]).replace("/thumb", "")
        img = urllib.request.urlopen(imageurl).read()
        pixmapImage = QPixmap()
        pixmapImage.loadFromData(img)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 170)
        width = round(height * aspectRatio)
        pixmapImage = pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        img_dict[0] = pixmapImage

        self.finished.emit(img_dict)


class MainScreen(QMainWindow):
    img_request1 = pyqtSignal(str)
    img_request2 = pyqtSignal(str)
    img_request3 = pyqtSignal(str)
    img_request4 = pyqtSignal(str)
    img_request5 = pyqtSignal(str)
    img_request6 = pyqtSignal(str)

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

        self.names = []
        self.names_without = []
        randomlist = sample(range(1, 200), 6)
        all_names = list_birds()
        for i in randomlist:
            name = (all_names[i])[4:]
            self.names.append(name)
            self.names_without.append(name.replace('_', ' '))

        print(self.names)
        self.worker1=Worker1()
        self.worker1_thread=QThread()
        self.worker1.moveToThread(self.worker1_thread)
        self.worker1_thread.start()
        self.img_request1.connect(self.worker1.run)
        self.worker1.finished.connect(self.img1_Set)

        self.worker2 = Worker2()
        self.worker2_thread = QThread()
        self.worker2.moveToThread(self.worker2_thread)
        self.worker2_thread.start()
        self.img_request2.connect(self.worker2.run)
        self.worker2.finished.connect(self.img2_Set)

        self.worker3 = Worker3()
        self.worker3_thread = QThread()
        self.worker3.moveToThread(self.worker3_thread)
        self.worker3_thread.start()
        self.worker3.finished.connect(self.img3_Set)
        self.img_request3.connect(self.worker3.run)

        self.worker4 = Worker4()
        self.worker4_thread = QThread()
        self.worker4.moveToThread(self.worker4_thread)
        self.worker4_thread.start()
        self.worker4.finished.connect(self.img4_Set)
        self.img_request4.connect(self.worker4.run)

        self.worker5 = Worker5()
        self.worker5_thread = QThread()
        self.worker5.moveToThread(self.worker5_thread)
        self.worker5_thread.start()
        self.worker5.finished.connect(self.img5_Set)
        self.img_request5.connect(self.worker5.run)

        self.worker6 = Worker6()
        self.worker6_thread = QThread()
        self.worker6.moveToThread(self.worker6_thread)
        self.worker6_thread.start()
        self.worker6.finished.connect(self.img6_Set)
        self.img_request6.connect(self.worker6.run)

        self.run_long_task()

        self.fname = ""
        self.bname = ""
        self.MoreInfoButton.clicked.connect(self.showDetails)
        pixmapImage = QPixmap("Resource/image.png")
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 200)
        width = height * aspectRatio
        self.BirdImage.setPixmap(pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio))
        self.BirdNameValue.setText("<small>Select Image First</small>")
        self.Expl = Explore()
        self.ResultFrame.hide()


    def loadImage(self, fname):
        pixmapImage = QPixmap(fname)
        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 250)
        width = round(height * aspectRatio)
        print(height, width)
        self.BirdImage.setPixmap(pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def explore(self):
        self.Expl.show()

    def predictClass(self, fname):
        res = requests.post("https://bird-species-class-prediction.herokuapp.com/predict",files={'file': open(fname, 'rb')})
        resJson = json.loads(res.text)
        birdName = resJson['birdName']
        birdName = birdName.replace("_", " ")
        self.BirdNameValue.setText(birdName)
        self.bname = resJson['birdName']
        return str(resJson['birdName'])

    def run_long_task(self):
        self.img_request1.emit(self.names[0])
        self.img_request2.emit(self.names[1])
        self.img_request3.emit(self.names[2])
        self.img_request4.emit(self.names[3])
        self.img_request5.emit(self.names[4])
        self.img_request6.emit(self.names[5])

    def img1_Set( self, img1):
        self.Expl.image1.setPixmap(img1[0])
        self.Expl.lname1.setText(self.names_without[0])
        self.Expl.name1.clicked.connect(lambda :self.load_details_exp(self.names[0]))

    def img2_Set(self, img2):
        self.Expl.image2.setPixmap(img2[0])
        self.Expl.lname2.setText(self.names_without[1])
        self.Expl.name2.clicked.connect(lambda: self.load_details_exp(self.names[1]))

    def img3_Set(self, img3):
        self.Expl.image3.setPixmap(img3[0])
        self.Expl.lname3.setText(self.names_without[2])
        self.Expl.name3.clicked.connect(lambda: self.load_details_exp(self.names[2]))

    def img4_Set(self, img4):
        self.Expl.image4.setPixmap(img4[0])
        self.Expl.lname4.setText(self.names_without[3])
        self.Expl.name4.clicked.connect(lambda: self.load_details_exp(self.names[3]))

    def img5_Set(self, img5):
        self.Expl.image5.setPixmap(img5[0])
        self.Expl.lname5.setText(self.names_without[4])
        self.Expl.name5.clicked.connect(lambda: self.load_details_exp(self.names[4]))

    def img6_Set(self, img6):
        self.Expl.image6.setPixmap(img6[0])
        self.Expl.lname6.setText(self.names_without[5])
        self.Expl.name6.clicked.connect(lambda: self.load_details_exp(self.names[5]))

    def selectImage(self):
        self.SelectImageButton.setText("Loading")
        self.SelectImageButton.setEnabled(False)
        self.MoreInfoButton.setEnabled(False)
        self.explore_btn.setEnabled(False)

        filepath = QFileDialog.getOpenFileName(
            self, 'Open file', 'c:\\', "Image files (*.jpg *.gif *.jpeg)")
        self.fname = filepath[0]
        if(self.fname != ''):
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

    def load_details_exp(self,b_name):
        apiurl='https://aves-detail.herokuapp.com/getDetails/'
        print(b_name)
        response=requests.get(apiurl+b_name)
        data=json.loads(response.text)
        self.det=BirdDetailsGuiLoader(data)
        self.det.show()

    def showDetails(self):
        if(self.bname != ''):
            self.birdDetailsGuiLoader.show()

    def moveWindow(self, event):
        # IF LEFT CLICK MOVE WINDOW
        if event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


def main():
    app = QApplication(sys.argv)
    mainScreen = MainScreen()
    mainScreen.show()
    app.exec_()


if __name__ == '__main__':
    main()