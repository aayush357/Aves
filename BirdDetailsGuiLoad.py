from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from PyQt5 import QtCore, QtMultimedia
from PyQt5.QtCore import Qt
from PyQt5.uic import *
import requests, json, re, urllib, webbrowser

global_State=0


class BirdDetailsGuiLoader(QMainWindow):
    def __init__(self, data):
        super().__init__()
        loadUi('UI/BirdDetailScreen.ui', self)
        self.mediaPlayer = QtMultimedia.QMediaPlayer()
        self.bird_name = data['birdName']
        filename = self.audio_links(self.bird_name)
        if filename == "none":
            self.MediaPlayer.hide()
        else:
            url = QtCore.QUrl(filename)
            content = QtMultimedia.QMediaContent(url)
            self.mediaPlayer.setMedia(content)


        self.slider.setRange(0, 0)
        self.ListenAudio.clicked.connect(self.listenAudio)
        self.slider.sliderMoved.connect(self.setPosition)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)


        # Removing windows title bar
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # adding function to close button
        self.closeButton.clicked.connect(self.cls)
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        self.titleBarFrame.mouseMoveEvent = self.moveWindow
        self.maximizeButton.clicked.connect(lambda: self.maximize_restore())

        if data['birdId'] > 0:
            self.loadDetails(data)
        self.ReadMoreButton.clicked.connect(
            lambda: self.openWiki(data['birdId']))

    def cls(self):
        self.mediaPlayer.stop()
        self.close()

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.ListenAudio.setText("Pause")
        else:
            self.ListenAudio.setText("Listen")

    def positionChanged(self, position):
        self.slider.setValue(position)

    def durationChanged(self, duration):
        self.slider.setRange(0, duration)

    def listenAudio(self):
        self.slider.show()
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()

        else:
            self.mediaPlayer.play()

    def audio_links(self, nam):
        audiolinks = {}
        f = open('text_files/Audio_links.txt')
        x = f.read()
        x = x.split('\n')
        for l1 in range(0, len(x) - 1):
            l = x[l1]
            name = l.split(' ')[0]
            link = l.split(' ')[-1]
            audiolinks[name] = link
        print(audiolinks)

        return audiolinks[nam]


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

    def loadDetails(self, birdData):
        birdId = birdData['birdId']
        birdName = birdData['birdName']
        birdImage = birdData['birdImageUrl']
        birdStatus = birdData['statusOfBird']
        birdOrder = birdData['orderOfBird']
        birdFamily = birdData['familyOfBird']
        birdGenus = birdData['genusOfBird']
        birdSpecies = birdData['speciesOfBird']
        birdBinomialName = birdData['binomialName']
        birdDetails = birdData['descriptionOfBird']
        birdMap = birdData['mapImageURl']

        birdName = birdName.replace("_", " ")
        birdDetails = re.sub(r'\[\d\]', "", birdDetails)
        birdImage = (birdImage.split("/220")[0]).replace("/thumb", "")
        pixmapImage = QPixmap()
        data = urllib.request.urlopen(birdImage).read()
        pixmapImage.loadFromData(data)

        if birdMap == 'Not Available':
            self.Map.setText("Map Not Available")
        else:
            pixmapMap = QPixmap()
            data = urllib.request.urlopen(birdMap).read()
            pixmapMap.loadFromData(data)
            self.Map.setPixmap(pixmapMap)

        aspectRatio = pixmapImage.width() / pixmapImage.height()
        height = min(pixmapImage.height(), 500)
        width = height * aspectRatio
        self.BirdImageMain.setPixmap(pixmapImage.scaled(height, width, QtCore.Qt.KeepAspectRatio))

        self.BirdName.setText(birdName)
        self.BinomialName.setText(birdBinomialName)
        self.SpeciesValue.setText(birdSpecies)
        self.OrderValue.setText(birdOrder)
        self.FamilyValue.setText(birdFamily)
        self.GenusValue.setText(birdGenus)
        self.StatusValue.setText(birdStatus)
        self.Description.setText(birdDetails)

    def openWiki(self, birdId):
        links = open(r'Resource/onlyWikiLinks.txt').read().splitlines()
        webbrowser.open(links[birdId - 1])

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

    # API url to fetch Bird Details
    apiUrl = 'https://aves-detail.herokuapp.com/getDetails/'

    # Bird Name to be searched
    birdName = "Red_faced_Cormorant"
    # birdName = "abc"
    # Fetch the Bird Details
    response = requests.get(apiUrl + birdName)
    data = json.loads(response.text)

    birdDetailsGuiLoader = BirdDetailsGuiLoader(data)
    birdDetailsGuiLoader.show()
    app.exec_()


if __name__ == '__main__':
    main()
