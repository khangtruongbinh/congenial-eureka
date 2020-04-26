import os
import queue
import sys
import threading
import time

import cv2 as cv
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtWidgets import QWidget, QMainWindow, QDesktopWidget

from config.config import get_config
from constant.constants import NUM_FRAMES_PER_CLIP
from src.display.single_video import data_process
from src.video_processing.args import cli
from src.video_processing.c3d_realtime import process_images
from src.video_processing.network import PifPaf, MonoLoco
from src.video_processing.shared import SharedObj
from src.display.single_video import visualize, add_predict

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

running = False
q1 = queue.Queue()
# q2 = queue.Queue()
FRAMES_QUEUE_1 = queue.Queue(maxsize=NUM_FRAMES_PER_CLIP)
# FRAMES_QUEUE_2 = queue.Queue(maxsize=NUM_FRAMES_PER_CLIP)

isStop = False
low_threshold = False


def grab(cam, display_queue, frame_queue, confidence, args):
    global running
    global isStop
    global ALERT_LEVEL
    capture = cv.VideoCapture(cam)
    count = 0
    while True:
        while running:
            capture.grab()
            ret, img = capture.read()
            if not ret:
                break
            if not display_queue.empty():
                try:
                    display_queue.get_nowait()
                except Exception as ex:
                    print(str(ex))
                    pass
            count += 1
            # data_process(img, args)
            pre_process_frame = data_process(img, args)
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(pre_process_frame)
            # # if confidence.confidence > ALERT_LEVEL[0] and count % 60 < 40:
            # #     add_predict(img, predict=1, confidence_score=confidence.confidence, alert_level=ALERT_LEVEL)
            if count % 60 < 40:
                visualize(img, confidence.image_loc, args.scale)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            display_queue.put(img)
            time.sleep(0.07)
        while not display_queue.empty():
            display_queue.get_nowait()
        while isStop:
            sys.exit()


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def set_image(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class Main(QMainWindow, uic.loadUiType("mainwindow.ui")[0]):
    def __init__(self, confidence_1, confidence_2, parent=None, args=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.center()
        self.args = args
        self.startButton.clicked.connect(self.start_streaming)
        # self.startButton.setStyleSheet("background: black; color: white;")
        # self.thresholdButton.clicked.connect(self.set_threshold)
        # self.thresholdButton.setStyleSheet("background: red; color: white;")
        self.addCam.clicked.connect(self.add_cam)
        self.Cam_1 = OwnImageWidget(self.Cam_1)
        # self.Cam_2 = OwnImageWidget(self.Cam_2)
        self.window_width = 600
        self.window_height = 525
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.confidence_1 = confidence_1
        self.confidence_2 = confidence_2
        self.queues = [
            [q1, self.Cam_1],
            # [q2, self.Cam_2]
        ]
        self.capture_threads = []
        self.process_threads = []

        # self.set_event_list()
        self.listEvent.itemClicked.connect(self.event_click)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def start_streaming(self):
        global running
        running = True
        capture_thread_1 = threading.Thread(
            target=grab,
            # args=("rtsp://admin:iphone3gs@%s:554/onvif1" % self.ipCam_1.text(), q1, FRAMES_QUEUE_1, self.confidence_1))
            args=(get_config("video", "camera_1"), q1, FRAMES_QUEUE_1, self.confidence_1, self.args))
        # capture_thread_2 = threading.Thread(
        #     target=grab,
        #     # args=("rtsp://admin:iphone3gs@%s:554/onvif1" % self.ipCam_2.text(), q2, FRAMES_QUEUE_2, self.confidence_2))
        #     args=(get_config("video", "camera_2"), q2, FRAMES_QUEUE_2, self.confidence_2, self.args))

        # Video processing thread
        process_thread_1 = threading.Thread(
            target=process_images,
            args=(FRAMES_QUEUE_1, pifpaf_1, monoloco_1, confidence_1, name1, self.args))
        # process_thread_2 = threading.Thread(
        #     target=process_images,
        #     args=(FRAMES_QUEUE_2, pifpaf_2, monoloco_2, confidence_2, name2, self.args))
        # self.ipCam_1.setVisible(False)
        # self.ipCam_2.setVisible(False)
        # if self.ipCam_1.text() is not '':
        if not capture_thread_1.isAlive():
            capture_thread_1.start()
        # if self.ipCam_2.text() is not '':
        # if not capture_thread_2.isAlive():
        #     capture_thread_2.start()
        if not process_thread_1.isAlive():
            process_thread_1.start()
        # if not process_thread_2.isAlive():
        #     process_thread_2.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')


    # Ham lay ip camera va them vao stream
    def add_cam(self):
        ip_address = self.ipCam.text()
        self.ipCam.setText('')

    def update_frame(self):
        # Update video frame
        for thread in self.queues:
            if not thread[0].empty():
                if running:
                    self.startButton.setText('Camera is live')
                img = thread[0].get()

                img_height, img_width, img_colors = img.shape
                scale_w = float(self.window_width) / float(img_width)
                scale_h = float(self.window_height) / float(img_height)
                scale = min([scale_w, scale_h])

                if scale == 0:
                    scale = 1

                img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
                height, width, bpc = img.shape
                bpl = bpc * width
                image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
                thread[1].set_image(image)

    def set_event_list(self):
        events = os.listdir(get_config('video', 'events'))
        for event in events[::-1]:
            ts = event.split('.')[0]
            # event_datetime = datetime.fromtimestamp(int(ts))
            self.listEvent.addItem(ts)

    def event_click(self, item):
        video_path = os.path.join(get_config('video', 'events'), '%s.mp4'%item.text())
        self._event = VideoPlayer(video_path)

    def close_event(self, event):
        global running
        global isStop

        isStop = True
        running = False
        event.accept()


class VideoPlayer:
    def __init__(self, video_path):
        self.video = QVideoWidget()
        self.video.resize(400, 300)
        self.video.move(0, 0)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.callback() # Start play video
        self.b = QPushButton('Play')
        self.b.clicked.connect(self.callback)
        self.b.show()

    def callback(self):
        self.player.setPosition(0) # to start at the beginning of the video every time
        self.video.show()
        self.player.play()


if __name__ == '__main__':
    args = cli()
    args.camera = True
    args.webcam = True
    args.scale = 0.5
    confidence_1 = SharedObj()
    confidence_2 = SharedObj()
    name1 = "cam_1"
    name2 = "cam_2"
    args.model = "/mnt/d/Antimatlab/multi-cam/weights/monoloco-190513-1437.pkl"
    # graph = load_graph(get_config("model", "c3d"))
    # c3d_graph_1 = ImportGraph(graph)  # day la cho load frozen model
    # c3d_graph_2 = ImportGraph(graph)  # day la cho load frozen model
    pifpaf_1 = PifPaf(args)
    monoloco_1 = MonoLoco(model=args.model, device=args.device)
    # pifpaf_2 = PifPaf(args)
    # monoloco_2 = MonoLoco(model=args.model, device=args.device)

    app = QApplication(sys.argv)
    form5 = Main(confidence_1, confidence_2, args=args)
    form5.setWindowTitle('AntiMatlab')
    form5.show()
    app.exec_()
