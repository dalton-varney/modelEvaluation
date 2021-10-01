import time
import edgeiq
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from collections import deque
from helpers import *
import numpy as np
import xml.etree.ElementTree as ET
from numpy.lib.type_check import imag, real

app = Flask(__name__, template_folder='./templates/')

socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(
    app, logger=socketio_logger, engineio_logger=socketio_logger)

SESSION = time.strftime("%d%H%M%S", time.localtime())

obj_detect = edgeiq.ObjectDetection("dvarney/PPEDetectionMobileNetJune6") #Set your Model Here
slideShowSpeed = 3 #set how fast the slideshow goes
zipped_data = 'HelmetTest2.zip' #Set your Dataset Here

obj_detect.load(engine=edgeiq.Engine.DNN)
SAMPLE_RATE = 50


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@socketio.on('connect')
def connect_cv():
    print('[INFO] connected: {}'.format(request.sid))


@socketio.on('disconnect')
def disconnect_cv():
    print('[INFO] disconnected: {}'.format(request.sid))


@socketio.on('close_app')
def close_app():
    print('Stop Signal Received')
    controller.close()


class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler
        (https://github.com/alwaysai/video-streamer) and is modified here.

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.all_frames = deque()
        self.video_frames = deque()
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self

    def run(self):
        # loop detection
        self.fps.start()
        text = [""]
        dataset = zipped_data[:-4]
        folder_set_up(dataset)
        open_dataset(dataset, zipped_data)

        image_files = get_all_files(os.path.join(dataset, 'JPEGImages'))
        xml_files = get_all_files(os.path.join(dataset, 'Annotations'))
        avgf1 = []
        for i in range(len(image_files)):
            # print("starting image", image_files[i])
            img = cv2.imread(get_file(image_files[i], dataset))
            frame = np.array(img)
            # Perform object detection
            results = obj_detect.detect_objects(
                        frame, confidence_level=.3)
            predictions = results.predictions

            # Get the ground truth boundaries
            xml_file = get_file(xml_files[i], dataset)
            real_bounds = self.parse_xml(xml_file)

            # Add the model predictions and annotations to the frame
            frame = edgeiq.markup_image(
                    frame, predictions, colors=[(0,80,200)])
            print("Predictions: ", predictions)
            frame = edgeiq.transparent_overlay_boxes(
                    frame, real_bounds, alpha=0.35, colors=[(255,0,0)], show_labels=False, show_confidences=False)

            accuracy = self.evaluate(predictions, real_bounds)

            f1 = self.f1Score(accuracy, real_bounds)

            text = ["Model Detections in Orange"]
            text.append(f"Ground Truth Dataset in Blue: {zipped_data}")
            avgf1.append(f1)
            text.append(" ")
            text.append("Performance: ")
            if len(avgf1) > 0:
                text.append("Avg F1Score: {:.2f}".format(sum(avgf1)/len(avgf1)))
            if len(predictions) == 0:
                text.append(" ")
                text.append("CAUTION: Model is not outputting any Detections")

            #A shortcut to hold a specific frame for a certain amount of seconds
            start = time.time()
            while (time.time() - start < slideShowSpeed):
                self.send_data(frame, text)
            socketio.sleep(0.01)


            self.fps.update()
            if self.check_exit():
                controller.close()
        print("Completed The Dataset")
        print("F1Score: {:.2f}".format(sum(avgf1)/len(avgf1)))
        print("MaP: {:.2f}".format(sum(avgmAP)/len(avgmAP)))
        controller.close()

    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def parse_xml(self, file):
        """
        Parses through xml file to create a list of
        ObjectDetectionPredictions from the true annotations.
        """
        predictions = []
        tree = ET.parse(file)
        root = tree.getroot()
        # Parse through all data annotations (under 'object' tag)
        for obj in root.findall('object'):
            label = obj.find('name').text
            # Get the coordinates of the Bounding Box
            start_x = int(float(obj.find('bndbox').find('xmin').text))
            start_y = int(float(obj.find('bndbox').find('ymin').text))
            end_x = int(float(obj.find('bndbox').find('xmax').text))
            end_y = int(float(obj.find('bndbox').find('ymax').text))

            # Create Bounding Box and the real Object Prediction
            box = edgeiq.BoundingBox(start_x, start_y, end_x, end_y)
            prediction = edgeiq.ObjectDetectionPrediction(box, 1.0, label, 1)
            predictions.append(prediction)
        return predictions

    def evaluate(self, predictions, real_bounds):
        """
        Calculates the accuracy of the model predictions compared to the
        ground truth annotations.
        """
        accuracy = []
        # Iterate through the model predictions and the ground truth annotations
        for i in range(len(predictions)):
            temp = []
            for j in range(len(real_bounds)):
                model_prediction = predictions[i].box
                ground_truth = real_bounds[j].box

                # Compute the accuracy of the model prediction
                overlap = model_prediction.compute_overlap(ground_truth)
                temp.append(overlap)
            # Add the most accurate overlapping model prediction and ground truth annotation
            accuracy.append(max(temp))
        return accuracy

    def f1Score(self, accuracy, real_bounds):
        """
        Calculates the f1 of the model predictions compared to the
        ground truth annotations.
        """
        positives = []
        accurancyLength = len(accuracy)
        real_boundsLength = len(real_bounds)
        f_negative = 0
        f_positive = 0

        # Determine if the predictions are true/false positives
        for thresh in accuracy:
            if thresh > 0.4:
                positives.append(1)
            else:
                positives.append(0)

        if accurancyLength < real_boundsLength:
            for i in range(real_boundsLength - accurancyLength):
                positives.append(0)
                f_negative += 1
        if accurancyLength > real_boundsLength:
            for i in range(accurancyLength - real_boundsLength):
                f_positive += 1

        # TODO: use sklearn module to calculate the mAP of the predictions
        y_true = np.array(positives)
        y_scores = np.array(accuracy)
        t_positive = sum(y_true)

        recall = t_positive / (t_positive + f_negative)
        bottom = t_positive + ((f_positive + f_negative)/2)
        f1 = 0.0
        if bottom > 0:
            f1 = t_positive/bottom

        return f1

    def send_data(self, frame, text):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                    frame, width=720, height=480, keep_scale=True)
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text)
                    })
            socketio.sleep(0.01)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.exit_event.is_set()

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()


class Controller(object):
    def __init__(self):
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('[INFO] Starting server at http://localhost:5000')
        socketio.run(app=app, host='0.0.0.0', port=5000)

    def close(self):
        self.fps.stop()
        print("elapsed time: {:.2f}".format(self.fps.get_elapsed_seconds()))

        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()

        print("Program Ending")


controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        controller.close()
