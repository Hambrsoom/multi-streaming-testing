'''
the following code was taken from : https://github.com/kkroening/ffmpeg-python/blob/master/examples/tensorflow_stream.py

Example streaming ffmpeg numpy processing.

Demonstrates using ffmpeg to decode video input, process the frames in
python, and then encode video output using ffmpeg.

This example uses two ffmpeg processes - one to decode the input video
and one to encode an output video - while the raw frame processing is
done in python with numpy.

At a high level, the signal graph looks like this:
  (input video) -> [ffmpeg process 1] -> [python] -> [ffmpeg process 2] -> (output video)
This example reads/writes video files on the local filesystem, but the
same pattern can be used for other kinds of input/output (e.g. webcam,
rtmp, etc.).

The simplest processing example simply darkens each frame by
multiplying the frame's numpy array by a constant value; see
``process_frame_simple``.

A more sophisticated example processes each frame with tensorflow using
the "deep dream" tensorflow tutorial; activate this mode by calling
the script with the optional `--dream` argument.  (Make sure tensorflow
is installed before running)

'''
from __future__ import print_function
import argparse
import ffmpeg
import logging
import numpy as np
import os
import subprocess
import zipfile
import cv2
from datetime import datetime
import numpy as np
from PIL import Image

from datetime import date


# import yolo neural net configurations
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# extract class/label names
classes = []
with open('./coco.names', 'r') as f:
  classes = f.read().splitlines()

parser = argparse.ArgumentParser(description='Example streaming ffmpeg numpy processing')
parser.add_argument('in_filename', help='Input filename')
parser.add_argument('out_filename', help='Output filename')
parser.add_argument(
    '--dream', action='store_true', help='Use DeepDream frame processing (requires tensorflow)')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

historyLength = 0
feedForwardHistory = []


def analyze_frame(img):
    height, width, _ = img.shape

    # image = Image.fromarray(img, 'RGB')
    # image.save('my.png')
    # image.show()

    # prepare image for yolo detection
    # normalize the value of each pixels by dividing by 255
    # requires size of 416 by 416, needs to be a square
    # needs to be in rgb ordering
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=False, crop=False)

    # set image as input of yolo neural network
    net.setInput(blob)
    # get the layer games from the neural net
    output_layers_names = net.getUnconnectedOutLayersNames()

    global historyLength, feedForwardHistory

    layerOutputs = feedForwardHistory

    if historyLength==0 :
        historyLength = 60

        # run the forward pass and get the output after running the neural network on the image
        layerOutputs = net.forward(output_layers_names)
        feedForwardHistory = layerOutputs

    historyLength = historyLength - 1

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output: # detection contains information about the bounded detection, with box coordinates and scores
            #extracts only the label/classification with the highest confidence for the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                #need to multiply by height or width since the values were normalized before passing into the neural network
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # coordinates of upper left corner
                x = int(center_x - w/2) 
                y = int(center_y - h/2)


                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # get count of estimated boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4) #same threshhold as confidence

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # loop over all objects detected
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i] #extract box coordinates
            label = str(classes[class_ids[i]]) #extract label of box
            confidence = str(round(confidences[i], 2)) #extract the confidence
            
            #format color, text, rectangle, for display on image
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)
        
        cv2.imwrite('./frames/video'+str(datetime.now())+'.jpg', img)



    # shows each blob (each Red Green Blue image)
    for b in blob:
        for n, img_blob in enumerate(b):
            cv2.imwrite('./frames/video'+str(datetime.now())+str(n)+'.jpg', img_blob)

    return img;

def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg process1')
    args = ['ffmpeg', '-rtsp_transport', 'tcp', '-i', in_filename, '-b', '900k', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:']

    # args = ['ffmpeg', '-i', 'rtsp://localhost:8554/live', '-b', '900k', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:']
    # args = (       
    #     ffmpeg
    #     .input(in_filename)
    #     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    #     .compile()
    # )
    # print("args***")
    # print(args);
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_process2(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    # args = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '852x480', '-i', 'pipe:', '-f', 'dash', './output/output.mpd', '-y'] 
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, format='hls', start_number=0, hls_time=2, hls_list_size=1, hls_flags='delete_segments', hls_allow_cache=0)
        .overwrite_output()
        .compile()
    )
    # print('args***')
    # print(args)
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

# calling YOLO here
def process_frame_simple(frame):
    return analyze_frame(frame)

def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def run(in_filename, out_filename, process_frame):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        out_frame = process_frame(in_frame)
        write_frame(process2, out_frame)

    logger.info('Waiting for ffmpeg process1')
    process1.wait()

    logger.info('Waiting for ffmpeg process2')
    process2.stdin.close()
    process2.wait()

    logger.info('Done')

if __name__ == '__main__':
    args = parser.parse_args()
    process_frame = process_frame_simple
    run(args.in_filename, args.out_filename, process_frame)
