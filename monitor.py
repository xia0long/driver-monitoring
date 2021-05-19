#!/usr/bin/env python3
import json
import argparse

import cv2
import numpy as np
import pygame
import onnxruntime

WIDTH = 320
HEIGHT = 160

def render(display, image, driver_state):

    array = np.frombuffer(image, dtype=np.dtype("uint8"))
    array = np.reshape(array, (HEIGHT, WIDTH, 3))
    array = array[:,:,::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0,0))
    
    # draw driver state
    font = pygame.font.SysFont('freesansbold.ttf', 16)
    h = 0
    for k, v in driver_state.items():
        if isinstance(v, str):
            text = "{}: {}".format(k, v)
        else:
            text = "{}: {:.4f}".format(k, v)
        info = font.render(text, False, (255,200,10))
        display.blit(info, (10, h))
        h += 15


def parse_arguments():

    parser = argparse.ArgumentParser(""""Driver Monitoring""")
    parser.add_argument("-v", "--video",
                        type=str,
                        default=None,
                        help="use stream from video")
    parser.add_argument("-c", "--camera",
                        action="store_true",
                        help="use stream from camera")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_arguments()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.camera:
        cap = cv2.VideoCapture(0)
    else:
        print("Wrong args")
    
    images = []
    session = onnxruntime.InferenceSession("dmonitoring_model.onnx", None)

    pygame.init()
    pygame.display.set_caption("Driver Monitoring")
    clock = pygame.time.Clock()
    display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

    while 1:
        ret, frame = cap.read()
        if not ret: 
            cap.release()
            break
        
        frame = cv2.resize(frame, dsize=(320, 160), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (len(images) >= 12):
            del images[0]
        images.append(gray)

        if len(images) == 12:
            
            clock.tick_busy_loop(30)
            pygame.display.flip()
            pygame.event.pump()
            
            images_arr = np.array(images)
            images_arr.resize(1, 6, 320, 160)
            data = json.dumps({"data": images_arr.tolist()})
            data = np.array(json.loads(data)["data"]).astype("float32")

            input_imgs = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            result = session.run([output_name], {input_imgs: data})[0].tolist()[0]

            driver_state = {
                "faceProb": result[12],
                "partialFace": result[35],
                "eyesProb": "{:.4f}   {:.4f}".format(result[21], result[30]),
                "blink_prob": "{:.4f}   {:.4f}".format(result[31], result[32]),
                "distracted_eyes": result[37]
            }

            render(display, frame, driver_state)
