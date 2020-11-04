import cv2
import time
import os
import numpy as np
import emotions as em

def capture_image(file_path):
    #capture image
    
    current_folder = os.getcwd()
    crop = os.path.join(current_folder,'croped\\croped.jpg')
    imotion_images = os.path.join(current_folder,'emotion_images')
    emotion_list = []
    name = file_path.split(".")[0]
    key = cv2. waitKey(1)
	#webcam = cv2.VideoCapture(0) # Webcam source
    cap = cv2.VideoCapture(file_path)#video file source

    start_time = time.time()
    count = 0
    
    
    while True:
        try:
            ret, frame = cap.read()# captures frame and returns boolean value and captured image
            if ret == False:
                return name,emotion_list
                break
            
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('frame',gray)
            
            cv2.imshow("Capturing", frame)
            end_time = time.time()

            duration = time.strftime("%S", time.gmtime(end_time - start_time))
            key = cv2.waitKey(25)
            
            if duration == '20':
                check, img = cap.read()
                cv2.imwrite(filename=crop, img=img)
                start_time = end_time
                value = em.predict_emotion(frame)
                cv2.imwrite(os.path.join(imotion_images,"{}_{}.jpg".format(value,count)),frame)
                count +=1
                emotion_list.append(value)

                
            elif key == ord('q'):#wait until 'q' key is pressed
                print("video off.")
                print("Program ended.")
                cap.release()
                cv2.destroyAllWindows()
                return name,emotion_list
                break
            
        except(KeyboardInterrupt):
            print("Turning off video.")
            cap.release()
            print("video off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
