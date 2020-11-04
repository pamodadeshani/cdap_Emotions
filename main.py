import Capture as cap
import os
import cv2
import pickle
import csv
from datetime import datetime

def calculate_overall_emotion(name,emotion_list):
 # to keep count of each category   
    angry = 0
    disgusted = 0
    fearful = 0
    happy = 0
    sad = 0
    surprised = 0

    
    l = len(emotion_list)
    for i in range(l):
        if emotion_list[i] == "Angry":
            angry = angry + 1
        elif emotion_list[i] == "Disgusted":
            disgusted = disgusted + 1
        elif emotion_list[i] == "Fearful":
            fearful = fearful + 1
        elif emotion_list[i] == "Happy":
            happy = happy + 1
        elif emotion_list[i] == "Sad":
            sad = sad + 1
        elif emotion_list[i] == "Surprised":
            surprised = surprised + 1
        
    total = angry + disgusted + fearful + happy + sad + surprised
    
    if total == 0:
        total = 1
        
    #calculation
    print("Angry: {} \nDisgusted: {} \nFearful: {} \nHappy: {} \nSad: {} \nSurprised: {}".format(100*angry/total,100*disgusted/total,100*fearful/total,100*happy/total,100*sad/total,100*surprised/total))
    print("P:{}".format(100*(happy+surprised)/total))
    print("N:{}".format(100*(angry+disgusted+fearful+sad)/total))
    now = datetime.now() 
    data_list = [now.strftime("%x"),name,(100*(happy+surprised)/total),(100*(angry+disgusted+fearful+sad)/total)]
    
    if not (os.path.exists("{}.csv".format(name))):
        with open("{}.csv".format(name),'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date","Name","P","N"])
            
    with open("{}.csv".format(name), 'a+', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data_list)
            
            


name,em_lst = cap.capture_image('Hiruni.mp4')
calculate_overall_emotion(name,em_lst)
cv2.destroyAllWindows()
