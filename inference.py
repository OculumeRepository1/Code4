import torch 

from util import load_data
from util import load_model
import cv2
from ByteTrack import Tracker
from datetime import datetime
from Yolov7 import Yolo_model
from util import email
from util import open_multiple_cameras
from util import image_processing
import time
import threading
from queue import Queue
from util import email_worker
import pandas as pd
import numpy as np
from util import can_send_email
import os
from vllm import tinyVision
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import ultralytics
import ollama
from util import sms_send
import ollama
import base64
from vllm import ollama_model
from vllm import ollama_QA
import pygame
from ultralytics import YOLO
import os
from streamgrid import StreamGrid
# # Initialize pygame once
# pygame.mixer.init()
from util import motion_detection
data_path=r"C:\Users\oosma\Oculume_Codes\Dashboard\Code3\data.csv"
data_log=load_data(data_path)
fontScale = 1
font = cv2.FONT_HERSHEY_COMPLEX
# Blue color in BGR 
color = (0, 0, 255) 
# Line thickness of 2 px 
thickness = 2
#Bounding Boxes color scheme
ALPHA = 0.2
TABLE_BORDER = (0, 0, 255) 

# Initialize the current page if not already set
###############Yolov9 Model##############################
#model=load_model("model")
All_variables={}


# Load a pretrained YOLO11n model
###################Yolov11###########################
model = YOLO(r"C:\Users\oosma\Oculume_Codes\Dashboard\Code3\best_11.pt")


class_names=model.names
##########################Camera Index#############################
#camera_index=[0,'rtsp://admin:Oculume2024@10.187.0.101/Preview_01_sub']
#video=rf"C:\Users\oosma\Oculume_Codes\Dashboard\code\Video.mp4"

#camera_index=[0,'rtsp://admin:oculume2025@192.168.1.247:554/Preview_01_sub']
#camera_index=['rtsp://test:test1@10.112.136.182/axis-media/media.amp?camera=3&videocodec=h264&resolution=640x480&fps=5',
#'rtsp://test:test@10.112.136.66/axis-media/media.amp?camera=1&videocodec=h264&resolution=640x480&fps=5',
#'rtsp://test:test@10.112.137.34/axis-media/media.amp?camera=3&videocodec=h264&resolution=640x480&fps=5',
#'rtsp://test:test@10.112.136.108/axis-media/media.amp?camera=1&videocodec=h264&resolution=640x480&fps=5']
camera_index=[0,'rtsp://admin:Oculume2024@192.168.1.95/Preview_01_sub']

#camera_index=[0]
reg_list = [[] for n in camera_index]
cameras=open_multiple_cameras(camera_index)
end_times_database = [0 for n in camera_index]
end_times = [0 for n in camera_index]
start_times_database = [0 for n in camera_index]
start_times = [0 for n in camera_index]
print(end_times)
print(cameras.keys())
#########################Change the location accordingly#######################
#locations=['100_Street_Entracne','LRT_Pedway','BAY_Entrance','101st_INT_PTZ']
locations=['100_Street_Entracne','ghar']
width=640
height=480
#############################





# prompt="Indicate if there is any weapon present. Only mention a weapon if you are completely certain. Begin the response with 'YES' if a weapon is detected. Avoid using the word 'image' in your description."
#prompt2="Analyze the image carefully and describe the situation in detail. Indicate if there are any guns, weapons, knife, or any thing in mouth or smoking gesture or smoking substances or person lying on the floor or person fallen — but only mention them if you are completely certain, based on clear visual evidence. Do not speculate or fabricate. If a weapon is confidently detected, specify the type of weapon observed. Then, describe: What is happening in the image (the situation) and the actions being taken by individuals.Any visible information about the individuals (e.g., clothing, posture, interaction. Also my YOLO model is detecting  \"\"\"{text}\"\"\" so keep this detection in consideration while giving the response"
#prompt="Elaborate what is happeing in the image(situation), what action are being taken and also provide information on the individual avaible in the image"
#prompt1="You are an AI agent whose job is to identify Crime or any sort of weapone in the scene. If any of these conditions are identify. 1: Check for weapons, knife any other kind of weapons. 2: person is smoking, cigerete. 3: Person lying on the floor or person fallen. 4: Violence; if any are present, Just response with 'Yes'or 'No'."
#prompt1="If any of these conditions are identify. 1: Check for weapons, knife any other kind of weapons. 2: person is smoking, cigerete. 3: Person lying on the floor or person fallen. 4: Violence; if any are present, start with 'Yes' and describe clearly, otherwise start with 'No' and summarize the situation and actions of the individuals without speculation."
# prompt3="""You are an AI agent whose job is to identify Crime scene or if person is fallen in anyway . Analyze the description and determine whether a person has fallen. If a fall is detected, describe the following:
# 	1.	The position of the person (e.g., lying on the floor, slumped over a surface)
# 	2.	Visible signs of distress or injury
# 	3.	Any environmental context that might have led to the fall (e.g., wet floor, stairs, crowding)
# 	4.	Whether the person appears conscious or unconscious
# 	5.	Urgency of medical attention based on visual clues
# Start your response with ‘FALL DETECTED’ if a fall is clearly visible. Otherwise, begin with ‘NO FALL DETECTED’. Avoid assumptions if the visual evidence is unclear."""

# Define the number of worker threads
# notification_classes = {
#                     0: 0.45,  # Class 0 threshold
#                     1: 0.40,  # Class 1 threshold
#                     2: 0.60,  # Class 2 threshold
#                     3: 0.65,  # Class 3 threshold
#                     4: 0.40,  # Class 4 threshold
#                     5: 0.45,  # Class 5 threshold
#                     6: 0.40,  # Class 6 threshold
#                                                 }
notification_classes = {
                    0: 1,  # Class 0 threshold
                    1: 0.9,  # Class 1 threshold
                    2: 0.1,  # Class 2 threshold
                    3: 0.2,  # Class 3 threshold
                    4: 0.2,  # Class 4 threshold
                    5: 0.2,  # Class 5 threshold
                    6: 0.2,  # Class 6 threshold
                                                }
##email Rate Limiting
email_interval = 4  # Minimum seconds between emails


#########################Uploading Moondream Vision Model#################################################
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"  # Pin to specific version
# # vmodel = AutoModelForCausalLM.from_pretrained(
# #     model_id, trust_remote_code=True, revision=revision
# # )

# # For gpus 
vmodel = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16
    #attn_implementation="flash_attention_2"
).to("cuda")


tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
#email_recipents=['notify@oculume.ai','smancini@edmontoncitycentre.com','pmyslicki@paladinsecurity.com','choggins@edmontoncitycentre.com','mbansal@paladinsecurity.com','mhowse@paladinsecurity.com']
##########################################################################
email_recipents=['notify@oculume.ai','haris@oculume.ai']
if not cameras:
    print("No Camera were Opened. Exiting.")
for key in cameras.keys():
    All_variables[f'reg_list_{key}']=reg_list[key]
    All_variables[f'tarcker_{key}']=Tracker()
    All_variables[f'email_message_{key}']= Queue()
    All_variables[f'email_image_{key}']= Queue()
    All_variables[f'threads_{key}']=threading.Thread(target=email_worker,args=[All_variables[f'email_message_{key}'],All_variables[f'email_image_{key}'],email_recipents])
    All_variables[f'threads_{key}'].daemon=True
    All_variables[f'threads_{key}'].start()
    All_variables[f'last_email_time_{key}']=0
    All_variables[f'Frame_{key}']=None
    All_variables[f'last_mean_{key}']=0
    All_variables[f'last_mean_time{key}']=0
    All_variables[f'location_{key}']=locations[key]
    All_variables[f'email_recipients_{key}']=email_recipents
    All_variables[f'Black_frame_{key}'] = np.zeros((height,width,3),dtype=np.uint8)  # Initialize last mean for each camer
    All_variables[f'cls_{key}'] = ''  # Initialize last mean for each camera
    image_path = rf"C:\Users\oosma\Oculume_Codes\Dashboard\Code4\static\images\static_image_{key}.jpg"
    if not os.path.exists(image_path):
        print("Static Image does not exist. Taking picture...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(image_path, frame)
            print(f"Picture saved at {image_path}")
            image=cv2.imread(image_path)
            image=cv2.resize(image, (width, height))
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            All_variables[f'Static_Image_{key}'] = image
        else:
            print("Failed to take picture.")
        cap.release()
    else:
        image=cv2.imread(image_path)
        image=cv2.resize(image, (width, height))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        All_variables[f'Static_Image_{key}'] = image

end=0
start_time= datetime.now()
Detection=[]
confidence_threshold = 0.30
last_detection_time = {key: 0 for key in cameras.keys()}
window_timeout = 14  # seconds to keep window open after last detection
window_active = {key: False for key in cameras.keys()}

# Initialize the start and end times for each camera
num_cameras = len(locations)
sound_played = [False] * num_cameras

    
   
while True:
    try:
        current_time = datetime.now().strftime('%H:%M:%S')
        current_date = datetime.now().date()
        save_dir=rf'{current_date}'
        database=rf"C:\Users\oosma\Oculume_Codes\Dashboard\Code4\Database\{save_dir}"
        os.makedirs(database, exist_ok=True)
        for idx,cap in cameras.items():
            ret,frame=cap.read()
            frame = cv2.resize(frame, (width, height))
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if ret:
                #print("Yes")dsa
                lis=All_variables[f'reg_list_{idx}']
                image1=frame.copy()
                image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
                start_times_database[idx]=time.time()
                ################################Storing Data for Training every 10min#########################
                if start_times_database[idx]-end_times_database[idx]>600:
                     rand_nda=np.random.randint(1,5000000+1)
                     save_loc_database=rf'{database}\{save_dir}_{rand_nda}.jpg'
                     cv2.imwrite(save_loc_database,image1)
                     end_times_database[idx]=time.time()
            
                # Motion Detection and Tracking
                # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                # gray = cv2.GaussianBlur(gray, (21, 21), 0)
                # if isinstance(All_variables[f'last_mean_{idx}'], int):
                #     All_variables[f'last_mean_{idx}'] = gray
                #     md_result = 0.0
                # else:
                #     md_result = np.mean(cv2.absdiff(gray, All_variables[f'last_mean_{idx}']))
                #     All_variables[f'last_mean_{idx}'] = gray
                # # md_result=np.mean(cv2.absdiff(gray,All_variables[f'last_mean_{idx}']))
                # # #md_result = float(np.mean(np.abs(gray.astype(np.float32) - All_variables[f'last_mean_{idx}'])))
                # # All_variables[f'last_mean_{idx}']=gray
                # print(f"Motion Detection Result {md_result}")
                # last_mean_time=time.time()-All_variables[f'last_mean_time{idx}']
                Detection=motion_detection(All_variables[f'Static_Image_{key}'],frame,threshold=30)
                #Detection=False
                #print(f"Motion Detection Result {Detection}")
                if Detection:
                    #print("Motion Detected")
                    image,results,classes,confs=image_processing(frame,data_log,model,All_variables[f'tarcker_{idx}'],font,fontScale,color,thickness,locations[idx])
                    for ids,cls,conf in zip(results,classes,confs):
                # print(f"Tracking before update {ids} {reg_list} and idx {idx}")
                        Text=f'{cls}'
                        prompt2="Analyze the image carefully and describe the situation in detail and ignore non violent behavior. Indicate if there are any guns, weapons, knife, or any thing in mouth or smoking gesture or smoking substances or person lying on the floor or person  or fighting stance — but only mention them if you are completely certain, based on clear visual evidence. Do not speculate or fabricate. If a weapon is confidently detected, specify the type of weapon observed. Then, describe: What is happening in the image (the situation) and the actions being taken by individuals.Any visible information about the individuals (e.g., clothing, posture, interaction. Also my YOLO model is detecting  \"\"\"{text}\"\"\" so keep this detection in consideration while giving the response"
                        # Get keys corresponding to the value
                        class_id = [key for key, value in class_names.items() if value == cls]
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        #image=cv2.putText(image, Text,(int(ids[0]),int(ids[1])),cv2.FONT_HERSHEY_TRIPLEX,0.8,color,1)
                        #image = cv2.putText(image, f'{ids[4]}',(int(ids[0])+10,int(ids[1])),cv2.FONT_HERSHEY_TRIPLEX,0.8,color,1)
                        start_times[idx]=time.time()
                        t=start_times[idx]-end_times[idx]  
                        if ids[4] not in lis and start_times[idx]-end_times[idx]>5:
                            if conf > notification_classes[class_id[0]]:
                                # We need a new DataFrame with the new contents
                                rand_n=np.random.randint(1,5000+1)
                                save_dir_dataset=rf'Dataset\{save_dir[0:4]+save_dir[5:7]+save_dir[8:10]}'
                                dataset=rf"C:\Users\oosma\Oculume_Codes\Dashboard\Code4\{save_dir_dataset}"
                                os.makedirs(dataset, exist_ok=True)
                                save_loc=rf'{dataset}\{save_dir[0:4]+save_dir[5:7]+save_dir[8:10]}{rand_n}.jpg'
                                save_loc1=rf'{save_dir_dataset}\{save_dir[0:4]+save_dir[5:7]+save_dir[8:10]}{rand_n}.jpg'
                                #save_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                                # Call Pandas.concat to create a new DataFrame from several existing ones
                                reg_list.append(ids[2])
                            #    print(f"Tracking after update {ids} {reg_list} and idx {idx}")
                                # print(f"Tracking After update{idx} {reg_list}")
                                end_times[idx]=time.time()
                                body = f"{cls} is detected at {locations[idx]}".capitalize()
                            
                                if can_send_email(All_variables[f'last_email_time_{idx}'],email_interval):
                                    #color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                                    pil_image = Image.fromaqrray(frame)
                                    success, buffer = cv2.imencode('.jpg', frame)
                                    if not success:
                                        raise ValueError("Failed to encode image")

                                    # Convert to base64
                                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                                    #response=llava_model(prompt,pil_image)
                                    #response1=tinyVision(vmodel,pil_image,prompt1,tokenizer)
                                    #response2=tinyVision(vmodel,pil_image,prompt3,tokenizer)
                                    ##################Ollama Gemma################
                                    #response=ollama_model(prompt2,img_base64)
                                    #################This is MoonDream###############
                                    response=tinyVision(vmodel,pil_image,prompt2,tokenizer)
                                    response2=ollama_QA(response)
                                    #response2="yes"
                                    #response1=ollama_model(prompt1,img_base64)
                                    #print(f'First LLM Response {response1}')
                                    print(f'Second LLM Response {response2}')
                                    if response2[0]=="y":
                                        print("email send")
                                        #if response2=="FALL DETECTED" or response2=="Yes" or response1=='Yes' or response1=='1':
                                        #response=tinyVision(vmodel,pil_image,prompt2,tokenizer)
                                        data={
                                                'Date':current_date,
                                                'Time':current_time,
                                                'Detection':cls,
                                                'Location': str(locations[idx]),
                                                "Summary": str(response),
                                                'Image': save_loc,
                                                'Status': 'pending',
                                                }
                                        All_variables[f'cls_{key}'] = cls
                                        response= f"Location {str(locations[idx])}: {response}"
                                        print(response)
                                        data_log=pd.read_csv(data_path)
                                        df = pd.DataFrame(data,index=[0])
                                        data_log = pd.concat([df,data_log],ignore_index=True)
                                        data_log.to_csv(data_path, index=False)
                                                    # print(response[0:3])
                                                    # if response[0:2]=='Yes':
                                                    # Add the email task to the queue
                                        # All_variables[f'email_message_{idx}'].put((str(response)))
                                        # All_variables[f'email_image_{idx}'].put((image))
                                        #sms_send(response,image)
                                        cv2.imwrite(save_loc,image)
                                        #if response2[0] == "y":
                                        last_detection_time[idx] = time.time()
                                        window_active[idx] = True  # Activate the window                    
                else:
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)                          
                    All_variables[f'Frame_{idx}']=frame
                    #print("No Motion Detected")
                # Only show the window if detection is positive
                cv2.imshow("Frame", All_variables[f'Frame_{idx}'])
                # lis=list_Que.get()
                All_variables[f'reg_list_{idx}']=lis
                All_variables[f'location_{idx}']=locations[idx]
                # print(All_variables)
                # Auto-hide the window if no detection for window_timeout seconds
                current_time_sec = time.time()
                if window_active[idx]:
                    frame=All_variables[f'Frame_{idx}']
                    Text = f'Warning: {All_variables[f'cls_{key}']} Detected'
                    print(Text)
                    frame=cv2.putText(frame,Text,(10,60),cv2.FONT_HERSHEY_TRIPLEX,0.8,color,thickness)
                    frame=cv2.putText(frame, f'Location: {All_variables[f'location_{idx}']}', (10, 30), font, fontScale, (0, 0, 0), thickness)
                    # frame = cv2.putText(frame, All_variables[f'location_{idx}'], (10,50), font,fontScale,(0,0,0),thickness)
                    # All_variables[f'Frame_{idx}'] = frame
                        # 🔊 Play sound only once per activation
                        # if not sound_played[idx]:
                        #     try:
                        #         pygame.mixer.init()
                        #         sound_path = "C:/Users/oosma/Oculume_Codes/Dashboard/Code3/static/sounds/notification2.wav"
                        #         pygame.mixer.music.load(sound_path)
                        #         pygame.mixer.music.play()
                        #     except Exception as e:
                        #         print(f"Sound error: {e}")

                        #     sound_played[idx] = True
                    #cv2.imshow(All_variables[f'location_{idx}'], All_variables[f'Frame_{idx}'])
                    top=np.hstack((All_variables[f'Black_frame_0'], All_variables[f'Frame_0'])) 
                    bottom=np.hstack((All_variables[f'Black_frame_0'], All_variables[f'Frame_1'])) 
                    grid=np.vstack((top,bottom))
                    cv2.imshow("Video Surveillance", grid)
                    if current_time_sec - last_detection_time[idx] > window_timeout:
                        window_active[idx] = False
                        # sound_played[idx] = False 
                        try:
                            #cv2.destroyWindow(All_variables[f'location_{idx}'])
                            print(f"Destroying window for {All_variables[f'location_{idx}']}")
                            #cv2.destroyWindow(grid)
                            cv2.destroyAllWindows()
                        except:
                            pass
        #cv2.imshow(All_variables['location_1'],All_variables['Frame_1'])
    except Exception as e:
        error_msg = f"Unexpected error processing image: {str(e)}"
        print(error_msg)
        print("Camera is not Detected")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

                  