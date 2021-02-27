import os
import cv2
import shutil

#!!!!!!!!!!!!!!! i download the videos to my computer!!!!!!!!!!!!!!!!!!!!
def cleanName(title):
    title = title.replace("\\", '/')
    title = title.replace("/Fall 2020/biometrics/proj/F20Data/", '')
    title = title.replace("/", '_')
    title = title.replace(".mp4", '_')
    title = title.replace(".MOV", '_')
    title = title.replace(".mov", '_')
    title = title.replace(".MP4", '_')
    return title

# the path of all of the videos in my computer
path = ''

# the name of the videos im interested should be added here
tasks = [ ]

# creating a folder whe i can keep all of the frames
os.mkdir("")

# the frames that im getting from each video
randnums= [90, 180, 270, 360, 450, 540, 630, 720, 810, 900]


for root, directories, files in os.walk(path, topdown=False):
    
    # get information of the file to create a name for the persons videos folder
    personsFrames = root
    #reaplce the neme with something new to make the file cleaner
    personsFrames = personsFrames.replace("", '')
    personsFrames = personsFrames+"_frames"
    # creating the folder of the person 
    os.mkdir(personsFrames)
    
    # iterating though all of the files inside the folder (the videos)
    for name in files:
        
        # if the name from the file is in my list of videos i want i get the frames
        if name in tasks:
            
            # code form the internet
            title = str(os.path.join(root,name))
            vidcap = cv2.VideoCapture(title)
            success,image = vidcap.read()
            count = 0
            
            #making the name of the picture nicer
            title = cleanName(title)
            
            # going thorugh all of the videos and getting all of the frame
            while success:
                
                # looking for the frames from my list
                if count in randnums:
                    success,image = vidcap.read()
                    cv2.imwrite(title+"frame%d.jpg" % count, image)     # save frame as JPEG file
                        
                    # moving the picture into their own folder
                    temp = str(title+"frame%d.jpg" % count)
                    shutil.move(temp, personsFrames)
                    
                # after i get 900 frames i go to the next video
                if count == 900:
                    success = False
                    
                #count for the next frame
                count += 1
    # moving the folder of the persons frames to my overl all folder
    #add the name of the new locaiton
    shutil.move(personsFrames, "")
    