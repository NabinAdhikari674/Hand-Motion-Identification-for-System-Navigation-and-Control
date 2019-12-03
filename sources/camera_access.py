# Video Capture and Processing Module for the Project
# IMPORT
try:
    print("Importing for \"camera_access\"")
    from collections import deque
    from cv2 import VideoCapture,bilateralFilter,flip,rectangle,putText,cvtColor,GaussianBlur,waitKey,absdiff,threshold,findContours,line,minEnclosingCircle
    from cv2 import FONT_HERSHEY_SIMPLEX,LINE_AA,COLOR_BGR2GRAY,resize,imshow,imwrite,drawContours,accumulateWeighted,morphologyEx,circle,moments
    from cv2 import THRESH_BINARY,THRESH_OTSU,MORPH_OPEN,MORPH_CLOSE,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE,contourArea,destroyAllWindows
    import numpy as np
    import pandas as pd
    from tensorflow import cast,float32
    #Local Imports
    import gesture_verify as gvf
    import global_var_tunnel as gv

except Exception as exp:
    print("\t!!! Error in \"camera_opener\" from module \"camera_access.py\" !!!")
    print("\t Error : \n\t\t",exp)
# Global Variables Setting
try:
    print("\tSetting Global Variables...",end=' ')
    gesture_model=gvf.ModelLoader()
    information_center_data = pd.read_csv('dataBase/information_center.csv')
    gesture_names = information_center_data['Gestures'].tolist()
    gesture_names = [i for i in gesture_names if i == i]
    print("\tThe Current Available Gestures are : ",gesture_names)
    currentframe=1
    #gesture_predict_mode=gv.gesture_predict_mode
    i=0
    bg=None
    flag1=0
    kernel = np.ones((2,2),np.uint8) #kernel for Opening and Closing morphologyEx
    pts = deque(maxlen=32)
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""
    print("## Done.")
except Exception as exp:
    print("\t!!! Error in \"camera_opener\" from module \"camera_access.py\" !!!")
    print("\t Error : \n\t\t",exp)

def image_bg_average(img):
    global bg
    if bg is None :
        bg = img.copy().astype("float")
        print("## Running Initial Background Scan . . . ")
        return
    accumulateWeighted(img,bg,0.3)
    #weight(0.3 now) is update bg speed(how fast the accumulator “forgets” about earlier images)
    if i==49:
        print(" Scan Done...Ready For Threshold ##")

def image_segmenter(img):
    global bg
    #cv2.accumulateWeighted(img,bg,0.001)
    diff = absdiff(bg.astype("uint8"),img)
    #th1 = cv2.threshold(diff,30,255,cv2.THRESH_BINARY) [1]
    #cv2.imshow("Binary only",th1)
    thres = threshold(diff,30,255,THRESH_BINARY + THRESH_OTSU) [1]
    thres = morphologyEx(thres, MORPH_OPEN, kernel)      #Opening
    thres = morphologyEx(thres, MORPH_CLOSE,np.ones((3,3),np.uint8) )     #Closing
    cnts,_ = findContours(thres,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segment = max(cnts, key=contourArea)
        return (thres,segment)

def blob_tracker(c,frame,frame_full):
    global i
    ((x, y), radius) = minEnclosingCircle(c)
    M = moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if i==50:
        for z in range (1,33):
            pts.append(center)
        i+=1
    #print(center)
    if radius > 10:
        circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
        #print('The point is ',pts)
    for index in np.arange(1, len(pts)):
        #print("here")
        # if either of the tracked points are None, ignore
        # them
        if pts[index - 1] is None or pts[index] is None:
                continue
        if index == 1 and pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = pts[-10][0] - pts[index][0]
            dY = pts[-10][1] - pts[index][1]
            (dirX, dirY) = ("", "")

            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"

            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

            # otherwise, only one direction is non-empty
            else:
                    direction = dirX if dirX != "" else dirY
            #print(dirX,dirY)

        thickness = int(np.sqrt(32/ float(index +2))*1.5)
        line(frame_full, pts[index-1], pts[index], (250, 0, 25), thickness)

    #print(dX,dY)

    #gv.update('direction',direction,'global')
    putText(frame_full, direction, (500,frame.shape[0]+85),FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 1)
    putText(frame_full, "dx: {}, dy: {}".format(dX,dY),
            (380, frame.shape[0]+85),FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)
    gv.update('dX',dX,'global')
    gv.update('dY',dY,'global')
    return frame

def camera_opener():
    print("Running \"camera_opener\" from module \"camera_access.py\"")
    global bg,i,currentframe,flag1,gesture_names,kernel
    print("\tOpening Camera...")
    gesturePredictMode = gv.get_global_values('gesturePredictMode','global')
    uiControlMode = gv.get_global_values('uiControlMode','global')
    print("<< gesturePredictMode :: ",gesturePredictMode," >>  << uiControlMode :: ",uiControlMode," >>")
    if uiControlMode==['True']:
        import ui_controller as uc
    camera = VideoCapture(0)
    while camera.isOpened():
        _, frame = camera.read()
        frame = flip(frame, 1)
        rectangle(frame,(370,90),(630,400),(255,0,0),0)
        putText(frame,'Region of Interest',(350,80),FONT_HERSHEY_SIMPLEX,1,(0,70,250),2,LINE_AA)

        frame_roi = frame[90:400,370:630]
        frame_roi = bilateralFilter(frame_roi, 5, 50, 100) #     smoothing filter
        frame_roi = GaussianBlur(frame_roi, (7, 7), 0)     ####### Gaussian Blur ###########

        #gray=frame_roi
        gray = cvtColor(frame_roi, COLOR_BGR2GRAY)
        gray = GaussianBlur(gray, (7, 7), 0)     ####### Gaussian Blur ###########


        keypress = waitKey(2) & 0xFF
        if keypress is not 0xFF:
            print("\t KeyPressed : ",keypress)
            if keypress == ord("c"):
                name = '../generatedData/Capture' + str(currentframe) + '.jpg'
                print ('Creating...' + name)
                imwrite(name,crop)
                currentframe += 1
            if keypress == ord("p"):
                flag1=1
            if keypress == ord("o") or currentframe==500:
                flag1=0
            if keypress == ord("r"):
                currentframe=1
            if flag1==1:
                name = '../generatedData/ContCapture' + str(currentframe) + '.jpg'
                print ('\t Creating...' + name)
                imwrite(name,crop)
                currentframe += 1

        while i<50:
            image_bg_average(gray)
            i=i+1
        segmenterReturn = image_segmenter( gray )
        if segmenterReturn is not None:
            (thres,contours) = segmenterReturn

            #extLeft = tuple(contours[c[:, :, 0].argmin()][0])
            #extRight = tuple(contours[contours[:, :, 0].argmax()][0])
            extTop = tuple(contours[contours[:, :,1].argmin()][0])
            #extBot = tuple(contours[contours[:, :, 1].argmax()][0])
            circle(frame_roi,extTop, 8, (255, 0, 0), -1)

            if gesturePredictMode==['TRUE']:
                predictions = gvf.GesturePredictor(gesture_model,thres) ## PREDICTION
                predictions = predictions.tolist()
                #print(predictions)
                maxPrediction = max(predictions[0])
                index = predictions[0].index(maxPrediction)
                gesture = gesture_names[index]
                #print(gesture)
                msg = str(gesture+' : '+str(maxPrediction*100)[:-12]+" % ")
                if uiControlMode==['True'] and gesture!='None':
                    drawContours(frame_roi,contours, -1,(100, 25,20),2,LINE_AA)
                    blob_tracker(contours,frame_roi,frame)
                    uc.mainController(gesture)     #========================================================
                #else :
                #    print(gv.get_global_values('uiControlMode','global'),'::',gesture)
                #print(gesture)
                putText(frame,msg,(370,120),FONT_HERSHEY_SIMPLEX,0.8,(0,70,250),2)
                imshow("Region of Interest in RGB",frame_roi)
                imshow("Binary Threshold",thres)

            #else :
            #    print('here : ',gv.get_global_values('gesturePredictMode','global'))




            #cv2.putText(frame,gesture2,(370,120),font,1,(0,70,250),2,cv2.LINE_AA)

        imshow('Live Feed from WebCam',frame)
        #cv2.imshow('Cropped',crop)

        if keypress is not 0xFF :
            #print("\t KeyPressed : ",keypress)
            if keypress == ord("q"):
                print("Exit Key Pressed")
                break
            if keypress == ord("b"):
                print("   ## Running Background Scan...",end=' ')
                accumulateWeighted(gray,bg,0.4)
                print(" .. DONE  ##")
            if keypress == ord("x"):
                name = '../generatedData/BlackCapture' + str(currentframe) + '.jpg'
                print ('Creating...' + name)
                imwrite(name,thres)
                currentframe += 1

    camera.release()
    destroyAllWindows()


if __name__ == "__main__":
    camera_opener()
elif __name__ == "sources.camera_access":
    print("\"camera_access.py\"Imported as : ",__name__)
else:
    print("\"camera_access.py\" - !! UnKnown Import Detected as !! : ",__name__)
