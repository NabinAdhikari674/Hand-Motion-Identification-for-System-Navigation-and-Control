print("\t\t\t##Running ","capture-test.py##");
try:
    print("Importing Packages Required...",end="##")
    import cv2
    import numpy as np
    import gesture_verify as gvf
    import PIL
    print("...Import Sucessful")
except:
    print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

print("Setting Global Variables...",end=' ')
gesture_model=gvf.ModelLoader()
names = ['1','2','3','4','5','Palm','Fist']
currentframe=1
i=0
bg=None
kernel = np.ones((2,2),np.uint8) #kernel for Opening and Closing morphologyEx
print("## Done.")

def average(img):
    global bg
    if bg is None :
        bg = img.copy().astype("float")
        print("## Running Initial Background Scan . . . ")
        return
    cv2.accumulateWeighted(img,bg,0.3)
#weight(0.3 now) is update bg speed(how fast the accumulator “forgets” about earlier images)
    if i==49:
        print(" Scan Done...Ready For Threshold ##")

def segmenter(img):
    global bg
    #cv2.accumulateWeighted(img,bg,0.001)
    diff = cv2.absdiff(bg.astype("uint8"),img)

    thres = cv2.threshold(diff,25,255,cv2.THRESH_BINARY) [1]
    thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)      #Opening
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8) )     #Closing
    #thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8) )  
    cnts,_ = cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segment = max(cnts, key=cv2.contourArea)
        return (thres,segment)

if __name__ == "__main__":
    print("\n\t\t** Hello from Main of 'capture-test.py' **")

    print("Opening Camera...")
    camera =cv2.VideoCapture(0)
    while camera.isOpened():
        ret, frame = camera.read()

        frame = cv2.bilateralFilter(frame, 5, 50, 100)#smoothing filter
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame,(370,90),(630,400),(255,0,0),0)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Region of Interest',(350,80),font,1,(0,70,250),2,cv2.LINE_AA)

        crop = frame[90:400,370:630]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if i<50:
            average(gray)
            i=i+1
        else:
            sg_return = segmenter( gray )
            if sg_return is not None:
                (thres,segment) = sg_return
                #print(thres)
                resized_image = cv2.resize(thres, (150, 150))
                image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
                pred_ges=gvf.GesturePredictor(gesture_model,image)
                #print(pred_ges)
                pred_ges=pred_ges.tolist()
                maxele=max(pred_ges[0])
                #print('Max is : ',pred)
                index= pred_ges[0].index(maxele)
                #print("Index is : ",index)
                gesture=names[index]
                #print("The prediction is : ",gesture)
                #cv2.imshow("Resized Image in RGB",image)
                cv2.imshow("Binary Threshold",thres)
                cv2.drawContours(crop,segment, -1, (0, 255, 25),3)
                cv2.putText(frame,gesture,(370,120),font,1,(0,70,250),2,cv2.LINE_AA)


        cv2.imshow('Live Feed',frame)
        #cv2.imshow('Cropped',crop)

        keypress = cv2.waitKey(2) & 0xFF
        if keypress == ord("q"):
            print("Exit Key Pressed")
            break
        if keypress == ord("b"):
            print("   ## Running Background Scan...",end=' ')
            cv2.accumulateWeighted(gray,bg,0.4)
            print(" .. DONE  ##")
        if keypress == ord("c"):
            name = 'Capture' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name,crop)
            currentframe += 1
        if keypress == ord("x"):
            name = 'DATA/BlackCapture' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name,thres)
            currentframe += 1
            
    camera.release()
    cv2.destroyAllWindows()

print("Done")
