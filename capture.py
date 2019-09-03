print("\t\t\t##Running ","capture-test.py##");
try:
    print("Importing Packages Required...",end="##")
    import cv2
    import numpy as np
    print("...Import Sucessful")
except:
    print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

print(" Setting Global Variables...",end=' ')
currentframe=1
print("## Done.")

if __name__ == "__main__":
    print("** Hello from Main of 'capture-test.py' **")
    
    print("Opening Camera...")
    camera =cv2.VideoCapture(0)
    while camera.isOpened():
        ret, frame = camera.read()
        
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)
        
        cv2.rectangle(frame,(370,90),(630,400),(255,0,0),0)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Region of Interest',(350,80),font,1,(0,70,250),2,cv2.LINE_AA)
        
        crop = frame[90:400,370:630]
        
        cv2.imshow('Live Feed',frame)
        cv2.imshow('Cropped',crop)
        
        keypress = cv2.waitKey(2) & 0xFF
        if keypress == ord("q"):
            print("Exit Key Pressed")
            break
        if keypress == ord("b"):
            print("   ## Running Background Scan...",end=' ')
            

            
        if keypress == ord("c"):
            name = 'Capture' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name,crop)
            currentframe += 1
    camera.release()
    cv2.destroyAllWindows()
        

print("Done")
