# Image Processing Tools for the Project

def image_filtering(image):
    image = bilateralFilter(image, 5, 50, 100) #     smoothing filter
    image = GaussianBlur(image, (7, 7), 0)     ####### Gaussian Blur ###########


def image_bg_average(img):
    global bg
    if bg is None :
        bg = img.copy().astype("float")
        print("## Running Initial Background Scan . . . ")
        return
    cv2.accumulateWeighted(img,bg,0.3)
#weight(0.3 now) is update bg speed(how fast the accumulator “forgets” about earlier images)
    if i==49:
        print(" Scan Done...Ready For Threshold ##")

def image_segmenter(img):
    global bg
    #cv2.accumulateWeighted(img,bg,0.001)
    diff = cv2.absdiff(bg.astype("uint8"),img)
    #th1 = cv2.threshold(diff,30,255,cv2.THRESH_BINARY) [1]
    #cv2.imshow("Binary only",th1)
    thres = cv2.threshold(diff,30,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
    thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)      #Opening
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8) )     #Closing
    cnts,_ = cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segment = max(cnts, key=cv2.contourArea)
        return (thres,segment)

def blob_tracker(c,frame):
    global i
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if i==50:
        for z in range (1,33):
            pts.append(center)
        i+=1
    #print(center)
    if radius > 10:
        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
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

        thickness = int(np.sqrt(32/ float(index + 1)) * 2)
        cv2.line(frame, pts[index - 1], pts[index], (0, 0, 255), thickness)

    cv2.putText(frame, direction, (150,frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 1)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)
