import pyautogui, sys
import global_var_tunnel as gv
print("============= UI Controller Imported ==========")
gestures,actions,infoDict = gv.get_global_values('Gestures','information')
#print(infoDict)
(sizex,sizey) = pyautogui.size()



def mouseMovement(a=0,b=0):
    global sizex,sizey
    try:
        x, y = pyautogui.position()
        if a>0:
            newx = x+10
        elif a<0:
            newx = x-10
        if b>0:
            newy = y-10
        elif b<0:
            newy = y+10

        if(newx>sizex):
            newx=sizex
        if(newy>sizey):
            newy=sizey

        pyautogui.moveTo(newx, newy)

    except Exception as exp:
        pass
        #print ('ERROR in mcontroller.py!!\n\tError:\n\t',exp)


def mouseDrag(a=0,b=0,drag=None):
    global sizex,sizey
    try:
        x, y = pyautogui.position()
        newx = x+a
        newy = y+b
        if(newx>sizex):
            newx=sizex
        if(newy>sizey):
            newy=sizey

        if drag=='left':
            pyautogui.dragTo(newx, newy, button='left')
        if drag=='right':
            pyautogui.dragTo(newx, newy, button='right')
    except Exception as exp:
        print ('ERROR in mcontroller.py (mouseDrag)!!\n\tError:\n\t',exp)

def mouseClick(click=None):
    if click=='left':
        pyautogui.click()
    if click=='right':
        pyautogui.click(button='right')

def mouseScroll(scroll=None,ticks=10):
    if scroll=='top':
        pyautogui.scroll(ticks)
    if scroll=='down':
        pyautogui.scroll(-ticks)
def keybordFunction(key=None):
    if(key=='shift'):
        pyautogui.keyDown('shift')
    if(key=='enter'):
        pyautogui.press('enter')
    if(key=='win'):
        pyautogui.press('win')
    if(key=='f1'):
        pyautogui.press('f1')
    if(key=='f2'):
        pyautogui.press('f2')
    if(key=='f3'):
        pyautogui.press('f3')
    if(key=='f4'):
        pyautogui.press('f4')
    if(key=='f5'):
        pyautogui.press('f5')
    if(key=='f6'):
        pyautogui.press('f6')
    if(key=='f7'):
        pyautogui.press('f7')
    if(key=='f8'):
        pyautogui.press('f8')
    if(key=='f9'):
        pyautogui.press('f9')
    if(key=='f10'):
        pyautogui.press('f10')
    if(key=='f11'):
        pyautogui.press('f11')
    if(key=='f12'):
        pyautogui.press('f12')
    if(key=='esc'):
        pyautogui.press('esc')
    if(key=='delete'):
        pyautogui.press('delete')
    if(key=='left'):
        pyautogui.press('left')
    if(key=='right'):
        pyautogui.press('right')
    if(key=='up'):
        pyautogui.press('up')
    if(key=='down'):
        pyautogui.press('down')
    if(key=='capslock'):
        pyautogui.press('capslock')
    if(key=='ctrl'):
        pyautogui.press('ctrl')
    if(key=='space'):
        pyautogui.press('space')
    if(key=='home'):
        pyautogui.press('home')
    if(key=='spec1'):
       pyautigui.hotkey('ctrl','a')
    if(key=='playpause'):
       pyautogui.press('playpause')
    if(key=='prevtrack'):
       pyautogui.press('prevtrack')
    if(key=='nexttrack'):
       pyautogui.press('nexttrack')
    if(key=='browserforward'):
       pyautogui.press('browserforward')
    if(key=='browserback'):
       pyautogui.press('browserback')
    if(key=='browsersearch'):
       pyautogui.press('browsersearch')
    if(key=='browserrefresh'):
       pyautogui.press('browserrefresh')
    if(key=='browserhome'):
       pyautogui.press('browserhome')

def mainController(gesture=None):
    #print('Gesture : ',gesture)
    action=infoDict[gesture]
    action='MouseMovement'
    #print("Action : ",action)
    if action == 'MouseMovement':
        a = int(gv.get_global_values('dX','global')[0])
        b = int(gv.get_global_values('dY','global')[0])
        mouseMovement(-a,b)
    #if gesture=='2':
    #    mouseClick('left')
    #if gesture=='3':
    #    mouseClick('right')
    #if gesture=='4':
    #    mouseScroll('top')
    #if gesture=='5':
    #    mouseScroll('down')
    #if gesture=='ok':
    #   keyBoardFunction()




#while True:
#    mainController('1',2,3)
