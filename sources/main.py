# Main App Controller


# Index file as Main file of the Project
#from sources import camera_access

print("\n==================== Welcome to Hand Motion Detection for System Navigation and Control ======================")
try:
    import eel
    #import time
    #import pandas as pd
    import global_var_tunnel as gv
    eel.init('../webEngine')
except Exception as exp:
    print("Error when Importing eel and Starting GUI engine.")
try:
    print("")
    infoDict = {}
    bufferData = {}
    newModel = None
    gesture_verify = None
    trainGen = None
    validGen = None
    custom_utilities = None
    #import camera_access
    #import custom_utilities
    #import gesture_verify
    #import global_var_tunnel as gtunnel
    #import image_processor
except Exception as exp:
    print("Exeception in main.py\n\tError: \n\t",exp)




@eel.expose
def home11():
    try:
        import camera_access
        #import gesture_verify
    except Exception as exp:
        print("Exeception in main.py\n\tError(1): \n\t",exp)

    print("Home : Camera Requested.\n")
    camera_access.camera_opener()

@eel.expose
def viewGesture():
    global infoDict
    print("Gesture : Gesture Info Requested.")
    gestures,actions,infoDict=gv.get_global_values('','information')
    #print(gestures,'\nActions\n',actions)
    print("Main.py : Gesture Info Passed.\n")
    return gestures,actions
    #return infodict

@eel.expose
def readcv():
    b=20;
    return b

@eel.expose
def newGestureName(a):
    print("Gesture : Passed New Gesture Name is : ",a)
    global infoDict,bufferData
    bufferData = infoDict
    bufferData[a] = 'None'
    gv.update('gestures',bufferData,'buffer')
    print("Main.py : Added New Gesture Name to BufferInformation Zone. Proceed With Adding New Gesture.\n")
    return True

@eel.expose
def addGestureonNN():
    print("Gesture : Add Gesture on Neural Network.")
    global custom_utilities,newModel
    #from tensorflow.keras.models import load_model
    import custom_utilities as cutils
    custom_utilities = cutils

    modelUsed=gv.get_global_values('modelUsed')
    classes,_ = gv.get_global_values('Gestures','buffer')
    print("Classes : ",classes)
    print("Adding New Output Neuron to Model : ",modelUsed[0])
    newModel = custom_utilities.build_new_model(modelUsed[0],classes)
    print("Main.py : Added new Gesture Output Neuron.\n")
    return True

@eel.expose
def takeGestureSamples():
    global gesture_verify
    print("Gesture : Take Sample Pics of Hand Making The new Gesture.")
    import gesture_verify as gfy
    gesture_verify = gfy
    return True

@eel.expose
def testGSamples():
    print("Gesture : Take Test Samples.")
    return True


@eel.expose
def trainGSamples():
    print("Gesture : Take Training Samples. ")
    return True

@eel.expose
def lightTrainNN():
    global trainGen,validGen,newModel,gesture_verify
    print("Gesture : Light Training (Output Layers) of the new Neural Network.")
    #import gesture_verify
    trainGen,validGen = gesture_verify.PreProcessor()
    newModel = gesture_verify.ModelTrainer(newModel,trainGen,validGen,True)

    print("Main.py : New Model Light Training Sucessful.\n")
    return True

@eel.expose
def finalTrainNN():
    print("Gesture : Final Training (All Layers) of the new Neural Network.\n")
    global newModel,custom_utilities

    newModel = custom_utilities.final_trainable_model(newModel)
    newModel = gesture_verify.ModelTrainer(newModel,trainGen,validGen,True)

    print("Main.py : New Model Final Training Sucessful. Model Can now be finalized and Used.\n")
    return True



@eel.expose
def finalizeModel():
    print("Gesture : Finalize the New Neural Network for use.")
    import shutil
    shutil.move("../generatedData/newModel.h5", "models/")
    gv.update('modelToBeUsed','newModel.h5','global')
    print("Main.py : New Model Finalized to be used in Next Start.\n")
    return True

@eel.expose
def uiControlMode(permission):
    print("Gesture : Finalize the New Neural Network for use.")
    if permission == 'ON' :
        gv.update('uiControlMode','True','global')
    elif permission == 'OFF' :
        gv.update('uiControlMode','False','global')

    print("Main.py : UI Control Mode Toggled :: ",permission,'\n')
    return True

print("\nOpening GUI ")
try:
    eel.start('index.html',size=(1000, 1000),position=(300, 50))
except:
    eel.start('index.html',mode='custom-app',size=(1000, 1000),position=(300, 50))
