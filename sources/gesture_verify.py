#####    Module to verify gesture using threshold Image   #####
#The gesture dataset is collected by randomly clicking images
#from threshold window while running capture module

print("\n\t\t\t##Running ","gesture-verify-test.py##");
print("Setting Global Variables and Packages...",end=' ')
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
flags=[];
names = ['1','2','3','4','5','Palm','Fist']
d1=d2=d3=d4=d5=dFist=dPalm=''
Gesture_1=Gesture_2=Gesture_3=Gesture_4=Gesture_5=Gesture_Fist=Gesture_Palm=[]
print("## Done.")

def NotFoundSoln(ftypes,title):
    print("\n\t\t!!! Model not Found !!!")
    choose=str(input("\n\t\tOpen File Picker to Choose Model to load?\nEnter [y/n]:   "))
    if choose=='y':
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        root=Tk()
        filename = askopenfilename(filetypes = ftypes,title = title)
        root.withdraw()
        print ("This is Chosen FilePath : ",filename)
        return filename
    elif choose=='n':
        print("\n\t\tThere is no Model to Load")
        print("Prediction without model is Impossible.So a model has to be Loaded or Trained.")
        print("To train a new model for 'skin_verify' run 'skin_verify.py' and Train model there.")
        print("Try again to Load a Saved 'Skin_verify_model.pkl' or similar Model")
        NotFoundSoln(ftypes,title)
    else:
        print("Wrong Choice...TRY AGAIN\n")
        NotFoundSoln(ftypes,title)

def DataReader(dir_path):
    print("\n\t\t\t##Running ","DataReader Function##");
    import os
    global names,flags
    global d1,d2,d3,d4,d5,DFist,DPalm
    global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm
    print("The Passed Path is : ",dir_path)
    d1 = os.path.join('DATA/TrainingData/1/')
    d2 = os.path.join('DATA/TrainingData/2/')
    d3 = os.path.join('DATA/TrainingData/3/')
    d4 = os.path.join('DATA/TrainingData/4/')
    d5 = os.path.join('DATA/TrainingData/5/')
    dFist = os.path.join('DATA/TrainingData/Fist/')
    dPalm = os.path.join('DATA/TrainingData/Palm/')

    print('Total Training Gesture (1) images:', len(os.listdir(d1)))
    print('Total Training Gesture (2) images:', len(os.listdir(d2)))
    print('Total Training Gesture (3) images:', len(os.listdir(d3)))
    print('Total Training Gesture (4) images:', len(os.listdir(d4)))
    print('Total Training Gesture (5) images:', len(os.listdir(d5)))
    print('Total Training Gesture (Fist) images:', len(os.listdir(dFist)))
    print('Total Training Gesture (Palm) images:', len(os.listdir(dPalm)))

    Gesture_1 = os.listdir(d1);   #print(Gesture_1[:10])
    Gesture_2 = os.listdir(d2);   #print(Gesture_2[:10])
    Gesture_3 = os.listdir(d3);   #print(Gesture_3[:10])
    Gesture_4 = os.listdir(d4);   #print(Gesture_4[:10])
    Gesture_5 = os.listdir(d5);   #print(Gesture_5[:10])
    Gesture_Fist = os.listdir(dFist);   #print(Gesture_Fist[:10])
    Gesture_Palm = os.listdir(dPalm);   #print(Gesture_Palm[:10])
    dir_path=os.path.join('DATA/TrainingData/')
    print("===============================\nThe Returned Path is : ",dir_path)

    return dir_path

def DataPrepper():
    print("\n\t\t\t##Running ","DataPrepper Function##");
    global names,flags
    global d1,d2,d3,d4,d5,DFist,DPalm
    global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        import os
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    try:
        print("Importing Data from Folders...##",end=" ")
        current_path=os.getcwd()
        #print("The Current Path is : ",current_path)
        dir_path=DataReader(current_path)
        flags.append(1)
        print("Read DONE.\n")
    except FileNotFoundError:
        print("\n\n\t\tImage Folders Not Found !!!")
        def choser():
            global flag
            choose=str(input("\n\t\tOpen File Picker to Choose Folder to open?\nEnter [y/n]:   "))
            if choose=='y':
                from tkinter import Tk
                from tkinter.filedialog import askdirectory
                root=Tk()
                root.withdraw()
                foldername = askdirectory(initialdir=os.getcwd,title="Folder Chooser" )
                #print ("This is Chosen Folder and Path : ",foldername)
                print("\tReading Data from Choosen Folder...##",end=" ")
                dir_path=DataReader(foldername)
                flags.append(2)
                return dir_path
                print("Read DONE.\n")
            elif choose=='n':
                print("\n\t\tThere are no Files or Folders to process")
                print("\n\t\tExiting...")
                exit()
            else:
                print("Wrong Choice...TRY AGAIN\n")
                choser()
        dir_path=choser()
        print("Chooser DONE")
    finally:
        print("Data LOAD Done")

    #import matplotlib.pyplot as plt
    #import matplotlib.image as mpimg

    #pic_index = 2

    #next_1 = [os.path.join(d1, fname)for fname in Gesture_1[pic_index-2:pic_index]]
    #next_2 = [os.path.join(d2, fname)for fname in Gesture_2[pic_index-2:pic_index]]
    #next_3 = [os.path.join(d3, fname)for fname in Gesture_3[pic_index-2:pic_index]]

    #for i, img_path in enumerate(next_1+next_2+next_3):
      #print(img_path)
    #  img = mpimg.imread(img_path)
    #  plt.imshow(img)
    #  plt.axis('Off')
    # plt.show()

    return dir_path
    print("\n\n DataPrepper END\n")

def PreProcessor(dir_path):
    print("\t\t\t##Running ","PreProcessor Function##");
    global flags
    #global d1,d2,d3,d4,d5,DFist,DPalm
    #global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        import tensorflow as tf
        import keras_preprocessing
        from keras_preprocessing import image
        from keras_preprocessing.image import ImageDataGenerator
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    TRAINING_DIR = dir_path
    print("Training Directory is :",TRAINING_DIR)
    training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=(150,150),color_mode='rgb',class_mode='categorical')

    VALIDATION_DIR='C:/Users/nabin/Desktop/Test Files/DATA/ValidationData/'
    validation_datagen = ImageDataGenerator(rescale = 1./255)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(150,150),color_mode='rgb',class_mode='categorical')
    
    print("\n\t##Data PreProcessing Done. END\n")
    flags.append(3)
    return (train_generator,validation_generator)

def NeuralNetBuilder():
    print("\t\t\t##Running ","NeuralNetBuilder Function##");
    global flags
    #global d1,d2,d3,d4,d5,DFist,DPalm
    #global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        import tensorflow as tf
        #from matplotlib import pyplot as plt
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    model = tf.keras.models.Sequential([
                                        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
                                        # This is the first convolution
                                        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
                                        tf.keras.layers.MaxPooling2D(2, 2),
                                        # The second convolution
                                        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        # The third convolution
                                        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        # The fourth convolution
                                        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        # Flatten the results to feed into a DNN
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dropout(0.5),
                                        # 512 neuron hidden layer
                                        tf.keras.layers.Dense(512, activation='relu'),
                                        tf.keras.layers.Dense(7, activation='softmax')
                                        ])

    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("\n\t##Neural Net Model Built and Compiled(Ready for Training). END\n")
    flags.append(7)
    return(model)

def ModelTrainer(model,train_generator,validation_generator):
    print("\t\t\t##Running ","ModelTrainerTester Function##");
    global flags
    #global d1,d2,d3,d4,d5,DFist,DPalm
    #global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        import tensorflow as tf
        #from matplotlib import pyplot as plt
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    trained=0
    choose=str(input("\n\t\tStart the training?\nEnter [y/n]:   "))
    if choose=='y':
        print("Sit Tight !! Training takes a whole lot of Time.\n")
        history = model.fit_generator(train_generator,validation_data = validation_generator, epochs=100, verbose = 1)# ,validation_data = validation_generator
        trained = 1
    if choose=='n':
        choose=str(input("\n\t\tHalt the Training for Sure?\nEnter [y/n]:   "))
        if choose=='y':
            pass
        if choose=='n':
            print("Buckle Up !! You are up for a Long Ride !!\n")
            history = model.fit_generator(train_generator, epochs=100, verbose = 1)  # ,validation_data = validation_generator
            trained = 1

    if trained == 1:
        choose=str(input("\n\t\tSave Model to the Disk?\nEnter [y/n]:   "))
        if choose =='y':
            model.save("gesture_neural_weights1.h5")
            print("Model Saved !! ('gesture_neural_weights1.h5')")
        if choose =='n':
            choose=str(input("\n\t\tYou Sure to not save the Hard-earned Trained Model?\nEnter [y/n]:   "))
            if choose =='y':
                pass
            if choose =='n':
                model.save("gesture_neural_weights1.h5")
                print("Knew It !! Model Saved as ('gesture_neural_weights1.h5')")
    flags.append(4)
    return(model)

def ModelSaver(model):
    print("This Model Saver")
    print("The Flags are : ",flags)
    ok=1
    if 5 in flags:
        print("It seems that You have Trained a new Model without Loading one")
        ok=1
    if 4 in flags:
        if 5 not in flags:
            print("It seems that You have Trained a old Model loaded through a Pickle file")
            ok=1
    if 6 in flags:
        print("Model was loaded so NO Need TO SAVE")
    if ok==1:
        print("Choser Ahead")
        def choser1234(model):
            print("The Q now is :")
            choose=str(input("\tDo you want to save the Neural Model( Frame Only ) for later use?\nEnter [y/n]:   "))
            if choose=='y':
                print("==================================================")
                print("\nSaving Model to Jason File (gesture_neural_model.json)...##",end=" ")
                import time
                file="gesture_neural_model1.json"
                print("Creating Model At: ", file)
                start_time = time.time()
                #model = NeuralNetBuilder()
                json_model = model.to_json()
                with open(file, "w") as json_file:
                    json_file.write(json_model)
                end_time = time.time()
                total_time = end_time - start_time
                total_time=int(total_time)
                print("Model Saved in : ", total_time, " seconds")
                print("SAVE MODEL DONE.")
                print("==================================================")
            elif choose=='n':
                print("!!! Exiting without Saving the model !!!")
        choser1234(model)

def ModelLoader():
    print("\t\t\n  #### Running ModelLoader Function ####\n")
    try:
        print("Importing Packages Required...",end="##")
        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import model_from_json
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    print("Importing Trained Model using \"json\" and \"h5\" Files...##",end=" ")
    try:
        json_file = open('gesture_neural_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    except FileNotFoundError:
        ftypes = [('Json FIle',"*.json")]
        tt1  = "Json File Picker"
        filename1=NotFoundSoln(ftypes,tt1)
        json_file = open(filename1, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    try:
        model.load_weights("gesture_neural_weights.h5")
    except FileNotFoundError:
        ftypes = [('Hierarchical Data Format(HDF)',"*.h5")]
        tt1 = "Hierarchical Data Format (HDFv5) Picker"
        filename2 = NotFoundSoln(ftypes,tt1)
        model.load_weights(filename2)
    print("\n $$$$$$$$$$ Model Loaded from disk. $$$$$$$$$$$")
    return model

def GesturePredictor(model,array):
    #img=image.array_to_img(array)
    #img = image.load_img(images, target_size=(150, 150))
    #x = image.img_to_array(img)
    #x = img_to_array(array)
    #resize=cv2.resize(x,(150,150))
    #resize=img_to_array(x)
    #img=resize[np.newaxis,...]
    img = np.expand_dims(array,axis=0)
    img = np.vstack([img])
    prediction5=model.predict(img)

    return prediction5

if __name__ == "__main__":
    print("===== ## This is Main Block of the Program of 'gesture_verify.py' ## =====")
    choose=str(input("\n\t\tLoad Model From the Disk?\nEnter [y/n]:   "))
    if choose=='y':
        model=ModelLoader()
    if choose=='n':
        choose=str(input("\n\t\tYou Sure? Are you upto the heavy Task uphead?\nEnter [y/n]:   "))
        if choose=='y':
            dir_path=DataPrepper()
            train_generator,validation_generator=PreProcessor(dir_path)
            #print(train_generator)
            labels = (train_generator.class_indices)
            labels = dict((v,k) for k,v in labels.items())
            print("The Training Labels are ================================\n",labels,'\n=================================================')
            model=NeuralNetBuilder()
            ModelSaver(model)
            ModelTrainer(model,train_generator,validation_generator)
        if choose=='n':
            model=ModelLoader()
    from keras.preprocessing import image
    import cv2
    while True:
        ftypes = [('Jpg File',"*.jpg")]
        tt1  = "Picture Picker"
        path=NotFoundSoln(ftypes,tt1)
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes)
        keypress = cv2.waitKey(2) & 0xFF
        if keypress == ord("q"):
            print("Exit Key Pressed")
            break
        
    print("\nPro Tip : You can Make Gesture Predictions using this Module")
    print("\n====== ## Main Block DOne ##======")

print("\n\tEND of 'gesture_verify.py'")
