#####    Module to verify gesture using threshold Image   #####
#The gesture dataset is collected by randomly clicking images
#from threshold window while running capture module

print("\t\t\t##Running ","gesture-verify-test.py##");
print("Setting Global Variables and Packages...",end=' ')
import numpy as np
flags=[];
names = ['1','2','3','4','5','Palm','Fist']
d1=d2=d3=d4=d5=dFist=dPalm=''
Gesture_1=Gesture_2=Gesture_3=Gesture_4=Gesture_5=Gesture_Fist=Gesture_Palm=[]
print("## Done.")

def DataReader(dir_path):
    print("\t\t\t##Running ","DataReader Function##");
    global names,flags
    global d1,d2,d3,d4,d5,DFist,DPalm
    global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm
    print("The Passed Path is : ",dir_path)
    d1 = os.path.join('DATA/1/')
    d2 = os.path.join('DATA/2/')
    d3 = os.path.join('DATA/3/')
    d4 = os.path.join('DATA/4/')
    d5 = os.path.join('DATA/5/')
    dFist = os.path.join('DATA/Fist/')
    dPalm = os.path.join('DATA/Palm/')

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



def DataPrepper():
    print("\t\t\t##Running ","DataPrepper Function##");
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
        DataReader(current_path)
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
                DataReader(foldername)
                flags.append(2)
                print("Read DONE.\n")
            elif choose=='n':
                print("\n\t\tThere are no Files or Folders to process")
                print("\n\t\tExiting...")
                exit()
            else:
                print("Wrong Choice...TRY AGAIN\n")
                choser()
        data=choser()
        print("Chooser DONE")
    finally:
        print("Data LOAD Done")

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pic_index = 2

    next_1 = [os.path.join(d1, fname)for fname in Gesture_1[pic_index-2:pic_index]]
    next_2 = [os.path.join(d2, fname)for fname in Gesture_2[pic_index-2:pic_index]]
    next_3 = [os.path.join(d3, fname)for fname in Gesture_3[pic_index-2:pic_index]]

    for i, img_path in enumerate(next_1+next_2+next_3):
      #print(img_path)
      img = mpimg.imread(img_path)
      plt.imshow(img)
      plt.axis('Off')
      plt.show()

    if 2 in flags:
        print("It seems that You have choosen a File to load instead of default file")
        def choser1():
            print("The Q now is :")
            choose=str(input("\n\t\tDo you want to save the current open file for later use?\nEnter [y/n]:   "))
            if choose=='y':
                print("\nSaving Data to Text file (Skin_NonSkin.txt)...##",end=" ")
                data.to_csv('Skin_NonSkin.txt',sep = "\t", index=False, header=False)
                print("SAVE TO TEXT File DONE.")
        choser1()
    return (data)
    print("\n\nData DataPrepper END\n")

def PreProcessor(dir_path):
    print("\t\t\t##Running ","PreProcessor Function##");
    global names,flags
    global d1,d2,d3,d4,d5,DFist,DPalm
    global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

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
     train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=(150,150),class_mode='categorical')

    #VALIDATION_DIR = "/tmp/rps-test-set/"
    #validation_datagen = ImageDataGenerator(rescale = 1./255)
    #validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(150,150),class_mode='categorical')

    print("\n\t##Data PreProcessing Done. END\n")
    flags.append(3)
    return (train_generator)

def TrainerTester(x_train,x_test,y_train,y_test):
    print("\t\t\t##Running ","TrainerTester Function##");
    global names,flags
    global d1,d2,d3,d4,d5,DFist,DPalm
    global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        impoprt tensorflow as tf
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
    tf.keras.layers.Dense(3, activation='softmax')
    ])


    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit_generator(train_generator, epochs=25, validation_data = validation_generator, verbose = 1)

    model.save("rps.h5")

    flags.append(4)
    return(model5)

def ModelSaver(model):
    print("This Model Saver")
    print("The Flags are : ",flags)
    ok=0
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
            choose=str(input("\n\t\tDo you want to save the current Model for later use?\nEnter [y/n]:   "))
            if choose=='y':
                print("\nSaving Model to Pickle file (Skin_verify_model.pkl)...##",end=" ")
                import pickle
                with open('Skin_verify_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print("SAVE MODEL DONE.")
            elif choose=='n':
                print("!!! Exiting without Saving the model !!!")
        choser1234(model)

def SkinPredict(B,G,R):
    print("\t\t\n  #### Running Skin Color Predictor from 'skin_verify.py' Module ####\n")
    try:
        print("Importing Model from Pickle File...##",end=" ")
        import pickle
        with open('Skin_verify_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model Loaded.\n")
    except FileNotFoundError:
        print("\n\t\t!!! Model not Found !!!")
        def choser1212():
            choose=str(input("\n\t\tOpen File Picker to Choose Model to load?\nEnter [y/n]:   "))
            if choose=='y':
                from tkinter import Tk
                from tkinter.filedialog import askopenfilename
                root=Tk()
                ftypes = [('Pickle FIle',"*.pkl"),('All Types',"*.*")]
                ttl  = "File Picker"
                filename = askopenfilename(filetypes = ftypes,title = ttl)
                root.withdraw()
                print ("This is Chosen FilePath : ",filename)
                print("\tReading Model from Choosen File...##",end=" ")
                import pickle
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
                return model
                print("Model Loaded.\n")
            elif choose=='n':
                print("\n\t\tThere is no Model to Load")
                print("Prediction without model is Impossible.So a model has to be Loaded or Trained.")
                print("To train a new model for 'skin_verify' run 'skin_verify.py' and Train model there.")
                print("Try again to Load a Saved 'Skin_verify_model.pkl' or similar Model")
                choser1212()
            else:
                print("Wrong Choice...TRY AGAIN\n")
                choser1212()
        model=choser12()
    import numpy as np
    zz=np.array([B,G,R])
    zz=zz.reshape(1,-1)
    prediction5=model.predict(zz)
    return prediction5

if __name__ == "__main__":
    print("This is Main Block of the Program")
    (data1) = DataPrepper()
    (x_train,x_test,y_train,y_test) = PreProcessor(data1)
    try:
        print("Importing Model from Pickle File...##",end=" ")
        import pickle
        with open('Skin_verify_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model Loaded.\n")
    except FileNotFoundError:
        print("\n\n\t\tModel not Found!!!")
        def choser12():
            global flag;global x_train,x_test,y_train,y_test;
            choose=str(input("\n\t\tOpen File Picker to Choose Model to load?\nEnter [y/n]:   "))
            if choose=='y':
                from tkinter import Tk
                from tkinter.filedialog import askopenfilename
                root=Tk()
                ftypes = [('Pickle FIle',"*.pkl"),('All Types',"*.*")]
                ttl  = "File Picker"
                filename = askopenfilename(filetypes = ftypes,title = ttl)
                root.withdraw()
                print ("This is Chosen FilePath : ",filename)
                print("\tReading Model from Choosen File...##",end=" ")
                import pickle
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
                flags.append(6)
                return model
                print("Model Loaded.\n")
            elif choose=='n':
                print("\n\t\tThere is no Model to Load")
                print("\n\t\tTraining a new model with the Data loaded  ...")
                model=TrainerTester(x_train,x_test,y_train,y_test);
                print("New Model Trained")
                flags.append(5)
                return(model)
            else:
                print("Wrong Choice...TRY AGAIN\n")
                choser12()
        model=choser12()
    from sklearn.metrics import accuracy_score
    print("\nMaking Predictions from Model")
    prediction5=model.predict(x_test)
    print("Accuracy Score MODEL 4 is : ",accuracy_score(prediction5,y_test))
    ModelSaver(model)
    print("Main Block DOne")

print("\n\tEND of 'skin_verify.py'")
