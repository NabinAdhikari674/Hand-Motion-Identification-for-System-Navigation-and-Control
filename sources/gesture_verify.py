#####    Module to verify gesture using threshold Image   #####
#The gesture dataset is collected by randomly clicking images
#from threshold window while running capture module

print("\n\t\t\t##Running ","gesture-verify-test.py##");
print("Setting Global Variables and Packages...",end=' ')
import numpy as np
import custom_utilities
import global_var_tunnel as gv
flags=[];
names = ['1','2','3','4','5','Palm','Fist']
d1=d2=d3=d4=d5=dFist=dPalm=''
Gesture_1=Gesture_2=Gesture_3=Gesture_4=Gesture_5=Gesture_Fist=Gesture_Palm=[]
print("## Done.")

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
        foldername=custom_utilities.FileFolderPicker([('Folder')],title='FolderPicker',mode='Folder')
        dir_path=DataReader(foldername)
    except Exception as exp:
        print("\t!! Error !! : \n\t\t",exp)
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

def PreProcessor(dir_path='../generatedData/gestureData/'):
    print("\t\t\t##Running ","PreProcessor Function##");
    global flags
    #global d1,d2,d3,d4,d5,DFist,DPalm
    #global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        #import tensorflow as tf
        #import keras_preprocessing
        #from keras_preprocessing import image
        from keras_preprocessing.image import ImageDataGenerator
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    TRAINING_DIR = dir_path+'TrainingData'
    print("Training Directory is :",TRAINING_DIR)
    training_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=(150,150),color_mode='grayscale',class_mode='categorical')

    VALIDATION_DIR=dir_path+'ValidationData'
    validation_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip=True)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(150,150),color_mode='grayscale',class_mode='categorical')

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

def DataSplitter(train_generator,validation_generator):
    print("=========== Data Preparation going On ... ============\n")

    #xt, yt = zip(*(train_generator[i] for i in range(len(train_generator))))
    #x_train, y_train = np.vstack(xt), np.vstack(yt)
    for d, l in validation_generator:
        data.append(d)
        labels.append(l)
        i += 1
        if i == max_iter:
            break
    print("Training Data Length : ",len(x_train),x_train.shape)


    #xv,yv = zip(*(validation_generator[i] for i in range(len(validation_generator))))
    #x_val, y_val = np.vstack(xv), np.vstack(yv)
    print("Validation Data Length : ",len(x_val),x_val.shape)
    print(x_val)

    print("============ Data Preparation DONE ... =============\n")
    return x_train,y_train,x_val,y_val

def SVMModel():

    #from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    model=DecisionTreeClassifier()


    '''import tensorflow as tf

    model = tf.keras.models.Sequential([

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

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7,kernel_regularizer=tf.keras.regularizers.l2(0.01)), #kernel_regularizer=regularizers.l2(0.01)
        tf.keras.layers.Dense(7, activation='softmax')

                                      ])


    model.compile(loss='squared_hinge',optimizer='adadelta',metrics=['accuracy'])
    '''

    print("\n\t##SVM Model Built and Compiled(Ready for Training). END\n")
    flags.append(8)
    return(model)

def ModelTrainer(*args):
    print("\t\t\t##Running ","ModelTrainerTester Function##");
    global flags
    model=train_generator=validation_generator=generator=x_train=y_train=None
    for argument in args:
        try:
            model=args[0]
            train_generator=args[1]
            validation_generator=args[2]
            generator=args[3]
            x_train=args[4]
            y_train=args[5]
        except:
            pass
            #print("Some Arguments were Missing.But we can Move on...")

    #global d1,d2,d3,d4,d5,DFist,DPalm
    #global Gesture_1,Gesture_2,Gesture_3,Gesture_4,Gesture_5,Gesture_Fist,Gesture_Palm

    try:
        print("Importing Packages Required...",end="##")
        import tensorflow as tf
        import logging
        logging.getLogger('tensorflow').disabled = True
        #from matplotlib import pyplot as plt
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    trained=0
    choose=str(input("\n\t\tStart the training?\nEnter [y/n]:   "))
    if choose=='y':
        print("!! Sit Tight !! Training takes a Whole lot of Time.\n")
        if generator==True:
            history = model.fit_generator(train_generator,validation_data = validation_generator, epochs=1, verbose = 1)# ,validation_data = validation_generator
        elif generator==False:
            history = model.fit(x_train,y_train)
        trained = 1
    if choose=='n':
        choose=str(input("\n\t\t!!! Halt the Training for Sure?\nEnter [y/n]:   "))
        if choose=='y':
            pass
        if choose=='n':
            print("Buckle Up !! You are up for a Long Ride !!\n")
            if generator==True:
                history = model.fit_generator(train_generator,validation_data = validation_generator, epochs=1, verbose = 1)# ,validation_data = validation_generator
            elif generator==False:
                history = model.fit(x_train,y_train)
            trained = 1

    if trained == 1:
        choose = str(input("\n\t\tSave Model to the Disk?\nEnter [y/n]:   "))
        if choose =='y':
            model.save("../generatedData/newModel.h5")
            print("Model Saved !! ('newModel.h5')")
        if choose =='n':
            choose=str(input("\n\t\tYou Sure to not save the Hard-earned Trained Model?\nEnter [y/n]:   "))
            if choose =='y':
                pass
            if choose =='n':
                model.save("../generatedData/newModel.h5")
                print("Knew It !! Model Saved as ('newModel.h5')")
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

def ModelLoader(loc='models/9G_tanh_model_105906112019.h5'):
    print("\t\t\n  #### Running ModelLoader Function ####\n")
    try:
        print("Importing Packages Required...",end="##")
        from tensorflow.keras.models import load_model
        #from tensorflow.keras.models import model_from_json
        print("...Import Sucessful")
    except Exception as exp:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\tError:\n\t\t",exp)

    print("Importing Trained Model using  \"h5\" File...##",end=" ")
    try:
        loc='models/9G_tanh_model_105906112019.h5'
        model=load_model(loc)
        gv.update('modelUsed',loc)
    except FileNotFoundError:
        ftypes = [('Hierarchical Data Format(HDF)',"*.h5")]
        tt1  = "Hierarchical Data Format (HDFv5) Picker"
        filename1=custom_utilities.FileFolderPicker(ftypes,tt1)
        model=load_model(filename1)
        gv.update(filename1)
    except Exception as ex:
        print("!! Error !! : \n\t",ex)

    print("======== Model Loaded from Disk. ========")

    return model

def GesturePredictor(model,array,mode='gray'):
    if mode=='gray':
        array = resize(array, (150, 150))
        array = cast(array, float32)
        array = np.expand_dims(array,axis=2)
        array = np.expand_dims(array,axis=0)
        array = np.vstack([array])
        prediction = model.predict(array)
        return prediction

    elif mode=='rgb':
        array = np.expand_dims(array,axis=0)
        array = np.vstack([array])
        prediction = model.predict(array)
        return prediction

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
            #model=NeuralNetBuilder()
            model=SVMModel()
            ModelSaver(model)
            x_train,y_train,x_val,y_val=DataSplitter(train_generator,validation_generator)
            ModelTrainer(model,train_generator,validation_generator,False,x_train,y_train)
        if choose=='n':
            model=ModelLoader()
    from keras.preprocessing import image
    import cv2
    while True:
        ftypes = [('Jpg File',"*.jpg")]
        tt1  = "Picture Picker"
        path=custom_utilities.FileFolderPicker(ftypes,tt1)
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
elif __name__ == "gesture_verify":
    print("Import Found======================")
    try:
        from cv2 import resize
        from tensorflow import cast,float32
    except Exception as exp:
        print(" !! Error !! : \n\t",exp)
else:
    print("Dunno.But Import Found as : **",__name__)

print("\n\tEND of 'gesture_verify.py'" )
