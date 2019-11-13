print("============ Custom Utilities Built by NabinAdhikari ==========")

def FileFolderPicker(ftypes=[('All FIles','*')],title="File/Folder Picker",mode='File'):
    if 'Model' in title:
        print("\t\t!!! Model not Found !!!")
    elif 'Json' in title:
        print("\t\t!!! Json File not Found !!!")
    elif 'Folder' in title:
        print("\t\t!!! Folder not Found !!!")
    choose=str(input("\n\t\tOpen File/Folder Picker to Choose Model to load?\n\tEnter [y/n]:   "))
    if choose=='y':
        if mode=='File':
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename
            root=Tk()
            root.withdraw()
            filename = askopenfilename(filetypes = ftypes,title = title)
            print ("This is Chosen FilePath : ",filename)
            return filename
        if mode=='Folder':
            from tkinter import Tk
            from tkinter.filedialog import askdirectory
            root=Tk()
            root.withdraw()
            foldername = askdirectory(initialdir=os.getcwd,title="Folder Chooser" )
            print("\tThis is Choosen FolderPath : ",foldername)
            return foldername
    elif choose=='n':
        print("\t\t!!! Cannot Proceed Without Choosing File/Folder Required.You have to Choose !!!")
        FileFolderPicker(ftypes,title,mode)
    else:
        print("\t\t!!! Wrong Choice...TRY AGAIN !!!")
        FileFolderPicker(ftypes,title,mode)
#FileFolderPicker([('Model','*.h5py')],"Model Picker",'File')

def build_new_model(base_model,class_list=["1", "2", "3","4","5","Fist","Palm"]):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import plot_model


    #class_list = ["1", "2", "3","4","5","Fist","Palm","None"]
    base_model = load_model(base_model)
    plot_model(base_model, to_file='../generaredData/base_model.png')
    num_classes=len(class_list)

    if num_classes==9:
        print(" !! No New Gestures are Added OR, New Gesture Not Found !! \n    New Model cannot be Made. Same Model will be Used.")
        return base_model

    #is still pointing to the old output of the model.
    #Pass main_model.layers[-1]
    #base_model.pop()
    #base_model.summary()
    final_model=Sequential()
    for layer in base_model.layers[:-1]:
        layer.trainable = False
        final_model.add(layer)

    output = Dense(num_classes, activation='sigmoid',name='final_output')
    final_model.add(output)
    #final_model = Dense.from_config(output)(base_model.layers[-1])
    plot_model(final_model, to_file='../generatedData/new_final_model.png')
    #final_model.summary()

    final_model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("\t\t== New Model Built and Compiled. Ready for Training ==")

    return final_model

def final_trainable_model(base_model):
    for layer in base_model.layers:
        layer.trainable = True
    base_model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("\t\t== New Model Made Fully Trainable and Compiled. Ready for Training ==")
    return base_model
