#####    Module to verify skin color in BGR   #####


#The skin dataset is collected by randomly sampling B,G,R
#values from face images of various age groups (young, middle, and old),
#race groups (white, black, and asian), and genders
#obtained from FERET database and PAL database.

#Total learning sample size is 245057;
#out of which 50859 (Labeled as 1) is the skin samples and
#194198 (labeled as 2) is non-skin samples

#This dataset is of the dimension 245057 * 4
#where first three columns are B,G,R (x1,x2, and x3 features) values and
#fourth column is of the class labels (decision variable y).


print("\t\t\t##Running ","skin-verify-test.py##");
print("Setting Global Variables and Packages...",end=' ')
import numpy as np
flags=[];
names = ['B','G','R','Labels']
print("## Done.")

def DataPrepper():
    print("\t\t\t##Running ","DataPrepper Function##");
    global names;global flags
    try:
        print("Importing Packages Required...",end="##")
        import pandas as pd
        from sklearn.utils import shuffle
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    try:
        print("Importing Data from File...##",end=" ")
        #data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt', delimiter = "\t", header=None)
        data = pd.read_csv('Skin_NonSkin.txt', delimiter = "\t",header=None)
        data.columns = names
        flags.append(1)
        print("Read DONE.\n")
    except FileNotFoundError:
        print("\n\n\t\tFile(Skin_NonSkin.txt) not Found!!!")
        def choser():
            global flag
            choose=str(input("\n\t\tOpen File Picker to Choose file to open?\nEnter [y/n]:   "))
            if choose=='y':
                from tkinter import Tk
                from tkinter.filedialog import askopenfilename
                root=Tk()
                ftypes = [('Text FIle',"*.txt"),('CSV File',"*.csv"),('Excel File',"*.xlsx"),('All Types',"*.*")]
                ttl  = "FilePicker"
                filename = askopenfilename(filetypes = ftypes,title = ttl)
                root.withdraw()
                print ("This is Chosen FilePath : ",filename)
                print("\tReading Data from Choosen File...##",end=" ")
                data=pd.read_csv(filename,delimiter = "\t", header=None)
                data.columns = ["B", "G", "R", "Label"]
                flags.append(2)
                return(data)
                print("Read DONE.\n")
            elif choose=='n':
                print("\n\t\tThere is no File to process")
                print("\n\t\tExiting...")
                exit()
            else:
                print("Wrong Choice...TRY AGAIN\n")
                choser()
        data=choser()
        print("Chooser DONE")
    finally:
        print("Data Processor Done")

    row,col=data.shape
    print("Rows and Cols of the data: ",row,col)
    data=shuffle(data)
    #print(data)
    print("Data Shuffled for Future Use")
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

def PreProcessor(data1):
    global flags;
    print("\t\t\t##Running ","PreProcessor Function##");
    try:
        print("Importing Packages Required...",end="##")
        #from dataprep import data,np,shuffle
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")
    inputs=data1.iloc[:,:3]
    targets=data1.iloc[:,3:4]
    #print(inputs)
    #print(targets)
    percent=int(input("\n\t\tEnter the TEST size(IN PERCENTAGE) : "))
    from sklearn.model_selection import train_test_split
    print("\n\tRandomly Choosing Training and Test Data...\n")
    x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=percent,random_state=1)
    print("\t\tThe Training and Test Data ARE Split")
    print("\n\t##Data PreProcessing Done. END\n")
    flags.append(3)
    return (x_train,x_test,y_train,y_test)

def TrainerTester(x_train,x_test,y_train,y_test):
    print("\t\t\t##Running ","TrainerTester Function##");
    global flags;
    try:
        print("Importing Packages Required...",end="##")
        from sklearn.metrics import accuracy_score
        #from matplotlib import pyplot as plt
        print("...Import Sucessful")
    except:
        print("\n\t##Error in IMPORT...Check if all packages are properly Installed##\n\t")

    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test)

    #Training
    print("\n\tRunning KNN Classifier . . .\n")
    from sklearn.neighbors import KNeighborsClassifier
    model5=KNeighborsClassifier()
    model5.fit(x_train,y_train)
    prediction5=model5.predict(x_test)
    #print(model5.score(Xtest,Ytest)*100)
    print("Accuracy Score MODEL 4 is : ",accuracy_score(prediction5,y_test))
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
