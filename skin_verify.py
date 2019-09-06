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
flag=0;
names = ['B','G','R','Labels']
print("## Done.")


def DataPrepper():
    print("\t\t\t##Running ","DataPrepper Function##");
    global names
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
        flag = 2
        print("Read DONE.\n")
    except FileNotFoundError:
        print("\n\n\t\tFile(Skin_NonSkin.txt) not Found!!!")
        def choser(data):
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
                flag=1
                print("Read DONE.\n")
            elif choose=='n':
                print("\n\t\tThere is no File to process")
                print("\n\t\tExiting...")
                exit()
            else:
                print("Wrong Choice...TRY AGAIN\n")
                choser(data)
        choser(data)
        print("Chooser DONE")
    finally:
        print("Data Ready to Process")

    row,col=data.shape
    print("Rows and Cols of the data: ",row,col)
    data=shuffle(data)
    #print(data)
    print("Data Shuffled for Future Use")
    if flag==1:
        print("It seems that You have choosen a File to load instead of default file")
        def choser1():
            print("The Q now is :")
            choose=str(input("\n\t\tDo you want to save the current open file for later use?\nEnter [y/n]:   "))
            if choose=='y':
                print("\nSaving Data to CSV file (flowdata.csv)...##",end=" ")
                data.to_csv('flowdata.csv')
                print("TO CSV DONE.")
        choser1()
    return (data)
    print("\n\nData DataPrepper END\n")

def PreProcessor(data1):
    print("\t\t\t##Running ","PreProcessor Function##");
    try:
        print("Importing Packages Required...",end="##")
        #from dataprep import data,np,shuffle
        from sklearn.preprocessing import StandardScaler
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
    return (x_train,x_test,y_train,y_test)


def TrainerTester(x_train,x_test,y_train,y_test):
    print("\t\t\t##Running ","TrainerTester Function##");
    try:
        print("Importing Packages Required...",end="##")
        from sklearn.metrics import accuracy_score
        from matplotlib import pyplot as plt
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

def ModelSaver(clf):
    import pickle
    # now you can save it to a file
    with open('filename.pkl', 'wb') as f:
        pickle.dump(clf, f)
        # and later you can load it
    with open('filename.pkl', 'rb') as f:
        clf = pickle.load(f)
    

if __name__ == "__main__":
    print("This is Main Block of the Program")
    (data1) = DataPrepper()
    (x_train,x_test,y_train,y_test) = PreProcessor(data1)
    TrainerTester(x_train,x_test,y_train,y_test);


print("\n\tEND")
