# Model Loader
print("Importing...",end=' ')
import pickle
from tensorflow.keras.models import load_model
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import skimage
import warnings
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.metrics import precision_score,roc_auc_score,recall_score,f1_score
import os
from tensorflow import cast,float32
print("Import DOne.")

nnmodel=load_model('sources/models/9G_tanh_model_105906112019.h5')
names=['LR','LDA','KNN','DTC','RF','GNB','SVM']
results=[]
warnings.filterwarnings('ignore')

def load_image_files(container_path, dimension=(150, 150)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    print("The categories of gestures are : ",categories,'\nLoading Images...\n')
    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def model_testing(names):
    results=[]
    global X_test,y_test

    for name in names:
          print("\tTesting %s : "%name)
          model_name=("Originals/Models/%s.pkl"%name)
          with open(model_name, 'rb') as f:
            model = pickle.load(f)
          prediction=model.predict(X_test)
          result = accuracy_score(prediction, y_test)
          results.append(result)
          msg = "Accuracy of %s : %f" % (name, result)
          print(msg)
    return results

def trainer(names):
    #C:\Users\nabin\Desktop\Test Files\Originals\Models
    warnings.filterwarnings('ignore')
    model_history=[]
    print("Training Models...")
    for name in names:
        print("   ======== Training %s : ==========="%name)
        model_name=("Originals/Models/%s.pkl"%name)
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        history=model.fit(X_train,y_train)
        print("\t=========== Training Done ===========")
        model_history.append([name,history])
        #trained_models.append([name,model])
        print("\t  Saving Model %s ..."%model_name,end='')
        with open(model_name, 'wb') as f:
          pickle.dump(model, f)
        names.append(name)
        print("Done ##")

def comparator(names):
    print("Received MOdels : ",names)
    results=[]
    global X_test,y_test

    for name in names:
        print("\n\tEvaluating %s : "%name)
        model_name=("Originals/Models/%s.pkl"%name)
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        import time
        start_time = time.time()
        ms = time.time_ns() // 1000000
        prediction=model.predict(X_test)
        end_time = time.time()
        ms2 = time.time_ns() // 1000000
        total_time = end_time - start_time
        ms3=ms2-ms
        total_time=int(total_time)
        x_len=len(X_test)
        print("Time taken to predict %d Images : "%x_len,total_time,' Second(/s) & ',ms3,' Milliseconds')

        precision = precision_score(prediction, y_test,average='weighted')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(prediction, y_test,average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(prediction, y_test,average='weighted')
        print('F1 score: %f' % f1)
        # ROC AUC
        #auc = roc_auc_score(prediction, y_test,average='micro')
        #print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(prediction, y_test)
        print(matrix)

        buffer=[precision,recall,f1,matrix]
        results.append(buffer)
    return results



print("Getting Data...",end=' ')
print(os.getcwd())
os.chdir('C:/Users/nabin/Desktop/Test Files')
image_dataset = load_image_files("C:/Users/nabin/Desktop/Test Files/DATASET/DATA/TrainingData/")
print("And Splitting IT...")
X_train, X_test, y_train, y_test = train_test_split(image_dataset.images,image_dataset.target,test_size=0.1,random_state=1)
print(" ##DONE#")

testX=[]
for array in X_test:
    array = resize(array, (150, 150))
    array = cast(array, float32)
    array = np.expand_dims(array,axis=2)
    array = np.expand_dims(array,axis=0)
    array = np.vstack([array])
    testX.append(array)

for image in testX:
    nnmodel.predict(image)

print(y_test)
#================================================
def nnTester():
    global testX,nnmodel
    import time
    prediction=[]
    total_time=0
    mstotal=0

    for image in testX:
        start_time = time.time()
        ms = time.time_ns() // 1000000
        pred=nnmodel.predict(image)
        ms2 = time.time_ns() // 1000000
        end_time = time.time()

        pred=pred.tolist()
        maxPrediction = max(pred[0])
        index = pred[0].index(maxPrediction)
        prediction.append(index)

        timenow = end_time - start_time
        ms3=ms2-ms
        total_time = total_time + timenow
        mstotal = mstotal+ ms3

    total_time=int(total_time)
    x_len=len(testX)
    print("Time taken to predict %d Images : "%x_len,total_time,' Second(/s) & ',mstotal,' Milliseconds')

    precision = precision_score(prediction, y_test,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(prediction, y_test,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(prediction, y_test,average='weighted')
    print('F1 score: %f' % f1)
    # ROC AUC
    #auc = roc_auc_score(prediction, y_test,average='micro')
    #print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(prediction, y_test)
    print(matrix)
nnTester()
#================================================
print("Training MOdels...")
#model_history=trainer(names)
print("=============Training DOne now TESTING...==============")
#results=model_testing(names)
#ComparisonRes=comparator(names)
i=0
Ascores=[]
for name in names:
    Ascores.append(ComparisonRes[i][0])
    #plt.plot(ComparisonRes[i][0],color='red')
    i+=1

print(Ascores)
plt.plot(Ascores)
