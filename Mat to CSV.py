print("Importing Packages...",end='##')
import scipy.io
import numpy as np
import pandas as pd
print(" ## DONE\n")

looptime=int(input("File No. Input : "))
loop=1
buffer=pd.DataFrame()
elements=[]

while loop<=looptime:
    file="labels/Train ("+str(loop)+").mat"
    print(file)
    loop+=1
    print("== Reading Data ==",end='')
    data = scipy.io.loadmat(file)
    print("== DONE ==")
    values=data['boxes']
    print("Shape is ",values.shape)
    for i in values:
        for j in i:
            for k in j:
                for l in k:
                    for m in l:
                        for n in m:
                            for o in n:
                                if type(o) is np.float64:
                                    #print(o,type(o))
                                    elements.append(o)
    numEl = len(elements)
    newRow = pd.DataFrame(np.array(elements).reshape(1,numEl))
    buffer = buffer.append(newRow,ignore_index=True)
    elements=[]
#print("\nThe Buffer is : ",buffer)

print("Saving Buffer in csv file ...",end=' ##')
buffer.to_csv('buffer.csv')
print("TO CSV DONE.")






'''
print("0 os ; ",lon[0][0][0][0][0][0][0])
print("1 is : ",lon[0][0][0][0][0][0][1])
print("2 is : ",lon[0][0][0][0][1][0][0])
print("3 is : ",lon[0][0][0][0][1][0][1])
print("4 is : ",lon[0][0][0][0][2][0][0])
print("5 is : ",lon[0][0][0][0][2][0][1])
print("6 is : ",lon[0][0][0][0][3][0][0])
print("7 is : ",lon[0][0][0][0][3][0][1])
print("\nNext one:")
print("0 os ; ",lon[0][1][0][0][0][0][0])
print("1 is : ",lon[0][1][0][0][0][0][1])
print("2 is : ",lon[0][1][0][0][1][0][0])
print("3 is : ",lon[0][1][0][0][1][0][1])
print("4 is : ",lon[0][1][0][0][2][0][0])
print("5 is : ",lon[0][1][0][0][2][0][1])
print("6 is : ",lon[0][1][0][0][3][0][0])
print("7 is : ",lon[0][1][0][0][3][0][1])'''

