# Global Variables and Its CommonPlace
import pandas as pd
gesture_predict_mode = False
#bg=None
dX = 0
dY = 0
direction=""

def update(global_var,value,tunnel='global'):
    #print("Loading..")
    if tunnel=='global':
        try:
            gdata = pd.read_csv('dataBase/global_var_tunnel.csv')
        except FileNotFoundError:
            print("=== Global Variable Tunnel Not Found !! ===")

        #print("Updating...")
        index=gdata.index[gdata['Variables'] == global_var].tolist()
        gdata.loc[index,'Values'] = value
        #print(global_var,'::',value)
        gdata.to_csv('dataBase/global_var_tunnel.csv',index=False)
        #print("== Updated ==")
    elif tunnel=='buffer':
        bufferData=value
        bufferInfo = pd.DataFrame(list(bufferData.items()), columns=['Gestures', 'Actions'])
        bufferInfo = bufferInfo.sort_values(['Gestures'])
        bufferInfo.to_csv('dataBase/bufferInfo.csv',index=False)
        print("== Buffer Updated ==")

def get_global_values(global_var='',tunnel='global'):
    if tunnel=='global':
        try:
            gdata = pd.read_csv('dataBase/global_var_tunnel.csv')
        except FileNotFoundError:
            print("=== Global Variable Tunnel Not Found !! ===")
        except Exception as exp:
            print("=== Error in global_var_tunnel.py ===\n\t",exp)
        index=gdata.index[gdata['Variables'] == global_var].tolist()
        value=gdata.loc[index,'Values'].tolist()
        return value
    elif tunnel=='information':
        try:
            idata = pd.read_csv('dataBase/information_center.csv')
        except FileNotFoundError:
            print("=== Information Center Not Found !! ===")
        #index = gdata.index[gdata['Gesture'] == global_var].tolist()
        gestures = idata['Gesture'].tolist()
        actions = idata['Action'].tolist()

        infoDict = dict(zip(gestures, actions))
        gestures = str(gestures)
        actions = str(actions)
        #infodict = str(infodict)
        return gestures,actions,infoDict
    elif tunnel=='buffer':
        try:
            bufferInfo = pd.read_csv('dataBase/bufferInfo.csv')
        except FileNotFoundError:
            print("=== BufferInformation Not Found !! ===")
        except Exception as exp:
            print("=== Error in global_var_tunnel.py ===\n\t",exp)
        gestures = bufferInfo['Gestures'].tolist()
        actions = bufferInfo['Actions'].tolist()
        return gestures,actions


#update('dY',1067)
#value=get_global_value('dY')
#print(value)

if __name__ == '__main__':
    while True:
        print(get_global_values('dX'),get_global_values('dY'))
        #print(get_global_values('direction'))
