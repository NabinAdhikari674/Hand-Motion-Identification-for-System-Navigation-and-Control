#from tensorflow.keras.models import load_model
import custom_utilities

bmodel=('models/best_model_083529102019.h5')
class_list=["1", "2", "3","4","5","Fist","Palm",'none','ok']
i=0

custom_utilities.build_new_model(bmodel,class_list)
    
