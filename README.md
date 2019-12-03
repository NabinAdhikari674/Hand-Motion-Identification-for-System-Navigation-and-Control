# Hand-Gesture-Recognition or Hand Motion Identification for System Navigation and Control

A real time video feed is analyzed for hand recognition and then trained for gesture recognition made through hands. The live video is 
fed through camera from devices that are connected to the model such as laptop and phones.The model uses machine learning techniques to 
train and predict the data i.e hand gestures. Python shall be the dominant programming language used along with Java and web languages 
for web deployment of the model. The model can also be deployed in Android phones. The output from the model can be used to control and 
direct various interfaces like mouse in computers, joysticks in gaming and even virtual simulation of the hand.
# Detailed Description :
## INTRODUCTION: 
The project is titled “Hand Motion Identification for System Navigation and Control”. In this project, a real time video feed is 
analyzed for hand recognition and then trained for gesture and motion recognition made through hands. The live video is fed through 
camera from devices that are connected to the devices such as laptop and phones. The model uses machine learning techniques to train and 
predict the data i.e. hand gestures. Python shall be the dominant programming language used along with Java and web languages for web 
deployment of the model. The model can also be deployed in Android phones. The output from the model can be used to control and direct 
various interfaces like mouse in computers and joysticks in gaming.

## OBJECTIVE: 
We are going to experiment if various hand gestures and motions can be utilized to control various interfaces on a system. The 
effectiveness of this method to control various systems will be analyzed. The comparison between various methods of hand recognition, 
with result being best method suitable for the task in hand will be

## STEPS: 
The initial data is the video feed from camera captures in certain frame rates suitable. The images fetched from video will go through a 
series of pre-processing such as grey scale image conversion, image correction and binary image conversion. This data will be used to 
recognize hand, then motion and gesture detection in the image using various libraries such as OpenCV and Sklearn. This identification 
data can now be used to control interfaces in various systems by associating various gestures or motion of a hand to actions in various 
System Interface.

## LANGUAGES And LIBRARIES USED: 
The programming language to be dominantly used is “Python”, probably version 3. Various libraries such as OpenCV, SkLearn, NumPy etc. 
will be used with python. The language used to control interfaces in various systems will be “Java”. If we decide to deploy the model 
using web technologies, various Web Languages can also be used.


### The Model files(Json with NN Frame and H5 with Weights) are uploaded in G-Drive and the links will be shared later.

# User-Manual : 
This product was made as an experiment and was not intended to be used for a daily use case an HCI tool. The product is preferred to be 
used in a controlled environment where the background is not usually changing or is static. The main application shall be distributed 
either as a Folder containing all the source codes or, as an executable file. The source codes have three main folders in the main root 
folder, namely, ‘generatedData’, ‘webEngine’ and ‘sources’. The ‘generatedData’ folder is utilized to save all the generated data during 
the workflow of the product. The ‘webEngine’ folder consists of the web pages or files required for the GUI of the application. The 
‘sources’ folder consists of all the python source codes and database used in the backend of the application. In the case of source 
codes, we have to move to the ‘sources’ sub-folder inside the main root folder. Inside the ‘source’ folder there is a python file named 
‘main.py’ which is the main file to be run. In this case, the user should have all the dependencies and packages required for the 
project (probable provided in a ‘requirements.txt’ file). We must run the ‘main.py’ file either by using command line or any python 
interpreter or IDE.  After ‘main.py’ is run, the loading starts. While on a project distributed as an executable file, we just have to 
open the executable file and the project starts loading. The GUI page of the application opens as a web-app if the user has Google 
Chrome browser installed or as a webpage in a default browser on both cases.

  The initial load of the GUI takes a little bit of time (on average 12 seconds) but after that (after initial load), the GUI loads very 
fast. After the application starts, the main console of the project also starts, and the GUI is completely loaded before performing 
any other tasks. After its complete loading, it looks as following :  

![alt text](https://raw.githubusercontent.com/NabinAdhikari674/Hand-Motion-Identification-for-System-Navigation-and-Control/webEngine/sources/FrontPage.PNG)
				Fig 16 : Front Page of GUI

  This is the front page of GUI after a successful loading of the application. There is an image slider in the forefront with a
welcoming text in its center and a transparent navigation bar on the top. Transparent navigation bar is at the complete right which 
includes four key menus and a three-layer button at its complete left. The ‘Explore Now’ button in its center guides you through the 
application by scrolling the bar with a significant pixel value. All the buttons and elements in this window are responsive and 
represent different part and function in GUI. As you click element ‘Home’ of the navigation bar, the  window will be scrolled down to 
the home division. This behavior is also similar for other elements where they will be scrolled down to their respective divisions. 
Three-layer element in left of the navigation bar is a button for opening sidebar which, contains some of the links like User Manual, 
About Us, etc. to make the user freer to explore application more deeply.
As we scroll further, the GUI looks like this:

![alt text](https://raw.githubusercontent.com/NabinAdhikari674/Hand-Motion-Identification-for-System-Navigation-and-Control/webEngine/sources/Home.PNG)
				Fig 17 : UI Of Home page

This is the Home page which contains the main aspect of this application. A button and a slider are at its left with some description 
for what these all signify. The ‘Start Detection’ button calls a python function as you click to open camera for detection of your hand 
gesture and motion. The ‘Control UI’ switch can be toggled for controlling the UI for navigation. In ‘ON’ state it starts the python 
function for navigating window based on the action assigned to a particular hand gesture and it ends as we switch to ‘OFF’ state. This 
however is applied when you restart the camera (closing the present session of camera, but not exiting the application). 
As we scroll further GUI looks like following:
    
![alt text](https://raw.githubusercontent.com/NabinAdhikari674/Hand-Motion-Identification-for-System-Navigation-and-Control/webEngine/sources/GESTURES.PNG)
				Fig 18 : UI Of Gesture page

This is the Gesture page which contains aspects related to the gestures, actions, addition of new gestures, trainings etc. Two 
buttons ‘View Gesture’ and ‘Add Gesture’ are present on the left side of the division. The ‘View Gesture’ button displays the gestures 
and actions associated with it in a table. The ‘Add Gesture’ button displays an input box where we must enter the name of the gesture, 
we want to add so that we can modify and train the model to recognize the new gestures along with the predefined gestures. If the 
gesture name we entered, is already present then it will alert us to enter another unique name. After this, there are series of steps to 
be followed, so that the newly added gesture can be processed in a machine learning model that is modified and trained to recognize and 
classify gestures along with the newly added gesture. After completing the series of steps, we can view the updated gestures by clicking 
‘View Gesture’ button which then displays updated gesture names and their associated actions in the table.

As we scroll further GUI looks like following :

![alt text](https://raw.githubusercontent.com/NabinAdhikari674/Hand-Motion-Identification-for-System-Navigation-and-Control/webEngine/sources/ACTIONS.PNG)
				Fig 19 : UI Of Action page

This is the Action page which contains aspects related to the gestures, actions, assigning a new action, editing existing 
actions, trainings etc. Two buttons ‘Assign Action’ and ‘Add Gesture’ are present on the right side of the division. The button 
‘Assign Action’ displays the gestures and its associated action in the table that are already been in the product or we can say which
are been properly trained. The ‘Assign Gesture’ button displays an input box where we must choose an action and assign it with a gesture 
present in the product system. Then after we choose any action available, there are series of steps to be followed so that the action 
can be assigned to a gesture. After this, we can again view the updated gestures and actions associated. Clicking the ‘Edit Action’ 
button also prompts us with series of steps to edit any action that is assigned to a gesture. 
