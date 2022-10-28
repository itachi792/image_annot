import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
from gtts import gTTS
import os


def get_output_layers(net):
    layer_names = net.getLayerNames()
    print("layers name = ", layer_names)
    print("unconnected layers = ", net.getUnconnectedOutLayers())
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print("outpy layer = ", output_layers)
    return output_layers


# open dialog box to select the image file
image_name = askopenfilename()

config = "yolov3.cfg"
weights = "yolov3.weights"
class_file = "yolov3.txt"

net = cv2.dnn.readNet(weights, config)

classes = None


with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


image = cv2.imread(image_name)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

#blogFromImage(image, sclefactor, size(H*W), rgb_mean(R, G, B), swapRB)
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

object_list = set()

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            object_list.add(classes[class_id])

print(object_list)

with open("object_detected_name.txt", 'w') as f:
    for word in object_list:
        f.write(word)
        f.write("\n")

cv2.imshow("object detection", image)
# cv2.waitKey()

# Read the text file generated after object detection

words = []

with open("object_detected_name.txt", 'r') as f:
    for word in f.readlines():
        words.append(word.strip("\n"))

print("word = ", words)
text = " and ".join(words)
print(text)

myobj = gTTS(text=text, lang="en", slow=True)
myobj.save("voice.mp3")
os.system("start voice.mp3")

