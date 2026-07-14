import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread(r"C:\Users\priya\computer_vision\computer_vision_task2.jpg")

if image is None:
    print("Image not Found")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


resized = cv2.resize(image, (640,480))



cropped = image[100:500,200:800]



height,width = image.shape[:2]

center = (width//2,height//2)

matrix = cv2.getRotationMatrix2D(center,30,1)

rotated = cv2.warpAffine(image,matrix,(width,height))



flipped = cv2.flip(image,1)



blurred = cv2.GaussianBlur(image,(9,9),0)


kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])

sharpened = cv2.filter2D(image,-1,kernel)


images = [
    rgb,
    cv2.cvtColor(resized,cv2.COLOR_BGR2RGB),
    cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB),
    cv2.cvtColor(rotated,cv2.COLOR_BGR2RGB),
    cv2.cvtColor(flipped,cv2.COLOR_BGR2RGB),
    cv2.cvtColor(blurred,cv2.COLOR_BGR2RGB),
    cv2.cvtColor(sharpened,cv2.COLOR_BGR2RGB)
]

titles = [
    "Original",
    "Resized",
    "Cropped",
    "Rotated",
    "Flipped",
    "Blurred",
    "Sharpened"
]

plt.figure(figsize=(15,10))

for i in range(len(images)):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i])

    plt.title(titles[i])

    plt.axis("off")

plt.tight_layout()

plt.show()