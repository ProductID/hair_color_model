from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

from deepface import DeepFace
from PIL import ImageDraw


def age_predictor(imageee):
# def age_predicter():
    imageee=imageee
    # imageee='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/celebrity_model-master/media/src_image/17.png'
    img = cv2.imread(imageee)

    obj = DeepFace.analyze(img_path=imageee, actions=['age', 'gender'])
    print(obj["age"], " years old ", obj["gender"])
    print(obj,"----------------------------------------mmmmmmmmmmmmmmmmmmmmmmmmm")
    z = obj['region']
    print(z)
    X = z['x']
    y = z['y']
    w = z['w']
    h = z['h']
    cv2.rectangle(img, (X, y), (X + w, y + h), (0, 255, 255), 2)


    if obj["gender"] == "Woman":
        gender = "Female"
    else:
        gender = "Male"
    text = f'{obj["age"]}\n{gender}'
    print(text,"-----------ttttttttttttttttttt")
    y_start = 150

    y_increment = 100

    for i, line in enumerate(text.split('\n')):

        # print(i, j)
        print(i,line,"=========================")
        y = y_start + i*y_increment
        cv2.putText(img=img, text=line, org=(150, y), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=4, color=(255,255,0),
        thickness=3)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgplot = plt.imshow(img)

    # image name and path
    imageee_lis=imageee.split("/")
    image_name=imageee_lis[-1]
    print(image_name,"----------------imgname")
    # plt.imsave('test.png', img)
    return img,image_name


def race_predictor(imageee):
    imageee=imageee
    img = cv2.imread(imageee)

    obj = DeepFace.analyze(img_path=imageee, actions=['race', 'gender'])
    print(obj["dominant_race"], obj["gender"])
    z = obj['region']
    print(z)
    X = z['x']
    y = z['y']
    w = z['w']
    h = z['h']
    cv2.rectangle(img, (X, y), (X + w, y + h), (0, 255, 255), 2)

    if obj["gender"] == "Woman":
        gender = "Female"
    else:
        gender = "Male"
    text = f'{obj["dominant_race"]}\n{gender}'
    print(text, "-----------ttttttttttttttttttt")
    y_start = 150

    y_increment = 100

    for i, line in enumerate(text.split('\n')):
        # print(i, j)
        print(i, line, "=========================")
        y = y_start + i * y_increment
        cv2.putText(img=img, text=line, org=(150, y), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=4,
                    color=(255, 255, 0),
                    thickness=3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgplot = plt.imshow(img)

    # image name and path
    imageee_lis = imageee.split("/")
    image_name = imageee_lis[-1]
    print(image_name, "----------------imgname")

    # plt.imsave('test1.png', img)
    return img, image_name

if __name__=='__main__':
    imageee='/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/celebrity_model-master/static/input_images/1_CTft3D8.jpeg'
    age_predictor(imageee)