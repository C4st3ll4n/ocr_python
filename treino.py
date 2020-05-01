from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import skimage.measure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial',max_iter=200000,verbose=1,tol=0.00001)

X = []
Y = []

img = np.zeros((50, 30, 3), np.uint8)
X.append(img)
Y.append("fundo")

fonts = os.listdir('fonts/')
os.system("mkdir images")
im_folder = os.listdir('images/')

if len(im_folder) < 1:
    for c in range(48,91):
        if c < 58 or c > 64:
            os.system('mkdir "images/{}"'.format(str(chr(c))))
im_folder = os.listdir('images/')

print(sorted(im_folder))

for c in range(48, 91):
    count = 0
    if c < 58 or c > 64:
        for i in range(40, 42):
            for fontpath in fonts:
                img = np.zeros((50, 30, 3), np.uint8)

                b, g, r, a = 255, 255, 255, 0
                font = ImageFont.truetype("fonts/"+fontpath, i)

                img_pill = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pill)

                draw.text((1, 1), str(chr(c)), font=font, fill=(b, g, r, a))

                img = np.array(img_pill)
                img = np.invert(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

                cv2.imwrite('images/'+ str(chr(c)) + '/' + str(chr(c)) + str(count) + ".jpg", img)

                print("Salvando: "+"images/"+ str(chr(c)) + "/" + str(chr(c)) + str(count) + ".jpg")

                img = img.astype(float) / 255.0

                X.append(img)
                Y.append(str(chr(c)))

                count += 1

                cv2.imshow("res", img)
                k = cv2.waitKey(60)
                if k == ord('q'):
                    exit()


X = np.array(X).reshape(len(Y), -1)

Y = np.array(Y)
Y = Y.reshape(-1)

ss = StandardScaler()
print("Treinando modelo")

X = X - (X/127.5)

clf.fit(X, Y)
print("Salvando modelo")
ss.fit(X)
joblib.dump((clf, ss), 'caracteres.pkl', compress=3)
print("Modelo salvo")
