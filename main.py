import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
st.image('C:\/Users\Personal\Documents\Python\MatematicasConGestos\Matematicas.png')

colum1, colum2 = st.columns([2,1])
with colum1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
with colum2:
    output_text_area = st.title('Respuesta')
    output_text_area = st.subheader("")


#Utilizar la API de Google Gemini
genai.configure(api_key="AIzaSyAXd0UElxbXCjYRPh-pP7TQbdhN0FjNYl8")
model = genai.GenerativeModel("gemini-1.5-flash")


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.6)

prev_pos = None
canvas = None

prompt = "De identificarse un problema matemático identificarlo nombrandolo, mostrar el proceso y el resultado, si no, enviar una notificación \"No se detecta ningún problema matemático, intente nuevamente.\""


def getHandInfo(img, bDraw = True):
    hands, img = detector.findHands(img, draw=bDraw, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        #length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),scale=10)

        return fingers, lmList
    else:
        return None

def draw(info,prev_pos, canvas, img):
    fingers, lmList = info
    current_pos = lmList[8][0:2]
    if prev_pos is None: prev_pos = current_pos
    cv2.circle(img,lmList[8][0:2],5,(0,0,255),5)
    if fingers == [0,1,0,0,0]:
        cv2.line(canvas, current_pos,prev_pos, (13, 155, 212),5)
    if fingers == [0, 1, 1, 0, 0]:
        cv2.line(canvas, current_pos, prev_pos, (0, 0, 0), 50)
    if fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
    return current_pos, canvas

def sendToAi(model, canvas, fingers):
        pil_image = Image.fromarray(canvas)
        response = model.generate_content([prompt,pil_image])
        #print(response.text)
        output_text_area.text(response.text)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    handInfo = getHandInfo(img, bDraw = False)
    if handInfo:
        fingers, lmList = handInfo
        #print(fingers)
        prev_pos, canvas = draw(handInfo, prev_pos, canvas, img)
        if fingers == [1, 1, 1, 1, 0]:
            sendToAi(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.65, canvas, 0.35, 0)
    FRAME_WINDOW.image(image_combined,channels="BGR")
    #cv2.imshow("last",image_combined)
    #cv2.imshow("Image", img)
    #cv2.imshow("Canvaaas", canvas)
    cv2.waitKey(1)

