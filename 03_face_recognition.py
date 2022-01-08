import cv2


def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')   # 学習したモデルデータのパス
    cascade_path = 'haarcascade_frontalface_default.xml'   # 顔認識用特徴量ファイルのパス
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # フォントの指定
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_id = 0

    # names related to ids: example ==> ogane: face_id=1
    names = ['None', 'ogane']

    # キャプチャの設定
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # 横幅
    cam.set(4, 480) # 高さ

    # 顔として認識される最小ウィンドウサイズを定義
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        try:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )

            for(x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                face_id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                # 信頼度が100未満かどうかを確認==>「0」の場合は完全に一致
                if (confidence < 100):
                    face_id = names[face_id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    face_id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(face_id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

            cv2.imshow('camera',img)

            k = cv2.waitKey(10) & 0xff   # ESCキーで終了
            if k == 27:
                break
        
        except:
            pass

    # カメラの後始末
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()