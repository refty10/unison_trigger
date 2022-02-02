import cv2
import time
import os
from monitor_ctl import MonitorCtl
from mtcnn import MTCNN
from difference_area import calc_amount_of_area


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlowのログを抑制

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')   # 学習したモデルデータのパス

    # MTCNN
    mtcnn_detector = MTCNN()

    # フォントの指定
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_id = 0

    # names related to ids: example ==> ogane: face_id=1
    names = ['None', 'ogane']

    # キャプチャの設定
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # 横幅
    cam.set(4, 480)  # 高さ

    # 顔として認識される最小ウィンドウサイズを定義
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    mc = MonitorCtl()
    unknown_count = 0

    # 変化量の計算用
    avg = None
    loop_count = 0

    while True:
        # 開始直後はおかしな画像が入るので無視する
        if loop_count <= 5:
            loop_count += 1
            continue

        try:
            ret, img = cam.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # グレースケールに変換
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGRからRGBに変換
            mtcnn_dets = mtcnn_detector.detect_faces(rgb_img)  # MTCNN顔検出

            # 前フレームを保存
            if avg is None:
                avg = gray_img.copy().astype("float")
                continue

            # 変化量を計算
            change_rate_img, change_rate = calc_amount_of_area(gray_img, avg)
            print(f'Change_Rate = {change_rate}%')

            # リアルタイムに差分領域を表示
            cv2.imshow('Change_Rate_img', change_rate_img)

            print(unknown_count)
            # 知らない顔を3回以上認識かつ画像に変化がなければ消す
            if unknown_count > 3 and change_rate <= 2:
                mc.sleep()

            # なにか動いてるものがあるときつけとく
            elif change_rate > 0.5:
                mc.wake_up()

            # それ以外はとりあえずつけとく
            else:
                mc.wake_up()

            for face in mtcnn_dets:
                x, y, w, h = face['box']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                face_id, confidence = recognizer.predict(gray_img[y:y+h, x:x+w])

                # 信頼度が100未満かどうかを確認==>「0」の場合は完全に一致
                if (round(100 - confidence) > 50):
                    unknown_count = 0
                    face_id = names[face_id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    face_id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                    unknown_count += 1

                # 名前と一致度を表示
                cv2.putText(img, str(face_id), (x+5, y-5),
                            font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x+5, y+h-5),
                            font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff   # ESCキーで終了
            if k == 27:
                break

        except:
            pass

        time.sleep(0.1)

    # カメラの後始末
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
