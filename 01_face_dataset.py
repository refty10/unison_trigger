import cv2
import os


def main():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # 横幅の指定
    cam.set(4, 480)  # 高さの指定

    # 顔認識用特徴量ファイルの読み込み
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 画像保存用のディレクトリが存在しない場合は作成
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # その人のIDを指定(数字のみ)
    face_id = input('\n enter user id end press <return> ==>  ')

    print("\n [INFO] カメラを初期化しています. カメラの方を向いた状態で待ってね ...")

    count = 0

    while(True):

        try:
            ret, img = cam.read()   # カメラから画像を取得
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # グレースケール変換
            faces = face_detector.detectMultiScale(gray, 1.3, 5)   # 顔を検出

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                print(f'撮影枚数 = {count}枚')

                # 撮影した写真を顔部分だけ切り抜いて保存
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff   # ESCキーで停止
            if k == 27:
                break
            elif count >= 30:   # 30枚撮影したら終了
                break
            
        except:
            pass

    # キャプチャをリリースして，ウィンドウをすべて閉じる
    cam.release()
    cv2.destroyAllWindows()

    print("\n [INFO] 完了しました!")


if __name__ == '__main__':
    main()