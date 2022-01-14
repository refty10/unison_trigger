import cv2
import os
from mtcnn import MTCNN


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlowのログを抑制
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # 横幅の指定
    cam.set(4, 480)  # 高さの指定

    # MTCNN
    mtcnn_detector = MTCNN()

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
            plot_img = img.copy()   # 表示用に元画像をコピー

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
            mtcnn_dets = mtcnn_detector.detect_faces(rgb_img)  # MTCNN顔検出

            # MTCNNの顔検出箇所の矩型描画ループ
            for face in mtcnn_dets:
                x, y, w, h = face['box']
                cv2.rectangle(plot_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                print(f'撮影枚数 = {count}枚')

                # 撮影した写真を顔部分だけ切り抜いて保存
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])

                cv2.imshow('image', plot_img)

            k = cv2.waitKey(100) & 0xff   # ESCキーで停止
            if k == 27:
                break
            elif count >= 100:   # 100枚撮影したら終了
                break

        except:
            print("Error")

    # キャプチャをリリースして，ウィンドウをすべて閉じる
    cam.release()
    cv2.destroyAllWindows()

    print("\n [INFO] 完了しました!")


if __name__ == '__main__':
    main()