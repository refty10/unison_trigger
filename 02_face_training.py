import cv2
import numpy as np
from PIL import Image
import os


def get_images_and_labels(path):

    # 顔認識用特徴量ファイルの読み込み
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    image_paths = [os.path.join(path,f) for f in os.listdir(path)]   # 写真のpathのリストを作成
    face_samples = []
    ids = []

    for image_path in image_paths:

        PIL_img = Image.open(image_path).convert('L')   # 画像を読み込み，グレースケールに変換
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return face_samples,ids


def main():

    path = 'dataset'  # 学習用の写真が格納されているディレクトリ
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print ("\n [INFO] 顔を学習しています...")
    faces,ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))

    # モデ保存用のディレクトリが存在しない場合は作成
    if not os.path.exists('trainer'):
        os.makedirs('trainer')

    # 作成したモデルを保存
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    print(f'\n [INFO] {len(np.unique(ids))} 学習完了!')


if __name__ == '__main__':
    main()