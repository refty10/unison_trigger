import cv2
import numpy as np
import os
from mtcnn import MTCNN


def get_images_and_labels(path):

    # MTCNN
    mtcnn_detector = MTCNN()

    image_paths = [os.path.join(path,f) for f in os.listdir(path)]   # 写真のpathのリストを作成
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path)   # 画像を読み込み
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGRからRGBに変換
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGRからグレースケールに変換
        mtcnn_dets = mtcnn_detector.detect_faces(rgb_img)  # MTCNN顔検出

        id = int(os.path.split(image_path)[-1].split(".")[1])

        for face in mtcnn_dets:
            x, y, w, h = face['box']
            face_samples.append(gray_img[y:y + h, x:x + w])
            ids.append(id)

    return face_samples,ids


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlowのログを抑制
    path = 'dataset'  # 学習用の写真が格納されているディレクトリ
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print ("\n [INFO] 顔を学習しています...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))

    # モデル保存用のディレクトリが存在しない場合は作成
    if not os.path.exists('trainer'):
        os.makedirs('trainer')

    # 作成したモデルを保存
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    print(f'\n [INFO] {len(np.unique(ids))} 学習完了!')


if __name__ == '__main__':
    main()