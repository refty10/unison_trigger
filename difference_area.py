import cv2

def main():
    # キャプチャの設定
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # 横幅
    cam.set(4, 480) # 高さ

    avg = None
    count = 0

    while(True):
        ret, frame = cam.read()

        # 開始直後はおかしな画像が入るので無視する
        if count <= 5:
            count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 前フレームを保存
        if avg is None:
            avg = gray.copy().astype("float")
            continue

        # 現在のフレームと移動平均との間の差を計算する
        # accumulateWeighted関数の第三引数は「どれくらいの早さで以前の画像を忘れるか」。小さければ小さいほど「最新の画像」を重視する。
        # http://opencv.jp/opencv-2svn/cpp/imgproc_motion_analysis_and_object_tracking.html
        # 小さくしないと前のフレームの残像が残る
        # 重みは蓄積し続ける。
        cv2.accumulateWeighted(gray, avg, 0.1)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        # 閾値を設定し、フレームを2値化
        thresh = cv2.threshold(frameDelta, 80, 255, cv2.THRESH_BINARY)[1]

        # リアルタイムに差分領域を表示
        cv2.imshow('th', thresh)

        # 全体の画素数
        all_area_pixel = thresh.size
        # 白部分と黒部分の画素数
        white_area_pixel = cv2.countNonZero(thresh)
        black_area_pixel = all_area_pixel - white_area_pixel

        # 白色と黒色の割合
        white_area = white_area_pixel / all_area_pixel * 100
        black_area = black_area_pixel / all_area_pixel * 100

        #それぞれの割合を表示
        print(f'White_Area = {white_area}%')
        # print(f'Black_Area = {black_area}%\n')

        # ESCキーで終了
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    # カメラの後始末
    cv2.waitKey(0)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()