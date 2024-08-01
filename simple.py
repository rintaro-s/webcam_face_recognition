
import face_recognition
import cv2
import numpy as np

# これはウェブカメラからのライブビデオで顔認識を実行する超簡単（しかし遅い）な例です。
# もう少し複雑ですが、より速く動作する例もあります。

# 注意：この例では、ウェブカメラからの読み取りに OpenCV（`cv2` ライブラリ）が必要です。
# OpenCV は face_recognition ライブラリの使用には必要ありません。この特定のデモを実行する場合のみ必要です。
# インストールに問題がある場合は、OpenCV を必要としない他のデモを試してみてください。

# ウェブカメラ #0（デフォルトのもの）への参照を取得
video_capture = cv2.VideoCapture(0)

# サンプル画像を読み込み、それを認識する方法を学習する
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# 2つ目のサンプル画像を読み込み、それを認識する方法を学習する
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# 既知の顔のエンコーディングとその名前の配列を作成
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

while True:
    # ビデオの単一フレームを取得
    ret, frame = video_capture.read()

    # 画像を BGR カラー（OpenCV が使用）から RGB カラー（face_recognition が使用）に変換
    rgb_frame = frame[:, :, ::-1]

    code = cv2.COLOR_BGR2RGB
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)

    # ビデオのフレーム内のすべての顔と顔エンコーディングを検出
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # このフレーム内の各顔をループ処理
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 顔が既知の顔と一致するか確認
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # 一致が見つかった場合は、最初の一致を使用する
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # 代わりに、新しい顔に対する距離が最も小さい既知の顔を使用
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 顔の周りにボックスを描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 顔の下に名前のラベルを描画
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 結果の画像を表示
    cv2.imshow('Video', frame)

    # 'q' キーを押すと終了します！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ウェブカメラのハンドルを解放
video_capture.release()
cv2.destroyAllWindows()
