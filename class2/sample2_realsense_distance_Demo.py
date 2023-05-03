'''realsense取得畫面座標距離'''
import pyrealsense2 as rs
import numpy as np
import cv2

# 定義realsense
pipeline = rs.pipeline()
config = rs.config()
# 設定realsense
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 深度圖解析度
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)# 彩色圖解析度
# 啟動realsense
pipeline.start(config)

def realsenseXY2XYZ(pl, x=0, y=0):
    '''realsense_2D座標轉3D座標'''
    # 2D與3D影像對齊
    aligned_depth_frame = rs.align(rs.stream.color).process(pl.wait_for_frames()).get_depth_frame()
    # 取得相機參數
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # 取得深度資訊
    dis = aligned_depth_frame.get_distance(x, y)
    # 座標轉換
    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x, y], depth=dis)
    # 回傳XYZ
    return camera_coordinate

while True:
    # 讀取影像
    frames = pipeline.wait_for_frames()
    # 取得彩色影像
    color_frame = frames.get_color_frame()
    # 轉換影像格式
    color_image = np.asanyarray(color_frame.get_data())

    # 計算物體3D空間位置(X,Y,Z 單位m)
    xyz = realsenseXY2XYZ(pipeline, 320, 240)

    # 轉換3D空間位置(單位mm)
    x = int(xyz[0] * 1000)
    y = int(xyz[1] * 1000)
    z = int(xyz[2] * 1000)
    # 打印結果
    print(x, y, z)

    # 影像中心畫圓
    cv2.circle(color_image,(320, 240), 3, (0, 255, 255), -1)

    # 將結果畫至影像
    cv2.putText(color_image, f'X:{x}, Y:{y}, Z:{z}', (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2, cv2.LINE_AA)

    # 顯示影像
    cv2.imshow('color_image', color_image)
    key = cv2.waitKey(1) 

    # 按下Esc關閉
    if key == 27:
        # 關閉相機
        pipeline.stop()
        break