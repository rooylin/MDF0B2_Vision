'''realsense顯示影像'''
import pyrealsense2 as rs
import numpy as np
import cv2

# 定義realsense
pipeline = rs.pipeline()
config = rs.config()
# 指定realsense裝置
config.enable_device('146322070607')
# 設定realsense
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 深度圖解析度
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)# 彩色圖解析度
# 啟動realsense
pipeline.start(config)

while True:
    # 讀取影像
    frames = pipeline.wait_for_frames()  
    # 取得深度禎
    depth_frame = frames.get_depth_frame()
    # 取得彩色禎
    color_frame = frames.get_color_frame()


    # 轉換影像格式
    color_image = np.asanyarray(color_frame.get_data())
    # 轉換深度影像至彩色影像
    depth_image = np.asanyarray(rs.colorizer(1).colorize(depth_frame).get_data())

    # 將影像拼接在一起
    images = np.hstack((color_image, depth_image))

    # 顯示影像
    cv2.imshow('images', images)
    key = cv2.waitKey(1) 

    # 按下Esc關閉
    if key == 27:
        # 關閉相機
        pipeline.stop()
        break
