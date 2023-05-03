import time
import torch   
import cv2     
import numpy as np 
import pyrealsense2 as rs
import pymcprotocol

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


'''初始設定'''
# PLC連線

try:
    # 建立物件並選擇型號
    pymc3e = pymcprotocol.Type3E(plctype="iQ-L")
    # 與設備連線連線
    pymc3e.connect("192.168.2.101", 5004)

except Exception as e:
    # 連線失敗顯示錯誤訊息並結束程式
    print(e)
    exit()

# 讀取YOLO權重
model = torch.hub.load('yolov5', 'custom', path='PFS20230502.pt', source='local')
model.conf = 0.5   # 信心度閥值
model.iou = 0.7     # IoU閥值
model.max_det = 1   # 最大檢測數量

# 定義realsense
pipeline = rs.pipeline()
config = rs.config()
# 設定realsense
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 深度圖解析度
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)# 彩色圖解析度
# 啟動realsense
pipeline.start(config)

def main():
    '''主程式'''
    # 讀取影像
    frames = pipeline.wait_for_frames()  
    # 取得彩色影像，並轉換格式
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # YOLO檢測
    results = model(color_image, augment=True, size=1280)

    # 檢測結果影像
    output_img = results.render()[0]

    if len(results.pandas().xywh[0]) :  # 如果檢測到物體
        xcenter = int(results.pandas().xywh[0].get('xcenter')[0])   #最佳檢測目標中心x
        ycenter = int(results.pandas().xywh[0].get('ycenter')[0])   #最佳檢測目標中心y

        # 計算物體3D空間位置(X,Y,Z)
        xyz = realsenseXY2XYZ(pipeline, xcenter, ycenter)

        # 取得目標裁切影像
        # crop = results.crop(save = False)[0].get('im')

        # 轉換3D空間位置(單位mm)
        x = int(xyz[0] * 1000)
        y = int(xyz[1] * 1000)
        z = int(xyz[2] * 1000)

        # 打印結果
        print(x, y, z)

        # 將結果畫至影像
        cv2.putText(output_img, f'X:{x}, Y:{y}, Z:{z}', (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)

    else:   # 如果未檢測到物體
        # 在影像上顯示未檢測到物體
        cv2.putText(output_img, f'No object', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        
    # 顯示影像
    cv2.imshow('output', output_img[:,:,::-1])

if __name__ == '__main__':
    # 進入迴圈
    while True:
        # 執行主程式
        main()

        # 等待X毫秒，按下ESC跳出
        key = cv2.waitKey(1) 
        if key == 27:
            # 關閉相機
            pipeline.stop()
            break