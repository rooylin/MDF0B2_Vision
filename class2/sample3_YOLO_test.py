'''乾燥包影像檢測'''
import pyrealsense2 as rs
import numpy as np
import torch   
import cv2

'''初始設定'''
# 讀取YOLO權重
model = torch.hub.load('yolov5', 'custom', path='desiccantRGB1222.pt', source='local')
model.conf = 0.4    # 信心度閥值
model.iou = 0.7     # IoU閥值
model.max_det = 1   # 最大檢測數量

# 定義realsense
pipeline = rs.pipeline()
config = rs.config()
# 指定realsense裝置
config.enable_device('146322070607')
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

    #檢測結果影像
    output_img = results.render()[0]

    if len(results.pandas().xywh[0]) :  # 如果檢測到物體
        xcenter = int(results.pandas().xywh[0].get('xcenter')[0])   #最佳檢測目標中心x
        ycenter = int(results.pandas().xywh[0].get('ycenter')[0])   #最佳檢測目標中心y

        print(xcenter, ycenter)

    # 顯示影像
    cv2.imshow('output', output_img[:,:,::-1])

# 啟動程式
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
