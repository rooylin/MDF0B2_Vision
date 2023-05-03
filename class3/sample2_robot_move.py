'''透過mc_protocol控制手臂移動'''
import time
import pymcprotocol

try:
    # 建立物件並選擇型號
    pymc3e = pymcprotocol.Type3E(plctype="iQ-L")
    # 與設備連線連線
    pymc3e.connect("192.168.2.101", 5004)

except Exception as e:
    # 連線失敗顯示錯誤訊息並結束程式
    print(e)
    exit()

def read_D(D = 0, V = 1):
    '''讀取D值功能: 起始位置, 讀取數量'''

    # 格式轉換
    D = f'D{D}'
    # 讀取設備
    wordunits_values = pymc3e.batchread_wordunits(headdevice=D, readsize=V)
    # 回傳結果
    return wordunits_values

def write_D(D = 0, V = [0]):
    '''寫入D值功能: 起始位置, 資料list'''
    
    # 格式轉換
    D = f'D{D}'
    # 寫入設備
    pymc3e.batchwrite_wordunits(headdevice=D, values=V)
    
def robot_move(pose): 
    '''手臂移動'''
    write_D(D=2020, V=pose)  # 寫入POSE，pose為一個List
    write_D(D=2001, V=[1])  # 執行動作觸發
    time.sleep(0.1) # 等待0.1秒
    write_D(D=2001, V=[0])  # 執行動作觸發復歸

if __name__ == '__main__':
    '''測試'''
    x, y, z, rx, ry, rz = 200, 10, 100, 0, 0, 0 # 手臂六姿態
    robot_move([200, 10, 100, 0, 0, 0]) # 移動手臂
    time.sleep(5) # 等待手臂作動完成