'''mc protocol 連線讀寫D值'''
import time
import pymcprotocol


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

if __name__ == '__main__':
    '''測試'''

    try:
        # 建立物件並選擇型號
        pymc3e = pymcprotocol.Type3E(plctype="iQ-L")
        # 與設備連線連線
        pymc3e.connect("192.168.2.101", 5004)

    except Exception as e:
        # 連線失敗顯示錯誤訊息並結束程式
        print(e)
        exit()

    # 寫入D2020, 資料為[1]
    write_D(2020, [1])
    # 等待0.1秒
    time.sleep(0.1)
    # 讀取D2020後的 1 筆資料
    print(read_D(2020, 1))