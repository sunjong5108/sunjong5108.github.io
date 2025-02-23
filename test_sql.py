import sqlite3
import json
import base64
import numpy as np
import cv2

def decode_base64_image(base64_str):
    # base64 문자열 디코딩: 바이트 데이터 생성
    img_data = base64.b64decode(base64_str)
    # 바이트 데이터를 numpy 배열로 변환 (uint8 타입)
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    # numpy 배열을 이미지로 디코딩 (cv2.IMREAD_COLOR: 컬러 이미지)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# SQLite 데이터베이스에 연결
conn = sqlite3.connect('data_record1.db')

# 커서 생성
cursor = conn.cursor()

# 데이터 조회
cursor.execute('SELECT * FROM records')
rows = cursor.fetchall()

group_type = []
obj_type = []
# 결과 출력
for row in rows:
    stored_dict = json.loads(row[1])  # 두 번째 열은 'data' 컬럼
    print(stored_dict)
    # cv2.imshow("Decoded Image", decode_base64_image(stored_dict['frame']))
    # cv2.waitKey(0)  # 아무 키나 누르면 창이 닫힙니다.
    # cv2.destroyAllWindows()
    obj_type.append(stored_dict['obj_name'])
    group_type.append(stored_dict['group_name'])

# 연결 종료
conn.close()

obj_type = list(set(obj_type))
group_type = list(set(group_type))
print(obj_type)
print(group_type)