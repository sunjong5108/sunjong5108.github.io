import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import time
import json
import ray
import grpc
from concurrent.futures import ThreadPoolExecutor
from ast import literal_eval
import logging

from test_start_game import Arma3Launcher
from game_record import GameRecorder
import communicator_pb2
import communicator_pb2_grpc
import sqlite3

logging.basicConfig(level=logging.INFO)

# SQL DB 초기화
@ray.remote
def init_db(db_name: str) -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            json_data TEXT
        )
    """)
    conn.commit()
    conn.close()

# SignalActor: 녹화 루프 종료 여부를 제어하는 플래그를 보관합니다.
@ray.remote
class SignalActor:
    def __init__(self):
        self.flag = True

    def get_flag(self):
        return self.flag

    def set_flag(self, value: bool):
        self.flag = value

# record_loop: 별도의 Ray remote task로 실행되어 녹화 루프를 수행합니다.
@ray.remote(num_gpus=1)
def record_loop(recorder, duration: float, signal_actor) -> str:
    # should_stop 콜백: SignalActor의 flag가 False이면 녹화를 중단합니다.
    def should_stop():
        return not ray.get(signal_actor.get_flag.remote())
    recorder.start_recording(duration, should_stop)
    return "Recording finished"

# save sql db
@ray.remote
def save_sql_db(db_name: str, msg, frame) -> str:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    for data in msg:
        try:
            # JSON 데이터를 문자열로 변환하여 저장
            if frame is not None:
                data['frame_id'] = frame[0]
                data['frame'] = ray.get(frame[1][0])
                logging.info(f"frame data save")
            else:
                logging.info(f"frame data is None")
            cursor.execute("INSERT INTO records (json_data) VALUES (?)", (json.dumps(data),))
            conn.commit()
            logging.info(f"Save Data")

        except json.JSONDecodeError:
            print("Fail JSON decoding")
    conn.close()
    return "Save data"

# RecorderActor: 녹화 작업을 제어하며, record_loop 작업을 실행합니다.
@ray.remote
class RecorderActor:
    def __init__(self, window_keyword: str, output_file: str, fps: float) -> None:
        self.window_keyword = window_keyword
        self.output_file = output_file
        self.fps = fps
        self.signal_actor = SignalActor.options(namespace="make_sa_data").remote()
        self.recorder = GameRecorder(self.window_keyword, self.output_file, self.fps)
        self.record_task = None

    def start(self, duration: float = 99999) -> str:
        # 녹화 시작 전 플래그를 True로 설정합니다.
        ray.get(self.signal_actor.set_flag.remote(True))
        # record_loop를 원격 작업으로 실행합니다.
        self.record_task = record_loop.remote(self.recorder, duration, self.signal_actor)
        return "Recording started"

    def stop(self) -> str:
        # 녹화 중지를 위해 플래그를 False로 설정합니다.
        ray.get(self.signal_actor.set_flag.remote(False))
        if self.record_task is not None:
            ray.get(self.record_task)
        return "Recording stopped"
    
    # def get_latest_frame(self, timeout=1.0):
    #     # GameRecorder의 새로 추가한 메서드를 호출하여 최신 프레임을 반환합니다.
    #     return ray.get(self.recorder.get_latest_frame(timeout))

# gRPC 서비스를 처리하는 CustomBridgeServiceServicer
class CustomBridgeServiceServicer(communicator_pb2_grpc.BridgeServiceServicer):
    def __init__(self, recorder_actor, shutdown_controller, db_name) -> None:
        """
        :param recorder_actor: RecorderActor의 Ray 핸들러
        :param shutdown_controller: GRPCBridgeServer actor (shutdown 명령 전달용)
        """
        self.recorder_actor = recorder_actor
        self.shutdown_controller = shutdown_controller
        self.db_name = db_name
        self.buffer = ray.get_actor("global_buffer", namespace="make_sa_data")
        self.last_processed_id = -1

    def SendMessage(self, request, context):
        logging.info("[CustomBridgeService] Received message: %s from %s",
                     request.content, request.sender)
        end_flag = False
        if request.sender == "arma_msg":
            try:
                content = literal_eval(request.content)
                if content.get("start") == 1:
                    result = ray.get(self.recorder_actor.start.remote())
                    logging.info("Start command processed.")
                    return communicator_pb2.MessageReply(result=result)
                elif content.get("end") == 1:
                    result = ray.get(self.recorder_actor.stop.remote())
                    end_flag = True
                    logging.info("Stop command processed.")
                    # 녹화 종료 시 shutdown_controller에 종료 신호 전달
                    self.shutdown_controller.set_shutdown()
                    return communicator_pb2.MessageReply(result=result)
            except Exception as e:
                logging.error("Error processing recording command: %s", e)
                return communicator_pb2.MessageReply(result=f"Error: {e}")
        elif request.sender == "arma 3 obj" and not end_flag:
            try:
                content = literal_eval(request.content)
                frame = ray.get(self.buffer.get_recent_refs.remote(self.last_processed_id)) 
                result = ray.get(save_sql_db.remote(self.db_name, content, frame))
                logging.info("Save data to SQL DB")
                return communicator_pb2.MessageReply(result=result)
            except Exception as e:
                logging.error("Error processing recording command: %s", e)
                return communicator_pb2.MessageReply(result=f"Error: {e}")  
        return communicator_pb2.MessageReply(result="Default processing")

# GRPCBridgeServer: gRPC Bridge 서버를 Ray actor로 실행합니다.
@ray.remote
class GRPCBridgeServer:
    def __init__(self, recorder_actor, db_name: str):
        self.recorder_actor = recorder_actor
        self.shutdown_flag = False
        self.db_name = db_name
        self.server = grpc.server(ThreadPoolExecutor(max_workers=10))
        servicer = CustomBridgeServiceServicer(self.recorder_actor, self, self.db_name)
        communicator_pb2_grpc.add_BridgeServiceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port("[::]:50051")
        self.server.start()
        logging.info("[GRPCBridgeServer] Server started on port 50051.")

    def wait_for_shutdown(self):
        while not self.shutdown_flag:
            time.sleep(1)
        self.server.stop(0)
        logging.info("[GRPCBridgeServer] Server stopped.")

    def set_shutdown(self):
        self.shutdown_flag = True
    

# launch_arma3_mission: Arma 3 미션을 실행하는 remote task (동기 함수)
@ray.remote
def launch_arma3_mission(scenario_name: str) -> str:
    launcher = Arma3Launcher()
    try:
        launcher.check_paths()
    except Exception as e:
        return f"Path error: {e}"
    logging.info("Launching Arma 3 mission...")
    process = launcher.launch_eden_editor(scenario_name)
    if process:
        time.sleep(5)
        logging.info("Arma 3 mission launched.")
        return "Arma 3 mission launched."
    return "Arma 3 mission launch failed."

def main() -> None:
    ray.init(namespace="make_sa_data")
    
    WINDOW_KEYWORD = r"C:\Program Files (x86)\Steam\steamapps\common\Arma 3\arma3_x64.exe"
    OUTPUT_FILE = "arma3_recording.mp4"
    FPS = 60.0
    DB_NAME = 'data_record1.db'
    
    #SQL DB init
    init_db.remote(DB_NAME)

    # RecorderActor를 생성합니다.
    recorder_actor = RecorderActor.options(namespace="make_sa_data").remote(WINDOW_KEYWORD, OUTPUT_FILE, FPS)

    # gRPC 서버들을 Ray actor로 등록합니다.
    grpc_bridge_server = GRPCBridgeServer.options(namespace="make_sa_data").remote(recorder_actor, DB_NAME)
    # grpc_data_generator_server = GRPCDataGeneratorServer.remote()

    # GRPCBridgeServer의 wait_for_shutdown()을 비동기로 호출해 종료 신호를 대기합니다.
    shutdown_future = grpc_bridge_server.wait_for_shutdown.remote()

    # Arma 3 미션 실행
    mission_task = launch_arma3_mission.remote("test_sqf_v3.Altis")
    mission_result = ray.get(mission_task)
    logging.info("Mission result: %s", mission_result)

    logging.info("Waiting for shutdown signal from GRPCBridgeServer...")
    try:
        ray.get(shutdown_future)
    except Exception as e:
        logging.error("Error waiting for shutdown: %s", e) 
    finally:
        ray.shutdown()
        logging.info("Ray has been shutdown.")

if __name__ == "__main__":
    main()
