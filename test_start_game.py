import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)


class Arma3Launcher:
    def __init__(self) -> None:
        # 기본 실행 경로 및 문서 경로 설정
        self.arma3_path: str = r"C:\Program Files (x86)\Steam\steamapps\common\Arma 3\arma3_x64.exe"
        self.documents_path: Path = Path.home() / "OneDrive" / "문서" / "Arma 3" / "missions"
        # 사용하고자 하는 모드 목록
        self.mods: List[str] = [
            # "!Workshop\\@CBA_A3",
            # "!Workshop\\@ACE",
            # "!Workshop\\@RHSUSAF",
            "!Workshop\\@Pythia",
            "!Workshop\\@3den Enhanced",
            # "!Workshop\\@ASR AI3",
            # "!Workshop\\@CBA_A3",
            "!Workshop\\@FileXT",
            "!Workshop\\test_car",
        ]

    def create_launch_command(self, scenario_name: Optional[str] = None) -> List[str]:
        """Arma 3 Eden Editor 실행 명령어 생성"""
        command = [
            self.arma3_path,
            "-skipIntro",
            "-noSplash",
            "-world=empty",
            "-showScriptErrors",
            "-noPause",
            "-noPauseAudio",
            "-noBattlEye",
            "-window",
        ]

        if self.mods:
            mods_param = f"-mod={';'.join(self.mods)}"
            command.append(mods_param)

        if scenario_name:
            mission_file = self.documents_path / scenario_name / "mission.sqm"
            if mission_file.exists():
                command.append(f'-init=playMission["","{scenario_name}",true]')
        return command

    def launch_eden_editor(self, scenario_name: Optional[str] = None) -> Optional[subprocess.Popen]:
        """Eden Editor 실행 및 프로세스 반환 (실패 시 None)"""
        try:
            command = self.create_launch_command(scenario_name)
            logging.info("Launching Arma 3 Eden Editor...")
            logging.info("Command: %s", " ".join(command))
            process = subprocess.Popen(command)
            time.sleep(5)
            if process.poll() is None:
                logging.info("Arma 3 Eden Editor launched successfully!")
                return process
            else:
                logging.error("Failed to launch Arma 3")
                return None
        except Exception as e:
            logging.error("Error launching Arma 3: %s", e)
            return None

    def check_paths(self) -> None:
        """필요한 경로의 존재 여부를 검사"""
        if not os.path.exists(self.arma3_path):
            raise FileNotFoundError(f"Arma 3 executable not found at: {self.arma3_path}")
        if not self.documents_path.exists():
            raise FileNotFoundError(f"Arma 3 missions directory not found at: {self.documents_path}")


def main() -> None:
    launcher = Arma3Launcher()
    try:
        launcher.check_paths()
        process = launcher.launch_eden_editor("test_sqf_v3.Altis")
        if process:
            while process.poll() is None:
                time.sleep(1)
            logging.info("Arma 3 Eden Editor has been closed")
    except FileNotFoundError as e:
        logging.error("Error: %s", e)
    except Exception as e:
        logging.error("Unexpected error: %s", e)


if __name__ == "__main__":
    main()
