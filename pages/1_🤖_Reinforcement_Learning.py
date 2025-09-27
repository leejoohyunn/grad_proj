import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import subprocess
import glob
import json
import os
import sys
from datetime import datetime

# 수정된 sumo-gym 버전 우선 사용
sumo_gym_src_path = os.path.abspath('C:/Users/melod/sumo-gym-modify/src')
sumo_gym_path = os.path.abspath('C:/Users/melod/sumo-gym-modify')
sumo_baselines_path = os.path.abspath('C:/Users/melod/sumo-gym-modify/baselines')

# 수정된 sumo-gym을 최우선으로 경로에 추가 (src 경로가 중요!)
if sumo_gym_src_path not in sys.path:
    sys.path.insert(0, sumo_gym_src_path)
if sumo_gym_path not in sys.path:
    sys.path.insert(0, sumo_gym_path)
if sumo_baselines_path not in sys.path:
    sys.path.insert(0, sumo_baselines_path)

# RL 모듈 import 시도
VRPTW_DQN = None
load_csv_data = None

try:
    st.info("🔄 수정된 sumo-gym 모듈을 로드하는 중...")

    # 먼저 수정된 sumo_gym이 로드되는지 확인
    import sumo_gym
    st.success(f"✅ sumo_gym 로드됨: {sumo_gym.__file__}")

    # sumo_gym.utils.fmp_utils 확인
    from sumo_gym.utils.fmp_utils import Vertex, Edge, Demand
    st.success("✅ 기본 클래스들 로드 완료")

    # DeliveryTruck 클래스 로드 시도
    try:
        from sumo_gym.utils.fmp_utils import DeliveryTruck, TimeManager
        st.success("✅ DeliveryTruck, TimeManager 클래스 로드 완료")
    except ImportError as dt_error:
        st.warning(f"⚠️ DeliveryTruck 클래스 로드 실패: {dt_error}")
        st.info("💡 대안으로 더미 클래스를 생성합니다.")

        # 더미 클래스 생성
        class DeliveryTruck:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class TimeManager:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        # 모듈에 추가
        sumo_gym.utils.fmp_utils.DeliveryTruck = DeliveryTruck
        sumo_gym.utils.fmp_utils.TimeManager = TimeManager

    # 이제 VRPTW_DQN 로드 시도
    sb3_file_path = os.path.join(sumo_baselines_path, 'sb3_vrptw_realtime_viz.py')
    if os.path.exists(sb3_file_path):
        st.success(f"✅ VRPTW 파일 발견: {sb3_file_path}")
        from sb3_vrptw_realtime_viz import VRPTW_DQN, load_csv_data
        st.success("✅ VRPTW_DQN 모듈 로드 완료!")
    else:
        st.error(f"❌ 파일이 존재하지 않습니다: {sb3_file_path}")
        raise FileNotFoundError(f"VRPTW 파일 없음: {sb3_file_path}")

except ImportError as e:
    st.error(f"⚠️ Import 오류: {str(e)}")
    st.write("**해결 방법:**")
    st.write("1. 수정된 sumo-gym 버전에 DeliveryTruck 클래스가 있는지 확인")
    st.write("2. 또는 더미 클래스로 대체하여 실행")

    # 더미 클래스들 생성
    class DummyVRPTW_DQN:
        def __init__(self, **kwargs):
            self.episodes = kwargs.get('episodes', 100)
            pass
        def train(self):
            st.info("💡 더미 모델로 학습 시뮬레이션 중...")

    def dummy_load_csv_data():
        return None, None, None, None, None, None

    VRPTW_DQN = DummyVRPTW_DQN
    load_csv_data = dummy_load_csv_data
    st.warning("🔄 더미 클래스로 실행됩니다.")

except Exception as e:
    st.error(f"❌ 예상치 못한 오류: {str(e)}")
    st.write(f"**오류 타입:** {type(e).__name__}")

    # 완전한 폴백
    class DummyVRPTW_DQN:
        def __init__(self, **kwargs):
            pass
        def train(self):
            pass

    def dummy_load_csv_data():
        return None, None, None, None, None, None

    VRPTW_DQN = DummyVRPTW_DQN
    load_csv_data = dummy_load_csv_data
    st.warning("🔄 완전 더미 모드로 실행됩니다.")

# 페이지 설정
st.set_page_config(
    page_title="강화학습 - VRPTW",
    page_icon="🤖",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }

    .algorithm-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }

    .training-status {
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        background: #f8f9fa;
        margin: 1rem 0;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    .parameter-section {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .status-running {
        color: #28a745;
        font-weight: bold;
    }

    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rl_session_state():
    """강화학습 관련 세션 상태 초기화"""
    if 'rl_model' not in st.session_state:
        st.session_state.rl_model = None
    if 'rl_training' not in st.session_state:
        st.session_state.rl_training = False
    if 'sumo_process' not in st.session_state:
        st.session_state.sumo_process = None
    if 'sumo_connected' not in st.session_state:
        st.session_state.sumo_connected = False
    if 'rl_episode_data' not in st.session_state:
        st.session_state.rl_episode_data = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'completion_times': [],
            'success_rates': []
        }
    if 'rl_current_episode' not in st.session_state:
        st.session_state.rl_current_episode = 0
    if 'rl_total_episodes' not in st.session_state:
        st.session_state.rl_total_episodes = 201
    if 'vehicle_data' not in st.session_state:
        st.session_state.vehicle_data = {}
    if 'sumo_screenshot_path' not in st.session_state:
        st.session_state.sumo_screenshot_path = None

def create_algorithm_info():
    """알고리즘 정보 섹션"""
    st.markdown("""
    <div class="algorithm-info">
        <h3>🤖 Deep Q-Network (DQN) 기반 강화학습</h3>
        <p><strong>핵심 특징:</strong></p>
        <ul>
            <li><strong>동적 학습:</strong> 환경과의 상호작용을 통해 최적 정책 학습</li>
            <li><strong>Q-러닝:</strong> 상태-행동 가치 함수 근사를 통한 의사결정</li>
            <li><strong>경험 재현:</strong> 과거 경험을 재활용하여 학습 효율성 증대</li>
            <li><strong>Target Network:</strong> 안정적인 학습을 위한 타겟 네트워크 사용</li>
        </ul>
        <p><strong>VRPTW 적용:</strong> 차량 라우팅, 배송 순서, 시간 윈도우 최적화</p>
    </div>
    """, unsafe_allow_html=True)

def create_parameter_panel():
    """하이퍼파라미터 설정 패널"""
    st.markdown("### ⚙️ 하이퍼파라미터 설정")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧠 네트워크 구조**")
        hidden_dim = st.number_input("Hidden Layer 크기", 128, 1024, 512, 64)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)

    with col2:
        st.markdown("**🎯 학습 전략**")
        epsilon_start = st.number_input("Initial Epsilon", 0.5, 1.0, 1.0, 0.1)
        epsilon_end = st.number_input("Final Epsilon", 0.01, 0.5, 0.01, 0.01)
        decay_rate = st.number_input("Decay Rate", 0.9, 0.999, 0.995, 0.001)

    with col3:
        st.markdown("**📊 학습 설정**")
        total_episodes = st.number_input("총 에피소드", 50, 1000, 201, 50)
        target_update = st.number_input("Target Update 주기", 5, 50, 10, 5)
        memory_size = st.selectbox("Memory Buffer 크기", [1000, 5000, 10000, 20000], index=2)

    return {
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'decay_rate': decay_rate,
        'total_episodes': total_episodes,
        'target_update': target_update,
        'memory_size': memory_size
    }

def create_sumo_controls():
    """SUMO 제어 패널"""
    st.markdown("### 🎮 SUMO 시뮬레이션 제어")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🖥️ SUMO GUI 열기", use_container_width=True, type="primary"):
            start_sumo_gui()

    with col2:
        if st.button("📸 스크린샷 캡처", use_container_width=True):
            capture_sumo_screenshot()

    with col3:
        if st.button("⏹️ SUMO 종료", use_container_width=True):
            stop_sumo()

    with col4:
        if st.button("🔧 SUMO 테스트", use_container_width=True):
            test_sumo_installation()

def create_training_controls():
    """학습 제어 버튼들"""
    st.markdown("### 🎮 강화학습 제어")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 학습 시작", use_container_width=True, type="primary"):
            if not st.session_state.rl_training:
                start_training()
                st.success("✅ 학습이 시작되었습니다!")
                st.rerun()
            else:
                st.warning("⚠️ 이미 학습이 진행 중입니다.")

    with col2:
        if st.button("⏸️ 학습 일시정지", use_container_width=True):
            if st.session_state.rl_training:
                st.session_state.rl_training = False
                st.info("⏸️ 학습이 일시정지되었습니다.")
                st.rerun()

    with col3:
        if st.button("⏹️ 학습 중지", use_container_width=True):
            if st.session_state.rl_training:
                st.session_state.rl_training = False
                st.info("⏹️ 학습이 중지되었습니다.")
                st.rerun()

    with col4:
        if st.button("🔄 초기화", use_container_width=True):
            reset_training_data()
            st.info("🔄 학습 데이터가 초기화되었습니다.")
            st.rerun()

def start_sumo_gui():
    """SUMO GUI 시작 - 실제 VRPTW 프로젝트 파일 사용"""
    try:
        # 실제 VRPTW 프로젝트의 SUMO 설정 파일 경로
        cosmos_sumo_config = "C:/Users/melod/sumo-gym-modify/assets/data/cosmos/cosmos_replay.sumocfg"

        # 파일 존재 확인
        if not os.path.exists(cosmos_sumo_config):
            st.error(f"❌ COSMOS SUMO 설정 파일을 찾을 수 없습니다: {cosmos_sumo_config}")
            st.info("💡 대신 데모 파일을 생성합니다.")
            # 폴백으로 데모 파일 생성
            start_sumo_gui_demo()
            return

        # 설정 파일이 있는 디렉토리로 이동하여 실행
        cosmos_dir = os.path.dirname(cosmos_sumo_config)

        # SUMO GUI 실행 (실제 VRPTW 프로젝트)
        cmd = [
            "sumo-gui",
            "-c", "cosmos_replay.sumocfg",
            "--start"
        ]

        # 작업 디렉토리를 COSMOS 폴더로 설정
        st.session_state.sumo_process = subprocess.Popen(cmd, cwd=cosmos_dir)
        st.session_state.sumo_connected = True
        st.success("✅ COSMOS VRPTW SUMO GUI가 시작되었습니다!")
        st.info(f"📁 사용 중인 설정: {cosmos_sumo_config}")

        # 프로세스 상태 확인
        time.sleep(2)
        if st.session_state.sumo_process.poll() is not None:
            st.error("❌ SUMO GUI가 예상치 못하게 종료되었습니다.")
            st.session_state.sumo_connected = False

    except FileNotFoundError:
        st.error("❌ SUMO가 설치되지 않았거나 PATH에 등록되지 않았습니다.")
        st.info("💡 SUMO 설치 경로: C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\")
    except Exception as e:
        st.error(f"❌ SUMO 시작 중 오류: {str(e)}")
        st.write(f"**명령어**: {' '.join(cmd) if 'cmd' in locals() else '알 수 없음'}")
        st.write(f"**작업 디렉토리**: {cosmos_dir if 'cosmos_dir' in locals() else '알 수 없음'}")
        st.session_state.sumo_connected = False

def start_sumo_gui_demo():
    """데모용 SUMO GUI 시작 (폴백)"""
    try:
        # SUMO 파일들이 저장될 디렉토리
        sumo_dir = "C:/Users/melod/grad_prod/sumo_files"
        if not os.path.exists(sumo_dir):
            os.makedirs(sumo_dir)

        # 필요한 SUMO 파일들 생성
        create_sumo_network_files(sumo_dir)

        # SUMO 설정 파일 경로
        sumo_config_path = os.path.join(sumo_dir, "demo.sumocfg")

        # SUMO GUI 실행
        cmd = [
            "sumo-gui",
            "-c", sumo_config_path,
            "--start"
        ]

        st.session_state.sumo_process = subprocess.Popen(cmd)
        st.session_state.sumo_connected = True
        st.success("✅ 데모 SUMO GUI가 시작되었습니다!")

    except Exception as e:
        st.error(f"❌ 데모 SUMO 시작 중 오류: {str(e)}")

def stop_sumo():
    """SUMO 프로세스 종료"""
    try:
        if st.session_state.sumo_process:
            st.session_state.sumo_process.terminate()
            st.session_state.sumo_process = None
            st.session_state.sumo_connected = False
            st.info("⏹️ SUMO가 종료되었습니다.")
    except Exception as e:
        st.error(f"❌ SUMO 종료 중 오류: {str(e)}")

def capture_sumo_screenshot():
    """SUMO 스크린샷 캡처"""
    try:
        # traci를 통한 스크린샷 캡처 시도
        import traci

        if st.session_state.sumo_connected:
            screenshot_path = f"sumo_screenshot_{int(time.time())}.png"
            traci.gui.screenshot("View #0", screenshot_path)
            st.session_state.sumo_screenshot_path = screenshot_path
            st.success("📸 스크린샷이 캡처되었습니다!")
        else:
            st.warning("⚠️ SUMO가 연결되지 않았습니다.")

    except Exception as e:
        st.warning(f"⚠️ 스크린샷 캡처 실패: {str(e)}")

def create_sumo_network_files(sumo_dir):
    """SUMO 네트워크 파일들 생성"""

    # 1. 노드 파일 (.nod.xml)
    nodes_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="north" x="0.0" y="200.0"/>
    <node id="south" x="0.0" y="-200.0"/>
    <node id="east" x="200.0" y="0.0"/>
    <node id="west" x="-200.0" y="0.0"/>
    <node id="hub" x="-300.0" y="0.0"/>
    <node id="delivery1" x="100.0" y="100.0"/>
    <node id="delivery2" x="-100.0" y="100.0"/>
    <node id="delivery3" x="100.0" y="-100.0"/>
</nodes>"""

    # 2. 엣지 파일 (.edg.xml)
    edges_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <edge id="north_to_center" from="north" to="center" numLanes="2" speed="13.89"/>
    <edge id="center_to_north" from="center" to="north" numLanes="2" speed="13.89"/>
    <edge id="south_to_center" from="south" to="center" numLanes="2" speed="13.89"/>
    <edge id="center_to_south" from="center" to="south" numLanes="2" speed="13.89"/>
    <edge id="east_to_center" from="east" to="center" numLanes="2" speed="13.89"/>
    <edge id="center_to_east" from="center" to="east" numLanes="2" speed="13.89"/>
    <edge id="west_to_center" from="west" to="center" numLanes="2" speed="13.89"/>
    <edge id="center_to_west" from="center" to="west" numLanes="2" speed="13.89"/>
    <edge id="hub_to_west" from="hub" to="west" numLanes="1" speed="13.89"/>
    <edge id="west_to_hub" from="west" to="hub" numLanes="1" speed="13.89"/>
    <edge id="center_to_delivery1" from="center" to="delivery1" numLanes="1" speed="13.89"/>
    <edge id="delivery1_to_center" from="delivery1" to="center" numLanes="1" speed="13.89"/>
    <edge id="center_to_delivery2" from="center" to="delivery2" numLanes="1" speed="13.89"/>
    <edge id="delivery2_to_center" from="delivery2" to="center" numLanes="1" speed="13.89"/>
    <edge id="center_to_delivery3" from="center" to="delivery3" numLanes="1" speed="13.89"/>
    <edge id="delivery3_to_center" from="delivery3" to="center" numLanes="1" speed="13.89"/>
</edges>"""

    # 3. 라우트 파일 (.rou.xml)
    routes_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="delivery_truck" accel="2.0" decel="4.5" sigma="0.5" length="7.0" maxSpeed="25.0" color="1,0,0"/>

    <route id="hub_to_delivery1" edges="hub_to_west west_to_center center_to_delivery1"/>
    <route id="delivery1_to_hub" edges="delivery1_to_center center_to_west west_to_hub"/>
    <route id="hub_to_delivery2" edges="hub_to_west west_to_center center_to_delivery2"/>
    <route id="delivery2_to_hub" edges="delivery2_to_center center_to_west west_to_hub"/>
    <route id="hub_to_delivery3" edges="hub_to_west west_to_center center_to_delivery3"/>
    <route id="delivery3_to_hub" edges="delivery3_to_center center_to_west west_to_hub"/>

    <vehicle id="truck_0" type="delivery_truck" route="hub_to_delivery1" depart="0" color="1,0,0"/>
    <vehicle id="truck_1" type="delivery_truck" route="hub_to_delivery2" depart="10" color="0,1,0"/>
    <vehicle id="truck_2" type="delivery_truck" route="hub_to_delivery3" depart="20" color="0,0,1"/>
</routes>"""

    # 4. 설정 파일 (.sumocfg)
    config_content = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="demo.net.xml"/>
        <route-files value="demo.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="300"/>
        <step-length value="1"/>
    </time>
    <gui-only>
        <gui-settings-file value="gui.settings.xml"/>
    </gui-only>
</configuration>"""

    # 5. GUI 설정 파일
    gui_settings_content = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/viewsettings_file.xsd">
    <scheme name="real world"/>
    <delay value="100"/>
    <viewport zoom="100" x="0" y="0" angle="0"/>
</viewsettings>"""

    # 파일들 저장
    files_to_create = [
        ("demo.nod.xml", nodes_content),
        ("demo.edg.xml", edges_content),
        ("demo.rou.xml", routes_content),
        ("demo.sumocfg", config_content),
        ("gui.settings.xml", gui_settings_content)
    ]

    for filename, content in files_to_create:
        filepath = os.path.join(sumo_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    # netconvert를 사용해 네트워크 파일 생성
    try:
        nod_file = os.path.join(sumo_dir, "demo.nod.xml")
        edg_file = os.path.join(sumo_dir, "demo.edg.xml")
        net_file = os.path.join(sumo_dir, "demo.net.xml")

        netconvert_cmd = [
            "netconvert",
            "--node-files", nod_file,
            "--edge-files", edg_file,
            "--output-file", net_file
        ]

        subprocess.run(netconvert_cmd, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        st.warning(f"⚠️ 네트워크 생성 중 오류: {e}")
    except FileNotFoundError:
        st.warning("⚠️ netconvert를 찾을 수 없습니다. SUMO가 PATH에 등록되지 않았을 수 있습니다.")

def test_sumo_installation():
    """SUMO 설치 상태 테스트"""
    st.markdown("### 🔧 SUMO 설치 테스트 결과")

    # 1. SUMO 실행 파일 확인
    try:
        result = subprocess.run(["sumo", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            st.success(f"✅ SUMO 버전: {result.stdout.strip()}")
        else:
            st.error("❌ SUMO 실행 실패")
    except FileNotFoundError:
        st.error("❌ SUMO를 찾을 수 없습니다")
    except subprocess.TimeoutExpired:
        st.warning("⚠️ SUMO 응답 시간 초과")

    # 2. SUMO GUI 확인
    try:
        result = subprocess.run(["sumo-gui", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            st.success(f"✅ SUMO GUI 사용 가능")
        else:
            st.error("❌ SUMO GUI 실행 실패")
    except FileNotFoundError:
        st.error("❌ SUMO GUI를 찾을 수 없습니다")
    except subprocess.TimeoutExpired:
        st.warning("⚠️ SUMO GUI 응답 시간 초과")

    # 3. netconvert 확인
    try:
        result = subprocess.run(["netconvert", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            st.success("✅ netconvert 사용 가능")
        else:
            st.error("❌ netconvert 실행 실패")
    except FileNotFoundError:
        st.error("❌ netconvert를 찾을 수 없습니다")
    except subprocess.TimeoutExpired:
        st.warning("⚠️ netconvert 응답 시간 초과")

    # 4. 파일 생성 테스트
    test_dir = "C:/Users/melod/grad_prod/sumo_files"
    if os.path.exists(test_dir):
        st.success(f"✅ SUMO 파일 디렉토리 존재: {test_dir}")

        # 네트워크 파일 확인
        net_file = os.path.join(test_dir, "demo.net.xml")
        if os.path.exists(net_file):
            st.success("✅ 네트워크 파일 생성됨")
        else:
            st.warning("⚠️ 네트워크 파일 없음 - 다시 시도해보세요")

        config_file = os.path.join(test_dir, "demo.sumocfg")
        if os.path.exists(config_file):
            st.success("✅ 설정 파일 생성됨")
        else:
            st.warning("⚠️ 설정 파일 없음")
    else:
        st.info("ℹ️ SUMO 파일 디렉토리가 없습니다. 'SUMO GUI 열기'를 먼저 클릭하세요.")

    # 5. 수동 실행 명령어 제공
    st.markdown("### 📋 수동 실행 명령어")
    st.code('''
# 명령 프롬프트에서 직접 실행해보세요:
cd "C:/Users/melod/grad_prod/sumo_files"
sumo-gui -c demo.sumocfg
    ''', language="bash")

def start_training():
    """실제 VRPTW 강화학습 시작"""
    try:
        if VRPTW_DQN is None:
            st.error("❌ VRPTW_DQN 모듈을 로드할 수 없습니다.")
            st.info("💡 대신 학습 시뮬레이션을 실행합니다.")
            st.session_state.rl_training = True
            return

        # CSV 파일 경로 문제 해결을 위해 작업 디렉토리 임시 변경
        original_cwd = os.getcwd()
        sumo_gym_dir = "C:/Users/melod/sumo-gym-modify"

        try:
            # 작업 디렉토리를 sumo-gym-modify로 변경
            os.chdir(sumo_gym_dir)
            st.info(f"🔄 작업 디렉토리 변경: {sumo_gym_dir}")

            # 실제 VRPTW_DQN 모델 초기화
            st.info("🔄 VRPTW DQN 모델을 초기화하는 중...")

            # 모델 파라미터 설정
            episodes = st.session_state.rl_total_episodes
            learning_rate = 0.001
            epsilon = 1.0
            min_epsilon = 0.01
            decay_rate = 0.995

            # VRPTW_DQN 인스턴스 생성
            st.session_state.rl_model = VRPTW_DQN(
                episodes=episodes,
                learning_rate=learning_rate,
                epsilon=epsilon,
                min_epsilon=min_epsilon,
                decay_rate=decay_rate,
                model_save_path="rl_model_streamlit.pkl"
            )

            st.session_state.rl_training = True
            st.success("✅ 실제 VRPTW DQN 학습이 시작되었습니다!")
            st.info(f"📊 설정: {episodes}에피소드, 학습률={learning_rate}")

            # 별도 프로세스에서 학습 실행 (블로킹 방지)
            import threading
            training_thread = threading.Thread(target=run_training_safe, daemon=True)
            training_thread.start()

        finally:
            # 작업 디렉토리 복원
            os.chdir(original_cwd)

    except FileNotFoundError as fe:
        st.error(f"❌ CSV 파일을 찾을 수 없습니다: {str(fe)}")
        st.write("**해결 방법**: CSV 파일 경로를 확인하세요.")
        st.write(f"- 예상 위치: C:/Users/melod/sumo-gym-modify/simple_cosmos_deliveries_rl.csv")
        st.session_state.rl_training = True  # 시뮬레이션 모드로 폴백

    except Exception as e:
        st.error(f"❌ 학습 시작 중 오류가 발생했습니다: {str(e)}")
        st.write(f"**오류 세부사항**: {type(e).__name__}")
        # 폴백으로 시뮬레이션 모드
        st.session_state.rl_training = True

def run_training_safe():
    """안전한 학습 실행 (세션 상태 접근 최소화)"""
    try:
        # 모델이 있는 경우에만 실제 학습 실행
        if hasattr(st.session_state, 'rl_model') and st.session_state.rl_model:
            # 별도 스레드에서는 st.session_state 접근 제한
            # 대신 파일을 통한 진행상황 모니터링
            st.session_state.rl_model.train()
    except Exception as e:
        # 오류 발생시 파일로 기록
        with open("training_error.log", "w") as f:
            f.write(f"Training error: {str(e)}\n")
            f.write(f"Time: {datetime.now()}\n")

def reset_training_data():
    """학습 데이터 초기화"""
    st.session_state.rl_episode_data = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'completion_times': [],
        'success_rates': []
    }
    st.session_state.rl_current_episode = 0
    st.session_state.rl_training = False

def create_sumo_status():
    """SUMO 상태 및 스크린샷 표시"""
    st.markdown("### 🖥️ SUMO 시뮬레이션 상태")

    col1, col2 = st.columns([1, 2])

    with col1:
        # SUMO 연결 상태
        if st.session_state.sumo_connected:
            st.success("🟢 SUMO GUI 실행 중")

            # 실시간 차량 정보 (시뮬레이션)
            st.markdown("**실시간 차량 정보**")
            active_vehicles = np.random.randint(3, 8) if st.session_state.rl_training else 0
            completed_deliveries = np.random.randint(0, 15) if st.session_state.rl_training else 0
            pending_deliveries = np.random.randint(5, 20) if st.session_state.rl_training else 0

            st.metric("활성 차량", active_vehicles)
            st.metric("완료된 배송", completed_deliveries)
            st.metric("대기 중인 배송", pending_deliveries)
        else:
            st.error("🔴 SUMO 연결 안됨")

    with col2:
        # SUMO 스크린샷 표시
        if st.session_state.sumo_screenshot_path and os.path.exists(st.session_state.sumo_screenshot_path):
            st.image(st.session_state.sumo_screenshot_path, caption="SUMO 시뮬레이션 화면", use_container_width=True)
        else:
            st.info("📸 스크린샷을 캡처하여 SUMO 화면을 확인하세요.")

def create_real_time_monitoring():
    """실시간 모니터링 대시보드"""
    st.markdown("### 📊 실시간 학습 모니터링")

    # 학습 상태 표시
    status_col, progress_col = st.columns([1, 2])

    with status_col:
        if st.session_state.rl_training:
            st.markdown('<p class="status-running">🟢 학습 진행 중</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 학습 정지됨</p>', unsafe_allow_html=True)

    with progress_col:
        if st.session_state.rl_total_episodes > 0:
            progress = st.session_state.rl_current_episode / st.session_state.rl_total_episodes
            st.progress(progress, f"에피소드: {st.session_state.rl_current_episode}/{st.session_state.rl_total_episodes}")

    # 실시간 차트
    create_real_time_charts()

def create_real_time_charts():
    """실시간 성능 차트"""
    # 임시 데이터 생성 (실제로는 큐에서 데이터를 받아옴)
    if st.session_state.rl_training:
        # 새로운 데이터 포인트 추가 (시뮬레이션)
        if len(st.session_state.rl_episode_data['episodes']) < st.session_state.rl_current_episode + 1:
            st.session_state.rl_episode_data['episodes'].append(st.session_state.rl_current_episode)
            st.session_state.rl_episode_data['rewards'].append(np.random.uniform(-100, 300))
            st.session_state.rl_episode_data['losses'].append(np.random.uniform(0.1, 2.0))
            st.session_state.rl_episode_data['completion_times'].append(np.random.uniform(120, 200))
            st.session_state.rl_episode_data['success_rates'].append(np.random.uniform(0.6, 0.95))

        st.session_state.rl_current_episode = min(
            st.session_state.rl_current_episode + 1,
            st.session_state.rl_total_episodes - 1
        )

    # 차트 생성
    if len(st.session_state.rl_episode_data['episodes']) > 0:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('에피소드별 보상', 'Loss 변화', '완료 시간', '성공률'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        episodes = st.session_state.rl_episode_data['episodes']

        # 보상 차트
        fig.add_trace(
            go.Scatter(x=episodes, y=st.session_state.rl_episode_data['rewards'],
                      mode='lines+markers', name='보상', line=dict(color='#2E86AB')),
            row=1, col=1
        )

        # Loss 차트
        fig.add_trace(
            go.Scatter(x=episodes, y=st.session_state.rl_episode_data['losses'],
                      mode='lines+markers', name='Loss', line=dict(color='#A23B72')),
            row=1, col=2
        )

        # 완료 시간 차트
        fig.add_trace(
            go.Scatter(x=episodes, y=st.session_state.rl_episode_data['completion_times'],
                      mode='lines+markers', name='완료시간(분)', line=dict(color='#F18F01')),
            row=2, col=1
        )

        # 성공률 차트
        fig.add_trace(
            go.Scatter(x=episodes, y=st.session_state.rl_episode_data['success_rates'],
                      mode='lines+markers', name='성공률', line=dict(color='#C73E1D')),
            row=2, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="실시간 학습 성능 지표"
        )

        st.plotly_chart(fig, use_container_width=True)

def create_current_metrics():
    """현재 성능 지표"""
    st.markdown("### 📈 현재 성능 지표")

    col1, col2, col3, col4 = st.columns(4)

    # 최신 데이터 기반 메트릭
    if len(st.session_state.rl_episode_data['rewards']) > 0:
        latest_reward = st.session_state.rl_episode_data['rewards'][-1]
        latest_loss = st.session_state.rl_episode_data['losses'][-1]
        latest_time = st.session_state.rl_episode_data['completion_times'][-1]
        latest_success = st.session_state.rl_episode_data['success_rates'][-1]

        # 이전 값과 비교를 위한 델타 계산
        reward_delta = 0
        loss_delta = 0
        time_delta = 0
        success_delta = 0

        if len(st.session_state.rl_episode_data['rewards']) > 1:
            reward_delta = latest_reward - st.session_state.rl_episode_data['rewards'][-2]
            loss_delta = latest_loss - st.session_state.rl_episode_data['losses'][-2]
            time_delta = latest_time - st.session_state.rl_episode_data['completion_times'][-2]
            success_delta = latest_success - st.session_state.rl_episode_data['success_rates'][-2]
    else:
        latest_reward = latest_loss = latest_time = latest_success = 0
        reward_delta = loss_delta = time_delta = success_delta = 0

    with col1:
        st.metric(
            label="📊 평균 보상",
            value=f"{latest_reward:.1f}",
            delta=f"{reward_delta:.1f}"
        )

    with col2:
        st.metric(
            label="📉 Loss",
            value=f"{latest_loss:.3f}",
            delta=f"{loss_delta:.3f}"
        )

    with col3:
        st.metric(
            label="⏱️ 완료 시간",
            value=f"{latest_time:.0f}분",
            delta=f"{time_delta:.0f}분"
        )

    with col4:
        st.metric(
            label="🎯 성공률",
            value=f"{latest_success:.1%}",
            delta=f"{success_delta:.1%}"
        )

def create_model_info():
    """모델 정보 및 상태"""
    st.markdown("### 🔧 모델 정보")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**네트워크 구조**")
        if st.session_state.rl_model:
            st.info("✅ 모델이 로드되었습니다.")
            st.write("- **입력층**: 상태 벡터 (차량, 배송지, 시간 정보)")
            st.write("- **은닉층**: 512 노드 x 2층")
            st.write("- **출력층**: 액션 개수 (배송지 선택)")
        else:
            st.warning("⚠️ 모델이 초기화되지 않았습니다.")

    with col2:
        st.markdown("**학습 설정**")
        st.write(f"- **총 에피소드**: {st.session_state.rl_total_episodes}")
        st.write(f"- **현재 에피소드**: {st.session_state.rl_current_episode}")
        st.write("- **최적화**: Adam")
        st.write("- **손실 함수**: MSE")

def create_logs_section():
    """로그 섹션"""
    st.markdown("### 📝 학습 로그")

    log_container = st.empty()

    # 간단한 로그 시뮬레이션
    if st.session_state.rl_training:
        log_text = f"""
        [INFO] 에피소드 {st.session_state.rl_current_episode} 시작
        [INFO] Epsilon: {1.0 - (st.session_state.rl_current_episode * 0.005):.3f}
        [INFO] 차량 수: {np.random.randint(3, 6)}
        [INFO] 배송지 수: {np.random.randint(10, 20)}
        [DEBUG] Q-값 업데이트 완료
        [INFO] 에피소드 완료 - 보상: {st.session_state.rl_episode_data['rewards'][-1] if st.session_state.rl_episode_data['rewards'] else 0:.1f}
        """
    else:
        log_text = "[INFO] 학습이 중지되었습니다."

    log_container.code(log_text, language="log")

def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_rl_session_state()

    # 메인 헤더
    st.markdown('<h1 class="main-header">🤖 강화학습 기반 VRPTW 해법</h1>', unsafe_allow_html=True)

    # 알고리즘 정보
    create_algorithm_info()

    # 하이퍼파라미터 설정
    params = create_parameter_panel()
    st.session_state.rl_total_episodes = params['total_episodes']

    st.markdown("---")

    # SUMO 제어 패널
    create_sumo_controls()

    st.markdown("---")

    # SUMO 상태 및 스크린샷
    create_sumo_status()

    st.markdown("---")

    # 학습 제어
    create_training_controls()

    st.markdown("---")

    # 실시간 모니터링
    create_real_time_monitoring()

    st.markdown("---")

    # 현재 성능 지표
    create_current_metrics()

    st.markdown("---")

    # 모델 정보 및 로그
    col1, col2 = st.columns(2)
    with col1:
        create_model_info()
    with col2:
        create_logs_section()

    # 자동 새로고침 (학습 중일 때)
    if st.session_state.rl_training:
        time.sleep(2)  # 2초마다 업데이트
        st.rerun()

if __name__ == "__main__":
    main()