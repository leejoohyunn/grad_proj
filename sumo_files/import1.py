import os
import sys  # sys 모듈 추가
import numpy as np
import json
import gymnasium as gym
import random
import pandas as pd
import math
import pickle  # 모델 저장/로딩용
import traci  # SUMO 직접 제어용
import subprocess  # SUMO 실행용

import sumo_gym
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    DeliveryTruck,
    TimeManager,
    generate_random_time_windows,
    get_available_deliveries,
)

# SUMO 도로 기반 경로 계산 모듈 추가
sys.path.append('..')
from sumo_route_calculator_stable import get_stable_route_calculator, calculate_stable_road_distance, calculate_stable_road_travel_time
from generate_sumo_delivery_coords import main as generate_new_coordinates
import subprocess
from DQN.dqn import (
    QNetwork,
    ReplayBuffer,
    run_target_update,
)
from statistics import mean
import sys
import json
import time

suffix = "-vrptw.json"

from sumo_gym.envs.fmp import VRPTWEnv

def load_csv_data():
    """CSV 파일들을 읽어서 VRPTW numerical 모드용 데이터 구조로 변환"""
    # CSV 파일 읽기
    deliveries_df = pd.read_csv("../simple_cosmos_deliveries_rl.csv")
    hub_df = pd.read_csv("../simple_cosmos_hub.csv")
    trucks_df = pd.read_csv("../simple_cosmos_trucks_rl.csv")

    # 1. Vertices 생성 (좌표 기반)
    vertices, vertex_mapping = create_vertices_from_csv(deliveries_df, hub_df)

    # 실제 고유 좌표 수 확인 (KMeans 클러스터링을 위해)
    unique_coords = set()
    for vertex in vertices:
        unique_coords.add((vertex.x, vertex.y))

    # 안전한 클러스터 수 계산: min(CSV구역수, 실제고유좌표수)
    csv_areas = deliveries_df['area'].nunique()
    actual_unique_coords = len(unique_coords)
    n_areas = min(csv_areas, actual_unique_coords)

    print(f"Areas found in CSV: {deliveries_df['area'].unique()}")
    print(f"CSV area count: {csv_areas}")
    print(f"Unique coordinates: {actual_unique_coords}")
    print(f"Clusters to use: {n_areas}")

    # 2. Demands 생성
    demands = create_demands_from_csv(deliveries_df, vertex_mapping)

    # 3. Edges 생성 (완전 그래프)
    edges = create_edges_from_vertices(vertices)

    # 4. DeliveryTrucks 생성
    hub_vertex_idx = 0  # 허브는 첫 번째 vertex
    delivery_trucks = create_delivery_trucks_from_csv(trucks_df, hub_vertex_idx)

    # 5. TimeManager 생성
    time_manager = TimeManager(max_episode_time=180)  # 3시간

    return vertices, demands, edges, delivery_trucks, time_manager, n_areas

def create_vertices_from_csv(deliveries_df, hub_df):
    """좌표 데이터에서 Vertex 객체들과 좌표→인덱스 매핑 생성"""
    # 허브 좌표 추가 (첫 번째로)
    hub_x = hub_df.iloc[0]['x_coordinate']
    hub_y = hub_df.iloc[0]['y_coordinate']

    # 모든 고유 좌표 수집 (허브 + 배송지)
    all_coords = [(hub_x, hub_y)]
    coord_to_idx = {(hub_x, hub_y): 0}  # 허브는 인덱스 0

    # 배송지 좌표들 추가
    for _, row in deliveries_df.iterrows():
        coord = (row['x_coordinate'], row['y_coordinate'])
        if coord not in coord_to_idx:
            idx = len(all_coords)
            all_coords.append(coord)
            coord_to_idx[coord] = idx

    # Vertex 객체들 생성
    vertices = [Vertex(x, y) for x, y in all_coords]

    return vertices, coord_to_idx

def create_demands_from_csv(deliveries_df, vertex_mapping):
    """배송 데이터에서 Demand 객체들 생성"""
    demands = []

    for _, row in deliveries_df.iterrows():
        # 좌표를 vertex 인덱스로 변환
        delivery_coord = (row['x_coordinate'], row['y_coordinate'])
        destination_idx = vertex_mapping[delivery_coord]

        # Demand 객체 생성
        demand = Demand(
            departure=0,  # 모든 배송이 허브(인덱스 0)에서 출발
            destination=destination_idx,
            earliest_time=int(row['earliest_time_minutes']),
            latest_time=int(row['latest_time_minutes']),
            delivery_id=row['delivery_id']
        )
        demands.append(demand)

    return demands

def create_edges_from_vertices(vertices):
    """모든 vertex 간 Edge 생성 (완전 그래프)"""
    edges = []
    n_vertices = len(vertices)

    # 모든 vertex 쌍에 대해 양방향 엣지 생성
    for i in range(n_vertices):
        for j in range(n_vertices):
            if i != j:  # 자기 자신으로의 엣지는 제외
                edges.append(Edge(i, j))

    return edges

def create_delivery_trucks_from_csv(trucks_df, hub_vertex_idx):
    """트럭 데이터에서 DeliveryTruck 객체들 생성"""
    delivery_trucks = []

    for _, row in trucks_df.iterrows():
        truck = DeliveryTruck(
            id=row['truck_id'],
            speed=float(row['average_speed_kmh']),  # km/h
            indicator=0,  # 초기 상태
            capacity=int(row['max_cargo_items']),
            location=hub_vertex_idx,  # 허브에서 시작
            cargo_count=int(row['current_cargo_count']),
            status=0,  # 0: hub waiting
            current_time=0,
            start_time=0
        )
        delivery_trucks.append(truck)

    return delivery_trucks

# CSV 데이터 로딩
vertices, demands, edges, delivery_trucks, time_manager, n_areas = load_csv_data()

# VRPTW 환경 설정 (numerical 모드)
env = VRPTWEnv(
    mode="numerical",
    n_vertex=len(vertices),
    n_area=n_areas,  # CSV에서 동적으로 계산된 실제 구역 수
    n_demand=len(demands),
    n_edge=len(edges),
    n_vehicle=len(delivery_trucks),
    n_delivery_truck=len(delivery_trucks),
    hub_location=0,  # 허브는 첫 번째 vertex
    vertices=vertices,
    demands=demands,
    edges=edges,
    delivery_trucks=delivery_trucks,
    departures=[0] * len(delivery_trucks),  # 모든 트럭이 허브에서 출발
    time_manager=time_manager,
    verbose=1,
)

# 데이터 로딩 검증 (디버깅용)
print(f"Vertices: {len(vertices)}")
print(f"Demands: {len(demands)}")
print(f"Edges: {len(edges)}")
print(f"Delivery trucks: {len(delivery_trucks)}")


class VRPTW_DQN(object):
    def __init__(
        self,
        env,
        lr=0.003,
        batch_size=8,
        tau=50,
        episodes=201,  # 에피소드 200까지 훈련 (0~200)
        gamma=0.95,
        epsilon=1.0,
        decay_period=25,
        decay_rate=0.95,
        min_epsilon=0.01,
        initial_step=100,
        coordinate_update_interval=100,  # 100 에피소드마다 좌표 갱신
        test_episodes=1,  # 테스트 에피소드 수 (실시간 시각화용)
        model_save_path="trained_model.pkl",  # 모델 저장 경로
    ):
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.initial_step = initial_step
        self.coordinate_update_interval = coordinate_update_interval
        self.test_episodes = test_episodes
        self.model_save_path = model_save_path

        # SUMO 경로 계산기 초기화
        self.route_calculator = get_stable_route_calculator()

        # 차량 경로 추적 시스템
        self.vehicle_routes = {agent: [] for agent in self.env.possible_agents}
        self.episode_routes = {}  # 에피소드별 경로 저장

        # CSV에서 로딩한 실제 데이터 크기 반영
        n_demands = len(env.vrptw.demands)  # CSV에서 로딩한 실제 배송 요청 수
        max_demands = 50  # 패딩을 위한 최대값 (기존 코드와 호환성 유지)

        # 상태와 액션 크기는 기존과 동일하게 유지 (패딩으로 처리)
        state_size = 3 + max_demands + (max_demands * 2)  # 153
        action_size = 1 + max_demands + 1  # 허브(1) + 배송지(50) + 대기(1) = 52

        print(f"실제 배송 요청 수: {n_demands}, 최대 지원: {max_demands}")

        self.q_principal = QNetwork(state_size, action_size, self.lr)
        self.q_target = QNetwork(state_size, action_size, self.lr)

        # 각 에이전트별 리플레이 버퍼 (경험은 개별 저장)
        self.replay_buffers = {
            agent: ReplayBuffer() for agent in self.env.possible_agents
        }

        self.total_steps = {agent: 0 for agent in self.env.possible_agents}
        self.time_manager = TimeManager(max_episode_time=180)  # 3시간

    def _initialize_output_files(self):
        """결과 저장용 파일 초기화"""
        files = ["reward", "loss", "metrics", "delivery_success", "test_results"]

        for file_name in files:
            full_name = file_name + suffix
            if os.path.exists(full_name):
                os.remove(full_name)
            with open(full_name, "w") as f:
                f.write("{")

    def _wrap_up_output_files(self):
        """결과 파일 마무리"""
        files = ["reward", "loss", "metrics", "delivery_success", "test_results"]

        for file_name in files:
            full_name = file_name + suffix
            with open(full_name, "a") as f:
                f.write("}")

    def _update_coordinates_if_needed(self, episode):
        """100 에피소드마다 새로운 SUMO 좌표로 갱신"""
        if episode % self.coordinate_update_interval == 0 and episode > 0:
            print(f"\n=== 에피소드 {episode}: 새로운 배송지 좌표 생성 중 ===")
            try:
                # 새로운 좌표 생성
                generate_new_coordinates()

                # 새로운 데이터로 환경 재로딩
                global vertices, demands, edges, delivery_trucks, time_manager
                vertices, demands, edges, delivery_trucks, time_manager, n_areas = load_csv_data()

                # 환경 업데이트
                self.env.vrptw.vertices = vertices
                self.env.vrptw.demands = demands
                self.env.vrptw.edges = edges

                print(f"배송지 좌표 갱신 완료: {len(demands)}개 배송지")

            except Exception as e:
                print(f"좌표 갱신 실패: {e}")

    def _calculate_road_based_distance(self, from_vertex_idx, to_vertex_idx):
        """SUMO 도로 기반 실제 거리 계산 (TCP 오류 복구 포함)"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                from_vertex = self.env.vrptw.vertices[from_vertex_idx]
                to_vertex = self.env.vrptw.vertices[to_vertex_idx]

                distance, route = calculate_stable_road_distance(
                    from_vertex.x, from_vertex.y,
                    to_vertex.x, to_vertex.y
                )

                return distance, route

            except Exception as e:
                error_msg = str(e).lower()
                print(f"도로 거리 계산 실패 (시도 {attempt + 1}/{max_retries}): {e}")

                # TCP 연결 오류나 SUMO 종료 오류인 경우
                if any(keyword in error_msg for keyword in ['socket', 'peer shutdown', 'connection', 'traci']):
                    print("SUMO 연결 오류 감지, 재시도 중...")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)  # 2초 대기 후 재시도
                        continue

                # 마지막 시도이거나 다른 종류의 오류인 경우
                if attempt == max_retries - 1:
                    raise Exception(f"도로 기반 거리 계산 필수 - 완전 실패 ({max_retries}회 시도): {e}")

        # 여기까지 오면 안됨
        raise Exception("알 수 없는 오류로 도로 기반 계산 실패")

    def _record_vehicle_movement(self, agent, from_location, to_location, action, timestamp, route_edges):
        """차량 이동 경로 기록"""
        agent_idx = self.env.agent_name_idx_mapping[agent]
        truck = self.env.vrptw.delivery_trucks[agent_idx]

        movement_record = {
            'timestamp': timestamp,
            'agent': agent,
            'action': action,
            'from_location': from_location,
            'to_location': to_location,
            'cargo_count': truck.cargo_count,
            'route_edges': route_edges,
            'speed': truck.speed
        }

        self.vehicle_routes[agent].append(movement_record)

    def _get_state_vector(self, agent):
        """현재 상태를 벡터로 변환"""
        agent_idx = self.env.agent_name_idx_mapping[agent]
        truck = self.env.vrptw.delivery_trucks[agent_idx]

        # 기본 상태 정보 (명시적 float 변환)
        state = [
            float(truck.current_time / 180.0),  # 정규화된 현재 시간
            float(truck.location / 50.0),       # 정규화된 위치
            float(truck.cargo_count / 5.0),     # 정규화된 적재량
        ]

        # 실제 demands 수 확인
        max_demands = 50
        num_actual_demands = len(self.env.vrptw.demands)

        # 배송 완료 상태 (50개로 패딩)
        for i in range(max_demands):
            if i < num_actual_demands:
                state.append(1.0 if self.env.vrptw.demands[i].is_completed else 0.0)
            else:
                state.append(0.0)  # 패딩

        # 시간창 정보 (50개 × 2 = 100차원으로 패딩)
        for i in range(max_demands):
            if i < num_actual_demands:
                demand = self.env.vrptw.demands[i]
                state.extend([
                    float(demand.earliest_time / 180.0),  # 정규화된 시작 시간
                    float(demand.latest_time / 180.0),    # 정규화된 종료 시간
                ])
            else:
                state.extend([0.0, 0.0])  # 패딩

        # 차원 검증
        assert len(state) == 153, f"State dimension mismatch: expected 153, got {len(state)}"
        return state

    def _calculate_reward(self, agent, action, delivery_completed=False, arrival_time=None):
        """VRPTW 보상 계산"""
        reward = 0

        if action == 0:  # 허브 복귀
            reward += 20

        elif 1 <= action <= 50:  # 배송지 이동
            demand_idx = action - 1
            # 액션 범위 검증 추가
            if demand_idx >= len(self.env.vrptw.demands):
                return -10  # 잘못된 액션 페널티
            demand = self.env.vrptw.demands[demand_idx]

            if delivery_completed and arrival_time is not None:
                # 배송 완료 보상
                reward += self.time_manager.calculate_delivery_reward(arrival_time, demand)

                # 시간창 위반 페널티
                if arrival_time > demand.latest_time:
                    penalty = self.time_manager.calculate_time_violation_penalty(arrival_time, demand)
                    reward -= penalty
            else:
                # 이동만 한 경우 이동 비용
                reward -= 1

        elif action == 51:  # 대기
            reward -= 0.5

        return reward

    def _generate_action(self, agent, state):
        """액션 생성 (epsilon-greedy)"""
        if np.random.rand() < self.epsilon:
            # 탐험: 랜덤 액션
            available_actions = self._get_available_actions(agent)
            return random.choice(available_actions) if available_actions else 51  # 대기
        else:
            # 활용: Q-네트워크 기반 액션
            q_values = self.q_principal.compute_q_values(state)
            available_actions = self._get_available_actions(agent)

            # 가능한 액션 중에서 최고 Q값 선택
            if available_actions:
                available_q_values = [(action, q_values[action]) for action in available_actions]
                return max(available_q_values, key=lambda x: x[1])[0]
            else:
                return 51  # 대기

    def _get_available_actions(self, agent):
        """현재 상태에서 가능한 액션 목록"""
        agent_idx = self.env.agent_name_idx_mapping[agent]
        truck = self.env.vrptw.delivery_trucks[agent_idx]
        available_actions = []

        # 허브 복귀 (항상 가능)
        available_actions.append(0)

        # 배송지 이동 (적재량이 있고 완료되지 않은 배송지)
        if truck.cargo_count > 0:
            for i, demand in enumerate(self.env.vrptw.demands):
                if not demand.is_completed:
                    # SUMO 도로 기반 시간 제약 확인
                    distance, route = self._calculate_road_based_distance(truck.location, demand.destination)
                    travel_time = distance / (truck.speed * 1000 / 60)  # 분 단위 변환
                    arrival_time = truck.current_time + travel_time

                    if arrival_time <= self.time_manager.max_episode_time:
                        available_actions.append(i + 1)  # 배송지 액션

        # 대기 (항상 가능)
        available_actions.append(51)

        return available_actions

    def _update_network(self, agent, state, action, next_state, reward, done):
        """Q-네트워크 업데이트"""
        # 리플레이 버퍼에 경험 저장
        self.replay_buffers[agent].push([state, action, next_state, reward])

        # 네트워크 업데이트 조건 확인
        if (self.total_steps[agent] % 10 == 0 and
            self.total_steps[agent] > self.initial_step and
            len(self.replay_buffers[agent]) >= self.batch_size):

            # 배치 샘플링
            samples = self.replay_buffers[agent].sample(self.batch_size)
            states, actions, next_states, rewards = zip(*samples)

            # Q값 계산 (리스트 형태로 직접 전달)
            # 상태 차원 검증
            state_dims = [len(s) for s in states]
            next_state_dims = [len(s) for s in next_states]

            if len(set(state_dims)) > 1 or len(set(next_state_dims)) > 1:
                print(f"Error: Inconsistent state dimensions")
                print(f"State dims: {state_dims}")
                print(f"Next state dims: {next_state_dims}")
                return 0

            # numpy 배열 변환 없이 리스트 직접 사용
            targets = rewards + self.gamma * self.q_target.compute_max_q(next_states)
            loss = self.q_principal.train(states, actions, targets)

            # 타겟 네트워크 업데이트
            if self.total_steps[agent] % self.tau == 0:
                run_target_update(self.q_principal, self.q_target)

            return loss

        return 0

    def save_model(self):
        """훈련된 모델 저장"""
        try:
            model_data = {
                'q_principal_weights': self.q_principal.state_dict() if hasattr(self.q_principal, 'state_dict') else None,
                'q_target_weights': self.q_target.state_dict() if hasattr(self.q_target, 'state_dict') else None,
                'hyperparameters': {
                    'lr': self.lr,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon,
                    'min_epsilon': self.min_epsilon,
                }
            }

            with open(self.model_save_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"모델 저장 완료: {self.model_save_path}")

        except Exception as e:
            print(f"모델 저장 실패: {e}")

    def load_model(self):
        """저장된 모델 로딩"""
        try:
            if os.path.exists(self.model_save_path):
                with open(self.model_save_path, 'rb') as f:
                    model_data = pickle.load(f)

                if hasattr(self.q_principal, 'load_state_dict') and model_data['q_principal_weights']:
                    self.q_principal.load_state_dict(model_data['q_principal_weights'])
                if hasattr(self.q_target, 'load_state_dict') and model_data['q_target_weights']:
                    self.q_target.load_state_dict(model_data['q_target_weights'])

                print(f"모델 로딩 완료: {self.model_save_path}")
                return True
            else:
                print(f"저장된 모델 없음: {self.model_save_path}")
                return False

        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            return False

    def _start_sumo_gui(self):
        """SUMO GUI 시작"""
        try:
            # 절대 경로로 SUMO 설정 파일 찾기
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sumo_config = os.path.join(base_dir, "assets", "data", "cosmos", "cosmos_replay.sumocfg")

            # 설정 파일 존재 확인
            if not os.path.exists(sumo_config):
                print(f"SUMO 설정 파일을 찾을 수 없습니다: {sumo_config}")
                return False

            print(f"SUMO 설정 파일 확인: {sumo_config}")

            # SUMO 바이너리 경로 찾기
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
            if not os.path.exists(sumo_binary):
                sumo_binary += '.exe'
            if not os.path.exists(sumo_binary):
                sumo_binary = 'sumo-gui'

            sumo_cmd = [
                sumo_binary,
                '-c', sumo_config,
                '--delay', '1000',  # 1초 지연
                '--step-length', '1.0'  # 1초 단위 시뮬레이션
            ]

            traci.start(sumo_cmd)
            print("SUMO GUI 시작 완료")

            # 차량 타입 정의
            try:
                if "delivery_truck" not in traci.vehicletype.getIDList():
                    traci.vehicletype.copy("DEFAULT_VEHTYPE", "delivery_truck")
                    traci.vehicletype.setColor("delivery_truck", (255, 0, 0))
                    traci.vehicletype.setLength("delivery_truck", 15.0)
                    traci.vehicletype.setWidth("delivery_truck", 4.0)
                    traci.vehicletype.setHeight("delivery_truck", 4.0)
                    print("차량 타입 'delivery_truck' 생성 완료")
            except Exception as e:
                print(f"차량 타입 생성 실패: {e}")

            return True

        except Exception as e:
            print(f"SUMO GUI 시작 실패: {e}")
            return False

    def _add_vehicle_to_sumo(self, agent, start_edge):
        """SUMO에 차량 추가"""
        try:
            route_id = f"route_{agent}"

            # 경로가 이미 존재하지 않으면 생성
            if route_id not in traci.route.getIDList():
                traci.route.add(route_id, [start_edge])

            # 차량이 이미 존재하지 않으면 추가
            if agent not in traci.vehicle.getIDList():
                traci.vehicle.add(agent, route_id, typeID="delivery_truck")
                # Vehicle visibility settings                traci.vehicle.setColor(agent, (255, 0, 0))                traci.vehicle.setLength(agent, 15.0)                traci.vehicle.setWidth(agent, 4.0)                traci.vehicle.setHeight(agent, 4.0)                traci.vehicle.moveTo(agent, start_edge + "_0", 10.0)                traci.vehicle.setSpeed(agent, 0.0)
                print(f"Vehicle {agent} added to SUMO successfully")

            return True
        except Exception as e:
            print(f"Vehicle {agent} SUMO addition failed: {e}")
            return False

    def _update_vehicle_route_in_sumo(self, agent, route_edges):
        """SUMO에서 차량 경로 업데이트"""
        try:
            if agent in traci.vehicle.getIDList() and route_edges:
                # 유효한 경로만 필터링
                valid_edges = []
                all_edges = traci.edge.getIDList()
                for edge in route_edges:
                    if edge in all_edges:
                        valid_edges.append(edge)

                if valid_edges:
                    new_route_id = f"route_{agent}_{len(valid_edges)}"
                    traci.route.add(new_route_id, valid_edges)
                    traci.vehicle.setRoute(agent, new_route_id)
                    print(f"차량 {agent} 경로 업데이트: {len(valid_edges)} edges")
                    return True
            return False
        except Exception as e:
            print(f"차량 {agent} 경로 업데이트 실패: {e}")
            return False

    def _add_poi_to_sumo(self, poi_id, edge_id, x, y, color=(255, 0, 0)):
        """SUMO에 POI(배송지) 추가"""
        try:
            if poi_id not in traci.poi.getIDList():
                traci.poi.add(poi_id, x, y, color)
                traci.poi.setWidth(poi_id, 20.0)
                traci.poi.setHeight(poi_id, 20.0)
                print(f"POI {poi_id} 추가 완료")
            return True
        except Exception as e:
            print(f"POI {poi_id} 추가 실패: {e}")
            return False

    def _update_poi_status(self, demand):
        """배송 완료 상태에 따라 POI 색상 변경"""
        try:
            poi_id = f"delivery_{demand.delivery_id}"
            if poi_id in traci.poi.getIDList():
                if demand.is_completed:
                    # 완료된 배송: 녹색
                    traci.poi.setColor(poi_id, (0, 255, 0))
                else:
                    # 미완료 배송: 빨간색
                    traci.poi.setColor(poi_id, (255, 0, 0))
            return True
        except Exception as e:
            print(f"POI status update failed for {poi_id}: {e}")
            return False
        except Exception as e:
            print(f"POI {poi_id} 추가 실패: {e}")
            return False

    def test(self):
        """훈련된 모델로 테스트 실행 (실시간 SUMO 시각화)"""
        print(f"\n=== 실시간 SUMO 시각화 테스트 시작 ===")

        # 테스트 모드 설정
        original_epsilon = self.epsilon
        self.epsilon = 0.1  # 10% 탐험으로 다양한 액션 보장

        # SUMO GUI 시작
        if not self._start_sumo_gui():
            print("SUMO GUI 시작 실패")
            return

        try:
            # 배송지 POI 추가
            print("배송지 위치 추가 중...")
            for i, demand in enumerate(self.env.vrptw.demands):
                vertex = self.env.vrptw.vertices[demand.destination]
                poi_id = f"delivery_{demand.delivery_id}"
                self._add_poi_to_sumo(poi_id, None, vertex.x, vertex.y, (255, 0, 0))
# 초기 배송 상태에 따라 POI 색상 설정                self._update_poi_status(demand)

            test_results = []

            # 단일 테스트 에피소드 실행
            print(f"실시간 시각화 테스트 시작")

            self.env.reset()
            episode_rewards = {agent: 0 for agent in self.env.possible_agents}
            episode_steps = 0

            # 트럭별 순차 출발 시간 설정
            n_trucks = len(self.env.vrptw.delivery_trucks)
            start_times = {f"truck_{i}": i * 2 for i in range(n_trucks)}

            # 허브 edge 찾기 (첫 번째 vertex 위치)
            hub_vertex = self.env.vrptw.vertices[0]
            all_edges = traci.edge.getIDList()
            print(f"Available edges: {len(all_edges)}")
            valid_edges = [e for e in all_edges if not e.startswith(":")]
            hub_edge = valid_edges[0] if valid_edges else "gneE0"
            print(f"Using hub edge: {hub_edge}")

# 각 차량을 SUMO에 분산 배치            print("차량을 SUMO에 추가 중...")            vehicle_count = 0            for agent in self.env.possible_agents:                # 다른 edge에 차량 분산 배치                if vehicle_count < len(valid_edges):                    vehicle_edge = valid_edges[vehicle_count]                else:                    vehicle_edge = hub_edge                                success = self._add_vehicle_to_sumo(agent, vehicle_edge)                if success:                    vehicle_count += 1                    print(f"Vehicle {agent} placed on edge: {vehicle_edge}")                else:                    print(f"Failed to add vehicle {agent}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

            print("실시간 시뮬레이션 시작...")
            print("SUMO GUI에서 재생 버튼을 눌러 시뮬레이션을 시작하세요")

            for agent in self.env.agent_iter():
                agent_idx = self.env.agent_name_idx_mapping[agent]
                truck = self.env.vrptw.delivery_trucks[agent_idx]

                # 출발 시간 확인
                if truck.current_time < start_times[agent]:
                    truck.current_time = start_times[agent]

                # 현재 상태 획득
                state = self._get_state_vector(agent)

                # 최적 액션 선택 (epsilon=0이므로 탐험 없음)
                action = self._generate_action(agent, state)
                previous_location = truck.location

                # 환경에서 액션 실행
                self.env.step(action)

                # 보상 계산
                calculated_reward = self._calculate_reward(agent, action)
                episode_rewards[agent] += calculated_reward
                episode_steps += 1

                # 차량 이동이 있는 경우 SUMO에 반영
                if previous_location != truck.location:
                    try:
                        distance, route_edges = self._calculate_road_based_distance(previous_location, truck.location)

                        # SUMO에서 차량 경로 업데이트
                        if route_edges:
                            # self._update_vehicle_route_in_sumo(agent, route_edges)  # Disabled to prevent route errors
                            pass  # Route update disabled

                        print(f"Vehicle {agent}: {previous_location} -> {truck.location}, reward: {calculated_reward:.2f}")

                        # SUMO 시뮬레이션 스텝 진행
# 모든 배송지 상태 업데이트 (시각적 피드백)                        for demand in self.env.vrptw.demands:                            self._update_poi_status(demand)
                        traci.simulationStep()

                    except Exception as e:
                        print(f"Vehicle {agent} movement failed: {e}")

                # 에피소드 종료 조건 확인
                all_delivered = all(demand.is_completed for demand in self.env.vrptw.demands)
                time_exceeded = any(truck.current_time > 180 for truck in self.env.vrptw.delivery_trucks)

                if all_delivered or time_exceeded:
                    print(f"테스트 완료: 배송 완료={all_delivered}, 시간 초과={time_exceeded}")
                    break

            # 테스트 에피소드 결과 계산
            delivered_count = sum(1 for demand in self.env.vrptw.demands if demand.is_completed)
            total_demands = len(self.env.vrptw.demands)
            success_rate = delivered_count / total_demands if total_demands > 0 else 0
            total_reward = sum(episode_rewards.values())
            avg_reward = total_reward / len(self.env.possible_agents)

            # 시간창 준수율 계산
            on_time_deliveries = 0
            for demand in self.env.vrptw.demands:
                if demand.is_completed and hasattr(demand, 'actual_delivery_time'):
                    if demand.earliest_time <= demand.actual_delivery_time <= demand.latest_time:
                        on_time_deliveries += 1

            on_time_rate = on_time_deliveries / delivered_count if delivered_count > 0 else 0


            # 최종 결과 출력
            print(f"\n=== 실시간 테스트 완료 ===")
            print(f"배송 성공률: {success_rate:.2%} ({delivered_count}/{total_demands})")
            print(f"평균 보상: {avg_reward:.2f}")
            print(f"총 스텝: {episode_steps}")

            print(f"\nSUMO GUI가 열린 상태입니다. 시뮬레이션을 계속 관찰하거나 창을 닫아서 종료하세요.")

            # SUMO GUI가 열린 상태로 유지 (사용자가 직접 제어)
            try:
                while True:
                    try:
                        traci.simulation.getTime()  # 연결 상태 확인
                        time.sleep(1)  # 1초 대기
                    except traci.exceptions.TraCIException:
                        print("SUMO GUI가 닫혔습니다.")
                        break
                    except Exception:
                        break
            except KeyboardInterrupt:
                print("\n수동 종료 요청됨")

        except Exception as e:
            print(f"실시간 테스트 중 오류: {e}")

        finally:
            # 안전한 SUMO 종료
            try:
                traci.close()
            except Exception:
                pass

        # 원래 epsilon 복원
        self.epsilon = original_epsilon

        print("실시간 테스트 세션 종료")

    def _save_test_results(self, test_summary):
        """테스트 결과 저장"""
        try:
            with open("test_results" + suffix, "w") as f:
                json.dump(test_summary, f, indent=2, ensure_ascii=False)
            print("테스트 결과 저장 완료: test_results" + suffix)
        except Exception as e:
            print(f"테스트 결과 저장 실패: {e}")

    def _save_test_replay_data(self, test_result, filename_prefix):
        """테스트 재생 데이터 저장"""
        try:
            replay_data = {
                'episode': test_result['episode'],
                'test_mode': True,
                'performance': {
                    'success_rate': test_result['success_rate'],
                    'on_time_rate': test_result['on_time_rate'],
                    'avg_reward': test_result['avg_reward_per_agent']
                },
                'vehicles': {},
                'deliveries': [],
                'simulation_info': {
                    'total_time': 180,
                    'speed_factor': 3.0,
                    'network_file': os.path.join('..', 'assets', 'cosmos', 'cosmos.sumocfg')
                }
            }

            # 차량별 경로 데이터
            for agent, route in test_result['routes'].items():
                replay_data['vehicles'][agent] = {
                    'route': route,
                    'vehicle_type': 'delivery_truck',
                    'color': self._get_vehicle_color(agent)
                }

            # 배송지 정보
            for i, demand in enumerate(self.env.vrptw.demands):
                vertex = self.env.vrptw.vertices[demand.destination]
                delivery_info = {
                    'delivery_id': demand.delivery_id,
                    'x_coordinate': vertex.x,
                    'y_coordinate': vertex.y,
                    'time_window': [demand.earliest_time, demand.latest_time],
                    'is_completed': demand.is_completed,
                    'assigned_truck': getattr(demand, 'assigned_truck', None),
                    'actual_delivery_time': getattr(demand, 'actual_delivery_time', None)
                }
                replay_data['deliveries'].append(delivery_info)

            # 파일 저장
            filename = f"{filename_prefix}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(replay_data, f, indent=2, ensure_ascii=False)
            print(f"테스트 재생 데이터 저장: {filename}")

        except Exception as e:
            print(f"테스트 재생 데이터 저장 실패: {e}")

    def train(self):
        """학습 실행"""
        self._initialize_output_files()

        for episode in range(self.episodes):
            # 100 에피소드마다 좌표 갱신 확인
            self._update_coordinates_if_needed(episode)
            self.env.reset()
            episode_rewards = {agent: 0 for agent in self.env.possible_agents}
            episode_losses = {agent: [] for agent in self.env.possible_agents}
            episode_steps = 0

            # Epsilon 감소
            if episode % self.decay_period == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

            # 트럭별 순차 출발 시간 설정 (동적 트럭 수 반영)
            n_trucks = len(self.env.vrptw.delivery_trucks)
            start_times = {f"truck_{i}": i * 2 for i in range(n_trucks)}  # 2분 간격

            for agent in self.env.agent_iter():
                agent_idx = self.env.agent_name_idx_mapping[agent]
                truck = self.env.vrptw.delivery_trucks[agent_idx]

                # 출발 시간 확인
                if truck.current_time < start_times[agent]:
                    truck.current_time = start_times[agent]

                # 현재 상태 획득
                state = self._get_state_vector(agent)

                # 액션 선택
                action = self._generate_action(agent, state)
                previous_location = truck.location

                # 환경에서 액션 실행 (순서 수정)
                self.env.step(action)

                # 액션 결과 획득
                observation, reward, done, info = self.env.last()
                next_state = self._get_state_vector(agent)
                calculated_reward = self._calculate_reward(agent, action)

                # 차량 이동 경로 기록
                if previous_location != truck.location:
                    distance, route_edges = self._calculate_road_based_distance(previous_location, truck.location)
                    self._record_vehicle_movement(
                        agent, previous_location, truck.location, action,
                        truck.current_time, route_edges
                    )

                # 배송 완료 시 시간창 검증 및 정보 기록
                if 1 <= action <= 50:
                    demand_idx = action - 1
                    if (demand_idx < len(self.env.vrptw.demands) and
                        self.env.vrptw.demands[demand_idx].is_completed):
                        # 배송 완료된 경우 시간창 검증
                        demand = self.env.vrptw.demands[demand_idx]
                        arrival_time = truck.current_time

                        # 시간창 검증
                        if demand.earliest_time <= arrival_time <= demand.latest_time:
                            # 시간창 내 배송 성공
                            demand.is_completed = True
                            if not hasattr(demand, 'assigned_truck'):
                                demand.assigned_truck = agent
                            if not hasattr(demand, 'actual_delivery_time'):
                                demand.actual_delivery_time = arrival_time
                        else:
                            # 시간창 위반 - 배송 실패 처리
                            demand.is_completed = False

                # 상태가 이미 리스트 형태이므로 변환 불필요
                # _get_state_vector에서 이미 Python 리스트로 반환됨

                # 네트워크 업데이트
                loss = self._update_network(agent, state, action, next_state, calculated_reward, done)
                if loss > 0:
                    episode_losses[agent].append(loss)

                # 통계 업데이트
                episode_rewards[agent] += calculated_reward
                self.total_steps[agent] += 1
                episode_steps += 1

                # 에피소드 종료 조건 확인
                all_delivered = all(demand.is_completed for demand in self.env.vrptw.demands)
                time_exceeded = any(truck.current_time > 180 for truck in self.env.vrptw.delivery_trucks)

                if all_delivered or time_exceeded:
                    break

            # 에피소드별 경로 데이터 저장
            self.episode_routes[episode] = {
                agent: self.vehicle_routes[agent].copy() for agent in self.env.possible_agents
            }

            # 100 에피소드마다, 마지막 에피소드, 또는 특정 중요 에피소드에서 SUMO GUI 재생용 데이터 저장
            save_replay = (
                (episode % 100 == 0 and episode > 0) or  # 100 에피소드마다
                (episode == self.episodes - 1) or        # 마지막 에피소드
                (episode in [50, 150])                   # 추가 중요 에피소드
            )

            if save_replay:
                self._save_sumo_replay_data(episode)
                print(f"재생용 데이터 저장: 에피소드 {episode}")

            # 에피소드 결과 저장
            self._save_episode_results(episode, episode_rewards, episode_losses, episode_steps)

            # 에피소드별 경로 초기화
            for agent in self.env.possible_agents:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                self.vehicle_routes[agent] = []
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if episode % 10 == 0:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                print(f"Episode {episode}: Average reward = {np.mean(list(episode_rewards.values())):.2f}, "
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                      f"Epsilon = {self.epsilon:.3f}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print(f"\n=== 훈련 완료 ===")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print(f"총 {self.episodes}개 에피소드 완료")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 모델 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        self.save_model()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print(f"SUMO 연결 종료 중...")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # SUMO 연결 정리
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if hasattr(self, 'route_calculator') and self.route_calculator:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                self.route_calculator.disconnect_sumo()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"SUMO 연결 종료 중 오류 (무시됨): {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        self._wrap_up_output_files()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print("모든 파일 저장 완료")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _update_delivery_csv(self):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """배송 완료 정보를 CSV에 업데이트"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # CSV 파일 읽기
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df = pd.read_csv("../simple_cosmos_deliveries_rl.csv")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 각 배송 요청에 대해 상태 업데이트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            for i, demand in enumerate(self.env.vrptw.demands):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                # delivery_id로 매칭하여 업데이트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                mask = df['delivery_id'] == demand.delivery_id
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                if mask.any():
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    df.loc[mask, 'is_completed'] = demand.is_completed
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    # 배송 완료된 경우 추가 정보 업데이트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    if demand.is_completed and hasattr(demand, 'assigned_truck'):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        df.loc[mask, 'assigned_truck'] = demand.assigned_truck
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    if demand.is_completed and hasattr(demand, 'actual_delivery_time'):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        df.loc[mask, 'actual_delivery_time'] = demand.actual_delivery_time
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # CSV 파일 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df.to_csv("../simple_cosmos_deliveries_rl.csv", index=False)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"배송 CSV 업데이트 중 오류: {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _update_truck_csv(self):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """트럭 상태 정보를 CSV에 업데이트"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # CSV 파일 읽기
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df = pd.read_csv("../simple_cosmos_trucks_rl.csv")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 각 트럭에 대해 상태 업데이트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            for i, truck in enumerate(self.env.vrptw.delivery_trucks):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                # truck_id로 매칭하여 업데이트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                mask = df['truck_id'] == truck.id
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                if mask.any():
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    df.loc[mask, 'current_cargo_count'] = truck.cargo_count
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    # 트럭 상태 설정
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    if truck.status == 0:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        status = "available"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    elif truck.status == 1:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        status = "busy"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    elif truck.status == 2:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        status = "returning"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    else:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        status = "available"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    df.loc[mask, 'status'] = status
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    # 위치 정보가 있다면 업데이트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    if hasattr(truck, 'location') and truck.location < len(self.env.vrptw.vertices):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        vertex = self.env.vrptw.vertices[truck.location]
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        df.loc[mask, 'current_location_x'] = vertex.x
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        df.loc[mask, 'current_location_y'] = vertex.y
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # CSV 파일 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df.to_csv("../simple_cosmos_trucks_rl.csv", index=False)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"트럭 CSV 업데이트 중 오류: {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _calculate_hub_remaining_cargo(self):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """허브에 남은 배송품 개수 계산"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 총 배송할 화물 개수
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            total_cargo = sum(demand.cargo_items_count for demand in self.env.vrptw.demands if hasattr(demand, 'cargo_items_count'))
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if total_cargo == 0:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                # demands에 cargo_items_count가 없으면 CSV에서 직접 계산
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                df = pd.read_csv("../simple_cosmos_deliveries_rl.csv")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                total_cargo = df['cargo_items_count'].sum()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 배송 완료된 화물 개수
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            delivered_cargo = 0
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            for demand in self.env.vrptw.demands:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                if demand.is_completed:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    if hasattr(demand, 'cargo_items_count'):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        delivered_cargo += demand.cargo_items_count
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    else:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        # CSV에서 해당 배송의 화물 개수 찾기
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        df = pd.read_csv("../simple_cosmos_deliveries_rl.csv")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        cargo_count = df[df['delivery_id'] == demand.delivery_id]['cargo_items_count'].iloc[0]
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                        delivered_cargo += cargo_count
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 트럭에 적재된 화물 개수
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            cargo_in_trucks = sum(truck.cargo_count for truck in self.env.vrptw.delivery_trucks)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 허브에 남은 화물 계산
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            remaining_at_hub = total_cargo - delivered_cargo - cargo_in_trucks
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            return max(0, remaining_at_hub)  # 음수 방지
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"허브 잔여 화물 계산 중 오류: {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            return 0
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _update_hub_csv(self):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """허브 상태 정보를 CSV에 업데이트"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # CSV 파일 읽기
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df = pd.read_csv("../simple_cosmos_hub.csv")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 허브 잔여 화물 계산
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            remaining_cargo = self._calculate_hub_remaining_cargo()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 허브 정보 업데이트 (첫 번째 허브)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df.loc[0, 'remaining_cargo_items'] = remaining_cargo
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # CSV 파일 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            df.to_csv("../simple_cosmos_hub.csv", index=False)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"허브 CSV 업데이트 중 오류: {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _update_csv_files(self):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """CSV 파일들 업데이트"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        self._update_delivery_csv()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        self._update_truck_csv()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        self._update_hub_csv()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _save_episode_results(self, episode, rewards, losses, steps):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """에피소드 결과를 파일에 저장"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 보상 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        reward_data = {episode: {agent: reward for agent, reward in rewards.items()}}
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        with open("reward" + suffix, "a") as f:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if episode > 0:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                f.write(",")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            f.write(json.dumps(reward_data)[1:-1])
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 손실 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        loss_data = {episode: {agent: np.mean(loss) if loss else 0 for agent, loss in losses.items()}}
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        with open("loss" + suffix, "a") as f:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if episode > 0:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                f.write(",")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            f.write(json.dumps(loss_data)[1:-1])
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 메트릭 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        delivered_count = sum(1 for demand in self.env.vrptw.demands if demand.is_completed)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        metric_data = {
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            episode: {
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                "delivered_count": delivered_count,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                "total_demands": len(self.env.vrptw.demands),
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                "success_rate": delivered_count / len(self.env.vrptw.demands),
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                "episode_steps": steps,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            }
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        }
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        with open("metrics" + suffix, "a") as f:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if episode > 0:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                f.write(",")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            f.write(json.dumps(metric_data)[1:-1])
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # CSV 파일 업데이트 추가
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        self._update_csv_files()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _save_sumo_replay_data(self, episode):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """SUMO GUI 재생용 데이터 저장"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        replay_data = {
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            'episode': episode,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            'vehicles': {},
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            'deliveries': [],
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            'simulation_info': {
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'total_time': 180,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'speed_factor': 3.0,  # 빠른 재생을 위한 배수
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'network_file': os.path.join('..', 'assets', 'cosmos', 'cosmos.sumocfg')
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            }
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        }
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 각 차량별 경로 데이터 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        for agent in self.env.possible_agents:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if agent in self.episode_routes[episode]:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                vehicle_route = self.episode_routes[episode][agent]
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                replay_data['vehicles'][agent] = {
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    'route': vehicle_route,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    'vehicle_type': 'delivery_truck',
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                    'color': self._get_vehicle_color(agent)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                }
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 배송지 정보 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        for i, demand in enumerate(self.env.vrptw.demands):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            vertex = self.env.vrptw.vertices[demand.destination]
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            delivery_info = {
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'delivery_id': demand.delivery_id,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'x_coordinate': vertex.x,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'y_coordinate': vertex.y,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'time_window': [demand.earliest_time, demand.latest_time],
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'is_completed': demand.is_completed,
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'assigned_truck': getattr(demand, 'assigned_truck', None),
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                'actual_delivery_time': getattr(demand, 'actual_delivery_time', None)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            }
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            replay_data['deliveries'].append(delivery_info)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # JSON 파일로 저장
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        replay_filename = f"sumo_replay_episode_{episode}.json"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            with open(replay_filename, 'w', encoding='utf-8') as f:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                json.dump(replay_data, f, indent=2, ensure_ascii=False)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"SUMO 재생용 데이터 저장 완료: {replay_filename}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print(f"SUMO 재생 데이터 저장 실패: {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    def _get_vehicle_color(self, agent):
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        """차량별 고유 색상 반환"""
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        agent_idx = int(agent.split('_')[1]) if '_' in agent else 0
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        return colors[agent_idx % len(colors)]
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
# 학습 및 테스트 실행
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
if __name__ == "__main__":
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    try:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print("=== VRPTW DQN Training and Testing Start ===")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        dqn = VRPTW_DQN(env=env)
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        # 기존 모델 로딩 시도
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        if dqn.load_model():
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print("기존 모델 로딩 완료, 테스트만 실행합니다.")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            user_choice = input("1: 새로 훈련, 2: 테스트만, 3: 훈련 후 테스트 (기본값: 3): ").strip() or "3"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        else:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            print("새로운 모델로 훈련을 시작합니다.")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            user_choice = input("1: 훈련만, 2: 훈련 후 테스트 (기본값: 2): ").strip() or "2"
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        if user_choice == "1":
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 훈련만
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            dqn.train()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        elif user_choice == "2":
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            if not dqn.load_model():
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                print("저장된 모델이 없으므로 훈련부터 시작합니다.")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
                dqn.train()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 테스트만
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            dqn.test()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        else:  # "3" 또는 기타
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            # 훈련 후 테스트
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            dqn.train()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
            dqn.test()
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print("=== 모든 작업이 성공적으로 완료되었습니다 ===")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")

# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    except Exception as e:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print(f"Error occurred: {e}")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print("이것이 'peer shutdown' 오류라면 정상적인 완료입니다.")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
    finally:
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
        print("Program terminated")
# 시뮬레이션 스텝 실행하여 차량이 나타나도록 함            traci.simulationStep()            vehicles_in_sumo = traci.vehicle.getIDList()            print(f"SUMO에 현재 있는 차량: {len(vehicles_in_sumo)}개 - {vehicles_in_sumo}")
# 각 차량을 정지 상태로 설정하여 확실히 보이게 함            for vehicle in vehicles_in_sumo:                traci.vehicle.setSpeed(vehicle, 0.0)                traci.vehicle.setColor(vehicle, (0, 255, 0))  # 녹색으로 변경            print(f"All vehicles set to stationary and green")
