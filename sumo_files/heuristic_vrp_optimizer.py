#!/usr/bin/env python3
"""
휴리스틱 기반 차량 라우팅 최적화 시스템
그리디, 시간슬롯 기반 휴리스틱 알고리즘을 사용한 VRPTW 해결

Requirements:
- pandas
- numpy
- streamlit (for web interface)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import os
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Delivery:
    """배송 정보 클래스"""
    delivery_id: str
    intersection_id: int
    x: float
    y: float
    area: str
    time_slot: int
    time_slot_label: str
    earliest_time: int  # minutes from start (12:00 = 0)
    latest_time: int    # minutes from start
    time_window: int    # window duration in minutes
    delivery_type: str
    service_time: int   # minutes
    cargo_weight: float
    cargo_items: int
    distance_from_hub: float


@dataclass
class Truck:
    """트럭 정보 클래스"""
    truck_id: str
    truck_name: str
    max_cargo_items: int
    max_cargo_weight: float
    speed_kmh: float
    current_x: float
    current_y: float
    status: str
    current_cargo_count: int


@dataclass
class Hub:
    """허브 정보 클래스"""
    hub_id: str
    name: str
    intersection_id: int
    x: float
    y: float
    area: str
    operating_start: int  # 0 = 12:00
    operating_end: int    # 180 = 15:00 (3시간 운영)
    total_operating_hours: int
    max_trucks: int


@dataclass
class RouteInfo:
    """경로 정보"""
    truck_id: str
    route: List[int]  # delivery indices
    total_distance: float
    total_time: float
    deliveries_count: int
    total_weight: float
    total_items: int
    algorithm_used: str


class HeuristicVRPOptimizer:
    """휴리스틱 기반 VRP 최적화 클래스"""

    def __init__(self):
        self.deliveries: List[Delivery] = []
        self.trucks: List[Truck] = []
        self.hub: Optional[Hub] = None
        self.distance_matrix = None
        self.time_matrix = None
        self.solutions = {}

    def load_csv_data(self, deliveries_file: str, trucks_file: str, hub_file: str):
        """실제 CSV 파일에서 데이터 로드"""
        print("=== 실제 CSV 데이터 로드 중 ===")

        # 배송지 데이터 로드
        df_deliveries = pd.read_csv(deliveries_file)
        print(f"배송지 데이터: {len(df_deliveries)} 행")

        # 모든 배송지를 대상으로 최적화 수행 (is_completed 상태 무시)
        print(f"전체 배송지: {len(df_deliveries)} 행 (모든 배송지를 최적화 대상으로 설정)")

        for _, row in df_deliveries.iterrows():
            delivery = Delivery(
                delivery_id=row['delivery_id'],
                intersection_id=int(row['intersection_id']),
                x=float(row['x_coordinate']),
                y=float(row['y_coordinate']),
                area=row['area'],
                time_slot=int(row['time_slot']),
                time_slot_label=row['time_slot_label'],
                earliest_time=int(row['earliest_time_minutes']),
                latest_time=int(row['latest_time_minutes']),
                time_window=int(row['time_window_minutes']),
                delivery_type=row['delivery_type'],
                service_time=int(row['service_time_minutes']),
                cargo_weight=float(row['cargo_weight_kg']),
                cargo_items=int(row['cargo_items_count']),
                distance_from_hub=float(row['distance_from_hub_meters'])
            )
            self.deliveries.append(delivery)

        # 트럭 데이터 로드
        df_trucks = pd.read_csv(trucks_file)
        print(f"트럭 데이터: {len(df_trucks)} 대")

        for _, row in df_trucks.iterrows():
            truck = Truck(
                truck_id=row['truck_id'],
                truck_name=row['truck_name'],
                max_cargo_items=int(row['max_cargo_items']),
                max_cargo_weight=float(row['max_cargo_weight_kg']),
                speed_kmh=float(row['average_speed_kmh']),
                current_x=float(row['current_location_x']),
                current_y=float(row['current_location_y']),
                status=row['status'],
                current_cargo_count=int(row['current_cargo_count'])
            )
            self.trucks.append(truck)

        # 허브 데이터 로드
        df_hub = pd.read_csv(hub_file)
        hub_row = df_hub.iloc[0]
        self.hub = Hub(
            hub_id=hub_row['hub_id'],
            name=hub_row['name'],
            intersection_id=int(hub_row['intersection_id']),
            x=float(hub_row['x_coordinate']),
            y=float(hub_row['y_coordinate']),
            area=hub_row['area'],
            operating_start=int(hub_row['operating_start_minutes']),
            operating_end=int(hub_row['operating_end_minutes']),
            total_operating_hours=int(hub_row['total_operating_hours']),
            max_trucks=int(hub_row['max_trucks'])
        )

        print(f"로드 완료: 배송지 {len(self.deliveries)}개, 트럭 {len(self.trucks)}대, 허브 1개")

    def load_sample_data(self):
        """샘플 데이터 로드 (CSV 파일이 없을 때 사용)"""
        # 허브 생성
        self.hub = Hub(
            hub_id="HUB01",
            name="메인 허브",
            intersection_id=0,
            x=0.0,
            y=0.0,
            area="중앙",
            operating_start=0,
            operating_end=480,  # 8시간 운영
            total_operating_hours=8,
            max_trucks=10
        )

        # 샘플 배송지 생성
        sample_deliveries = [
            ("D001", 1, 100, 100, "북동", 1, "12:00-13:00", 0, 60, 60, "일반", 10, 5.0, 2),
            ("D002", 2, -100, 100, "북서", 1, "12:00-13:00", 0, 60, 60, "일반", 15, 3.0, 1),
            ("D003", 3, 100, -100, "남동", 2, "13:00-14:00", 60, 120, 60, "긴급", 8, 7.0, 3),
            ("D004", 4, -100, -100, "남서", 2, "13:00-14:00", 60, 120, 60, "일반", 12, 4.5, 2),
            ("D005", 5, 150, 0, "동", 3, "14:00-15:00", 120, 180, 60, "일반", 10, 6.0, 2),
            ("D006", 6, -150, 0, "서", 3, "14:00-15:00", 120, 180, 60, "긴급", 20, 8.0, 4),
            ("D007", 7, 0, 150, "북", 1, "12:00-13:00", 0, 60, 60, "일반", 5, 2.5, 1),
            ("D008", 8, 0, -150, "남", 4, "15:00-16:00", 180, 240, 60, "일반", 25, 9.0, 5),
        ]

        for data in sample_deliveries:
            delivery = Delivery(
                delivery_id=data[0],
                intersection_id=data[1],
                x=data[2],
                y=data[3],
                area=data[4],
                time_slot=data[5],
                time_slot_label=data[6],
                earliest_time=data[7],
                latest_time=data[8],
                time_window=data[9],
                delivery_type=data[10],
                service_time=data[11],
                cargo_weight=data[12],
                cargo_items=data[13],
                distance_from_hub=math.sqrt(data[2]**2 + data[3]**2)
            )
            self.deliveries.append(delivery)

        # 샘플 트럭 생성
        sample_trucks = [
            ("T001", "truck_1", 10, 50.0, 30.0, 0.0, 0.0, "available", 0),
            ("T002", "truck_2", 12, 60.0, 35.0, 0.0, 0.0, "available", 0),
            ("T003", "truck_3", 8, 40.0, 28.0, 0.0, 0.0, "available", 0),
        ]

        for data in sample_trucks:
            truck = Truck(
                truck_id=data[0],
                truck_name=data[1],
                max_cargo_items=data[2],
                max_cargo_weight=data[3],
                speed_kmh=data[4],
                current_x=data[5],
                current_y=data[6],
                status=data[7],
                current_cargo_count=data[8]
            )
            self.trucks.append(truck)

    def calculate_euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """유클리드 거리 계산"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def build_distance_time_matrix(self):
        """거리 및 시간 매트릭스 구축"""
        # 위치 리스트: [허브] + [배송지들]
        locations = [(self.hub.x, self.hub.y)]

        for delivery in self.deliveries:
            locations.append((delivery.x, delivery.y))

        n = len(locations)
        self.distance_matrix = np.zeros((n, n))
        self.time_matrix = np.zeros((n, n))

        # 평균 속도 (km/h)
        avg_speed_kmh = 30.0
        if self.trucks:
            avg_speed_kmh = self.trucks[0].speed_kmh

        for i in range(n):
            for j in range(n):
                if i == j:
                    self.distance_matrix[i][j] = 0
                    self.time_matrix[i][j] = 0
                else:
                    x1, y1 = locations[i]
                    x2, y2 = locations[j]

                    distance = self.calculate_euclidean_distance(x1, y1, x2, y2)
                    self.distance_matrix[i][j] = distance

                    # 시간 계산 (분)
                    time_hours = (distance / 1000) / avg_speed_kmh
                    time_minutes = time_hours * 60
                    self.time_matrix[i][j] = time_minutes

    def is_feasible_route(self, truck: Truck, delivery_indices: List[int]) -> Tuple[bool, dict]:
        """경로가 실행 가능한지 확인"""
        if not delivery_indices:
            return True, {'time': 0, 'distance': 0, 'weight': 0, 'items': 0}

        # 용량 제약 확인
        total_weight = sum(self.deliveries[i].cargo_weight for i in delivery_indices)
        total_items = sum(self.deliveries[i].cargo_items for i in delivery_indices)

        if total_weight > truck.max_cargo_weight or total_items > truck.max_cargo_items:
            return False, {}

        # 시간 제약 확인
        current_time = 0  # 허브에서 12:00에 시작
        total_distance = 0

        for i, delivery_idx in enumerate(delivery_indices):
            delivery = self.deliveries[delivery_idx]

            # 이동 시간 계산
            if i == 0:
                # 허브에서 첫 배송지로
                travel_time = self.time_matrix[0][delivery_idx + 1]
                travel_distance = self.distance_matrix[0][delivery_idx + 1]
            else:
                # 이전 배송지에서 현재 배송지로
                prev_delivery_idx = delivery_indices[i-1]
                travel_time = self.time_matrix[prev_delivery_idx + 1][delivery_idx + 1]
                travel_distance = self.distance_matrix[prev_delivery_idx + 1][delivery_idx + 1]

            current_time += travel_time
            total_distance += travel_distance

            # 시간창 제약 확인
            if current_time > delivery.latest_time:
                return False, {}

            # 일찍 도착한 경우 대기
            if current_time < delivery.earliest_time:
                current_time = delivery.earliest_time

            # 서비스 시간 추가
            current_time += delivery.service_time

        # 허브로 복귀 시간
        last_delivery_idx = delivery_indices[-1]
        return_time = self.time_matrix[last_delivery_idx + 1][0]
        return_distance = self.distance_matrix[last_delivery_idx + 1][0]
        current_time += return_time
        total_distance += return_distance

        return True, {
            'time': current_time,
            'distance': total_distance,
            'weight': total_weight,
            'items': total_items
        }

    def solve_greedy_algorithm(self, max_trucks: int = None) -> Dict:
        """그리디 알고리즘으로 VRP 해결"""
        if max_trucks is None:
            # 최대 배송지 수만큼 트럭 사용 가능 (모든 배송 완료를 목표)
            max_trucks = len(self.deliveries)

        unassigned = set(range(len(self.deliveries)))
        routes = []
        truck_idx = 0

        while unassigned and truck_idx < max_trucks:
            if truck_idx < len(self.trucks):
                current_truck = self.trucks[truck_idx]
            else:
                # 가상 트럭 생성 (기존 트럭들과 동일한 스펙)
                base_truck = self.trucks[0] if self.trucks else None
                if base_truck:
                    current_truck = Truck(
                        truck_id=f"T{truck_idx+1:02d}",
                        truck_name=f"virtual_truck_{truck_idx+1}",
                        max_cargo_items=base_truck.max_cargo_items,
                        max_cargo_weight=base_truck.max_cargo_weight,
                        speed_kmh=base_truck.speed_kmh,
                        current_x=base_truck.current_x,
                        current_y=base_truck.current_y,
                        status="available",
                        current_cargo_count=0
                    )
                else:
                    # 기본 트럭 스펙
                    current_truck = Truck(
                        truck_id=f"T{truck_idx+1:02d}",
                        truck_name=f"virtual_truck_{truck_idx+1}",
                        max_cargo_items=12,
                        max_cargo_weight=100.0,
                        speed_kmh=30.0,
                        current_x=0.0,
                        current_y=0.0,
                        status="available",
                        current_cargo_count=0
                    )

            current_route = []

            while unassigned:
                best_delivery = None
                best_cost = float('inf')

                # 현재 위치
                if not current_route:
                    current_pos = 0  # 허브
                else:
                    current_pos = current_route[-1] + 1

                # 가장 좋은 배송지 찾기 (트럭 효율성 중시)
                for delivery_idx in unassigned:
                    test_route = current_route + [delivery_idx]
                    feasible, metrics = self.is_feasible_route(current_truck, test_route)

                    if feasible:
                        # 비용 계산: 거리 + 시간 패널티 + 트럭 활용도 보너스
                        distance_cost = self.distance_matrix[current_pos][delivery_idx + 1]
                        delivery = self.deliveries[delivery_idx]

                        # 시간창 긴급도 (패널티 줄임)
                        time_urgency = delivery.latest_time - delivery.earliest_time
                        time_penalty = (180 - time_urgency) * 0.02  # 기존 0.1에서 0.02로 대폭 줄임

                        # 트럭 활용도 보너스 (같은 트럭에 더 많이 배정할수록 좋음)
                        utilization_bonus = len(current_route) * -15  # 현재 경로 배송지 수만큼 보너스

                        # 같은 지역/시간슬롯 보너스
                        area_bonus = 0
                        time_slot_bonus = 0
                        if current_route:
                            last_delivery = self.deliveries[current_route[-1]]
                            if last_delivery.area == delivery.area:
                                area_bonus = -25  # 같은 지역 보너스
                            if last_delivery.time_slot == delivery.time_slot:
                                time_slot_bonus = -30  # 같은 시간슬롯 보너스

                        cost = distance_cost + time_penalty + utilization_bonus + area_bonus + time_slot_bonus

                        if cost < best_cost:
                            best_cost = cost
                            best_delivery = delivery_idx

                if best_delivery is not None:
                    current_route.append(best_delivery)
                    unassigned.remove(best_delivery)
                else:
                    break

            if current_route:
                feasible, metrics = self.is_feasible_route(current_truck, current_route)
                if feasible:
                    route_info = RouteInfo(
                        truck_id=current_truck.truck_id,
                        route=current_route.copy(),
                        total_distance=metrics['distance'],
                        total_time=metrics['time'],
                        deliveries_count=len(current_route),
                        total_weight=metrics['weight'],
                        total_items=metrics['items'],
                        algorithm_used="Greedy"
                    )
                    routes.append(route_info)

            truck_idx += 1

        return {
            'routes': routes,
            'unassigned': list(unassigned),
            'total_trucks_used': len(routes),
            'total_deliveries_assigned': len(self.deliveries) - len(unassigned),
            'algorithm': 'Greedy'
        }

    def solve_time_slot_algorithm(self, max_trucks: int = None) -> Dict:
        """시간 슬롯 기반 휴리스틱 알고리즘으로 VRP 해결"""
        if max_trucks is None:
            # 최대 배송지 수만큼 트럭 사용 가능 (모든 배송 완료를 목표)
            max_trucks = len(self.deliveries)

        # 시간 슬롯별로 배송지 그룹화
        time_slots = {}
        for i, delivery in enumerate(self.deliveries):
            slot = delivery.time_slot
            if slot not in time_slots:
                time_slots[slot] = []
            time_slots[slot].append(i)

        unassigned = set(range(len(self.deliveries)))
        routes = []
        truck_idx = 0

        # 시간 슬롯 순서로 처리
        for slot in sorted(time_slots.keys()):
            slot_deliveries = [d for d in time_slots[slot] if d in unassigned]

            while slot_deliveries and truck_idx < max_trucks:
                if truck_idx < len(self.trucks):
                    current_truck = self.trucks[truck_idx]
                else:
                    # 가상 트럭 생성 (기존 트럭들과 동일한 스펙)
                    base_truck = self.trucks[0] if self.trucks else None
                    if base_truck:
                        current_truck = Truck(
                            truck_id=f"T{truck_idx+1:02d}",
                            truck_name=f"virtual_truck_{truck_idx+1}",
                            max_cargo_items=base_truck.max_cargo_items,
                            max_cargo_weight=base_truck.max_cargo_weight,
                            speed_kmh=base_truck.speed_kmh,
                            current_x=base_truck.current_x,
                            current_y=base_truck.current_y,
                            status="available",
                            current_cargo_count=0
                        )
                    else:
                        # 기본 트럭 스펙
                        current_truck = Truck(
                            truck_id=f"T{truck_idx+1:02d}",
                            truck_name=f"virtual_truck_{truck_idx+1}",
                            max_cargo_items=12,
                            max_cargo_weight=100.0,
                            speed_kmh=30.0,
                            current_x=0.0,
                            current_y=0.0,
                            status="available",
                            current_cargo_count=0
                        )

                current_route = []
                remaining_slot_deliveries = slot_deliveries.copy()

                while remaining_slot_deliveries:
                    best_delivery = None
                    best_cost = float('inf')

                    current_pos = 0 if not current_route else current_route[-1] + 1

                    for delivery_idx in remaining_slot_deliveries:
                        test_route = current_route + [delivery_idx]
                        feasible, metrics = self.is_feasible_route(current_truck, test_route)

                        if feasible:
                            cost = self.distance_matrix[current_pos][delivery_idx + 1]
                            if cost < best_cost:
                                best_cost = cost
                                best_delivery = delivery_idx

                    if best_delivery is not None:
                        current_route.append(best_delivery)
                        remaining_slot_deliveries.remove(best_delivery)
                        slot_deliveries.remove(best_delivery)
                        unassigned.remove(best_delivery)
                    else:
                        break

                if current_route:
                    feasible, metrics = self.is_feasible_route(current_truck, current_route)
                    if feasible:
                        route_info = RouteInfo(
                            truck_id=current_truck.truck_id,
                            route=current_route.copy(),
                            total_distance=metrics['distance'],
                            total_time=metrics['time'],
                            deliveries_count=len(current_route),
                            total_weight=metrics['weight'],
                            total_items=metrics['items'],
                            algorithm_used="Time Slot"
                        )
                        routes.append(route_info)

                truck_idx += 1

        return {
            'routes': routes,
            'unassigned': list(unassigned),
            'total_trucks_used': len(routes),
            'total_deliveries_assigned': len(self.deliveries) - len(unassigned),
            'algorithm': 'Time Slot'
        }

    def solve_nearest_neighbor_algorithm(self, max_trucks: int = None) -> Dict:
        """최근접 이웃 알고리즘으로 VRP 해결"""
        if max_trucks is None:
            # 최대 배송지 수만큼 트럭 사용 가능 (모든 배송 완료를 목표)
            max_trucks = len(self.deliveries)

        unassigned = set(range(len(self.deliveries)))
        routes = []
        truck_idx = 0

        while unassigned and truck_idx < max_trucks:
            if truck_idx < len(self.trucks):
                current_truck = self.trucks[truck_idx]
            else:
                # 가상 트럭 생성 (기존 트럭들과 동일한 스펙)
                base_truck = self.trucks[0] if self.trucks else None
                if base_truck:
                    current_truck = Truck(
                        truck_id=f"T{truck_idx+1:02d}",
                        truck_name=f"virtual_truck_{truck_idx+1}",
                        max_cargo_items=base_truck.max_cargo_items,
                        max_cargo_weight=base_truck.max_cargo_weight,
                        speed_kmh=base_truck.speed_kmh,
                        current_x=base_truck.current_x,
                        current_y=base_truck.current_y,
                        status="available",
                        current_cargo_count=0
                    )
                else:
                    # 기본 트럭 스펙
                    current_truck = Truck(
                        truck_id=f"T{truck_idx+1:02d}",
                        truck_name=f"virtual_truck_{truck_idx+1}",
                        max_cargo_items=12,
                        max_cargo_weight=100.0,
                        speed_kmh=30.0,
                        current_x=0.0,
                        current_y=0.0,
                        status="available",
                        current_cargo_count=0
                    )

            current_route = []

            # 허브에서 가장 가까운 배송지부터 시작
            while unassigned:
                best_delivery = None
                best_distance = float('inf')

                current_pos = 0 if not current_route else current_route[-1] + 1

                for delivery_idx in unassigned:
                    test_route = current_route + [delivery_idx]
                    feasible, metrics = self.is_feasible_route(current_truck, test_route)

                    if feasible:
                        # 거리 기반이지만 트럭 효율성도 고려
                        distance = self.distance_matrix[current_pos][delivery_idx + 1]
                        delivery = self.deliveries[delivery_idx]

                        # 트럭 활용도 보너스
                        utilization_bonus = len(current_route) * -10

                        # 같은 지역/시간슬롯 보너스
                        area_bonus = 0
                        time_slot_bonus = 0
                        if current_route:
                            last_delivery = self.deliveries[current_route[-1]]
                            if last_delivery.area == delivery.area:
                                area_bonus = -20
                            if last_delivery.time_slot == delivery.time_slot:
                                time_slot_bonus = -25

                        adjusted_distance = distance + utilization_bonus + area_bonus + time_slot_bonus

                        if adjusted_distance < best_distance:
                            best_distance = adjusted_distance
                            best_delivery = delivery_idx

                if best_delivery is not None:
                    current_route.append(best_delivery)
                    unassigned.remove(best_delivery)
                else:
                    break

            if current_route:
                feasible, metrics = self.is_feasible_route(current_truck, current_route)
                if feasible:
                    route_info = RouteInfo(
                        truck_id=current_truck.truck_id,
                        route=current_route.copy(),
                        total_distance=metrics['distance'],
                        total_time=metrics['time'],
                        deliveries_count=len(current_route),
                        total_weight=metrics['weight'],
                        total_items=metrics['items'],
                        algorithm_used="Nearest Neighbor"
                    )
                    routes.append(route_info)

            truck_idx += 1

        return {
            'routes': routes,
            'unassigned': list(unassigned),
            'total_trucks_used': len(routes),
            'total_deliveries_assigned': len(self.deliveries) - len(unassigned),
            'algorithm': 'Nearest Neighbor'
        }

    def run_all_algorithms(self, use_real_data: bool = True) -> Dict:
        """모든 휴리스틱 알고리즘 실행 및 비교"""
        # 데이터 로드
        if use_real_data:
            try:
                # 실제 CSV 데이터 사용
                self.load_csv_data(
                    "C:/Users/Administrator/projects/cosmos_opt/simple_cosmos_deliveries.csv",
                    "C:/Users/Administrator/projects/cosmos_opt/simple_cosmos_trucks.csv",
                    "C:/Users/Administrator/projects/cosmos_opt/simple_cosmos_hub.csv"
                )
                print("✅ 실제 CSV 데이터 로드 성공")
            except Exception as e:
                print(f"⚠️ 실제 데이터 로드 실패: {e}")
                print("🔄 샘플 데이터로 대체합니다.")
                self.load_sample_data()
        else:
            # 샘플 데이터 사용
            self.load_sample_data()

        # 거리/시간 매트릭스 구축
        self.build_distance_time_matrix()

        # 모든 알고리즘 실행
        solutions = {}

        solutions['greedy'] = self.solve_greedy_algorithm()
        solutions['time_slot'] = self.solve_time_slot_algorithm()
        solutions['nearest_neighbor'] = self.solve_nearest_neighbor_algorithm()

        # 최적 해 선택
        best_solution = None
        best_score = float('-inf')

        for algo_name, solution in solutions.items():
            # 점수 계산: 배송 완료율 * 100 - 사용 트럭 수
            if len(self.deliveries) > 0:
                score = (solution['total_deliveries_assigned'] / len(self.deliveries)) * 100 - solution['total_trucks_used']
            else:
                score = 0
            solution['score'] = score

            if score > best_score:
                best_score = score
                best_solution = solution
                best_solution['best_algorithm'] = algo_name

        self.solutions = solutions

        # 최적화 완료 후 CSV 파일 업데이트 (실제 데이터 사용 시에만)
        if use_real_data and best_solution:
            csv_file = "C:/Users/Administrator/projects/cosmos_opt/simple_cosmos_deliveries.csv"
            self.update_csv_results(best_solution, csv_file)

        return {
            'solutions': solutions,
            'best_solution': best_solution,
            'deliveries': self.deliveries,
            'trucks': self.trucks,
            'hub': self.hub
        }

    def get_solution_summary(self) -> Dict:
        """솔루션 요약 정보 반환"""
        if not self.solutions:
            return {}

        summary = {}
        total_deliveries = len(self.deliveries)

        for algo_name, solution in self.solutions.items():
            # ZeroDivisionError 방지
            completion_rate = (solution['total_deliveries_assigned'] / total_deliveries) if total_deliveries > 0 else 0

            summary[algo_name] = {
                'trucks_used': solution['total_trucks_used'],
                'deliveries_completed': solution['total_deliveries_assigned'],
                'completion_rate': completion_rate,
                'total_distance': sum(route.total_distance for route in solution['routes']) if solution['routes'] else 0,
                'total_time': max(route.total_time for route in solution['routes']) if solution['routes'] else 0,
                'score': solution.get('score', 0)
            }

        return summary

    def update_csv_results(self, solution: Dict, deliveries_file: str):
        """최적화 결과를 CSV 파일에 업데이트"""
        print("=== CSV 파일 결과 업데이트 ===")

        try:
            # 원본 파일 읽기
            df = pd.read_csv(deliveries_file)

            # 모든 배송지를 일단 미완료로 초기화
            df['is_completed'] = False
            df['assigned_truck'] = ''
            df['actual_delivery_time'] = ''

            # 각 트럭 경로별로 결과 업데이트
            for route_info in solution['routes']:
                for delivery_idx in route_info.route:
                    delivery_id = self.deliveries[delivery_idx].delivery_id
                    mask = df['delivery_id'] == delivery_id

                    df.loc[mask, 'is_completed'] = True
                    df.loc[mask, 'assigned_truck'] = route_info.truck_id
                    # 실제 배송 시간은 간단히 시간슬롯 라벨 사용
                    delivery = self.deliveries[delivery_idx]
                    df.loc[mask, 'actual_delivery_time'] = delivery.time_slot_label.split('-')[0]

            # 파일 저장
            df.to_csv(deliveries_file, index=False)
            print(f"결과 저장 완료: {deliveries_file}")

            # 결과 요약
            completed = df['is_completed'].sum()
            total = len(df)
            print(f"배송 완료: {completed}/{total} ({completed/total*100:.1f}%)")

            return True

        except Exception as e:
            print(f"CSV 업데이트 실패: {e}")
            return False


def main():
    """메인 실행 함수 (테스트용)"""
    optimizer = HeuristicVRPOptimizer()
    results = optimizer.run_all_algorithms()

    print("=" * 60)
    print("    휴리스틱 VRP 최적화 결과")
    print("=" * 60)

    summary = optimizer.get_solution_summary()

    for algo_name, metrics in summary.items():
        print(f"\n[{algo_name.upper()}]")
        print(f"  사용 트럭: {metrics['trucks_used']}대")
        print(f"  완료 배송: {metrics['deliveries_completed']}/{len(optimizer.deliveries)} ({metrics['completion_rate']:.1%})")
        print(f"  총 거리: {metrics['total_distance']:.1f}m")
        print(f"  총 시간: {metrics['total_time']:.1f}분")
        print(f"  점수: {metrics['score']:.1f}")

    print(f"\n최적 알고리즘: {results['best_solution']['best_algorithm'].upper()}")


if __name__ == "__main__":
    main()