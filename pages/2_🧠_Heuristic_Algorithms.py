import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
import json
from datetime import datetime

# 휴리스틱 알고리즘 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sumo_files'))

try:
    from heuristic_vrp_optimizer import HeuristicVRPOptimizer
    st.success("✅ 휴리스틱 최적화 모듈 로드 완료!")
except ImportError as e:
    st.error(f"⚠️ 휴리스틱 모듈 로드 실패: {str(e)}")

    # 더미 클래스 생성
    class DummyHeuristicVRPOptimizer:
        def __init__(self):
            pass
        def run_all_algorithms(self):
            return {'solutions': {}, 'best_solution': None}
        def get_solution_summary(self):
            return {}

    HeuristicVRPOptimizer = DummyHeuristicVRPOptimizer
    st.warning("🔄 더미 클래스로 실행됩니다.")

# 페이지 설정
st.set_page_config(
    page_title="휴리스틱 알고리즘 - VRPTW",
    page_icon="🧠",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        margin-bottom: 2rem;
    }

    .algorithm-info {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }

    .optimization-status {
        border: 2px solid #27AE60;
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
        color: #27AE60;
        font-weight: bold;
    }

    .status-completed {
        color: #3498DB;
        font-weight: bold;
    }

    .algorithm-card {
        background: white;
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .algorithm-card:hover {
        border-color: #FF6B6B;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .winner-card {
        border-color: #F39C12 !important;
        background: linear-gradient(135deg, #FFF3CD 0%, #FCF3CF 100%);
    }
</style>
""", unsafe_allow_html=True)

def initialize_heuristic_session_state():
    """휴리스틱 관련 세션 상태 초기화"""
    if 'heuristic_optimizer' not in st.session_state:
        st.session_state.heuristic_optimizer = None
    if 'heuristic_running' not in st.session_state:
        st.session_state.heuristic_running = False
    if 'heuristic_results' not in st.session_state:
        st.session_state.heuristic_results = None
    if 'selected_algorithms' not in st.session_state:
        st.session_state.selected_algorithms = ['greedy', 'time_slot', 'nearest_neighbor']
    if 'optimization_progress' not in st.session_state:
        st.session_state.optimization_progress = 0
    if 'use_real_data' not in st.session_state:
        st.session_state.use_real_data = True

def create_algorithm_info():
    """알고리즘 정보 섹션"""
    st.markdown("""
    <div class="algorithm-info">
        <h3>🧠 휴리스틱 기반 최적화 알고리즘</h3>
        <p><strong>핵심 특징:</strong></p>
        <ul>
            <li><strong>빠른 계산:</strong> 실시간 의사결정에 적합한 고속 연산</li>
            <li><strong>직관적 해법:</strong> 인간의 경험과 직관을 모델링</li>
            <li><strong>실용적 접근:</strong> 완벽하지 않지만 실용적인 해 제공</li>
            <li><strong>확장성:</strong> 대규모 문제에도 적용 가능</li>
        </ul>
        <p><strong>제공 알고리즘:</strong> 그리디, 시간슬롯 기반, 최근접 이웃</p>
    </div>
    """, unsafe_allow_html=True)

def create_algorithm_selection():
    """알고리즘 선택 패널"""
    st.markdown("### ⚙️ 알고리즘 선택 및 설정")

    # 데이터 소스 선택
    st.markdown("#### 📊 데이터 소스 선택")
    col_data1, col_data2 = st.columns(2)

    with col_data1:
        use_real_data = st.radio(
            "사용할 데이터:",
            ["실제 CSV 데이터 (50개 배송지)", "샘플 데이터 (8개 배송지)"],
            index=0,
            key="data_source"
        )
        st.session_state.use_real_data = use_real_data.startswith("실제")

    with col_data2:
        if st.session_state.use_real_data:
            st.info("📁 **실제 데이터 경로:**\n"
                   "- 배송지: `cosmos_opt/simple_cosmos_deliveries.csv`\n"
                   "- 트럭: `cosmos_opt/simple_cosmos_trucks.csv`\n"
                   "- 허브: `cosmos_opt/simple_cosmos_hub.csv`")
        else:
            st.info("🧪 **샘플 데이터:**\n"
                   "- 8개 배송지 (4개 시간슬롯)\n"
                   "- 3대 트럭\n"
                   "- 1개 허브")

    st.markdown("#### 🔧 알고리즘 선택")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔄 그리디 알고리즘**")
        greedy_enabled = st.checkbox("그리디 알고리즘", value=True, key="greedy_check")
        if greedy_enabled:
            st.write("• 각 단계에서 최적 선택")
            st.write("• 빠른 계산 속도")
            st.write("• 거리 기반 우선순위")

    with col2:
        st.markdown("**⏰ 시간슬롯 기반**")
        timeslot_enabled = st.checkbox("시간슬롯 기반", value=True, key="timeslot_check")
        if timeslot_enabled:
            st.write("• 시간창 제약 우선 고려")
            st.write("• 시간대별 그룹화")
            st.write("• 시간 효율성 극대화")

    with col3:
        st.markdown("**📍 최근접 이웃**")
        nn_enabled = st.checkbox("최근접 이웃", value=True, key="nn_check")
        if nn_enabled:
            st.write("• 가장 가까운 지점 우선")
            st.write("• 직관적 경로 생성")
            st.write("• 거리 최소화 전략")

    # 선택된 알고리즘 업데이트
    selected = []
    if greedy_enabled:
        selected.append('greedy')
    if timeslot_enabled:
        selected.append('time_slot')
    if nn_enabled:
        selected.append('nearest_neighbor')

    st.session_state.selected_algorithms = selected

    return selected

def create_optimization_controls():
    """최적화 제어 버튼들"""
    st.markdown("### 🎮 최적화 제어")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 최적화 시작", use_container_width=True, type="primary"):
            if not st.session_state.heuristic_running:
                if st.session_state.selected_algorithms:
                    start_optimization()
                    st.success("✅ 최적화가 시작되었습니다!")
                    st.rerun()
                else:
                    st.warning("⚠️ 최소 하나의 알고리즘을 선택해주세요.")
            else:
                st.warning("⚠️ 이미 최적화가 진행 중입니다.")

    with col2:
        if st.button("⏹️ 최적화 중지", use_container_width=True):
            if st.session_state.heuristic_running:
                st.session_state.heuristic_running = False
                st.info("⏹️ 최적화가 중지되었습니다.")
                st.rerun()

    with col3:
        if st.button("🔄 결과 초기화", use_container_width=True):
            reset_optimization_data()
            st.info("🔄 최적화 데이터가 초기화되었습니다.")
            st.rerun()

    with col4:
        if st.button("📊 상세 분석", use_container_width=True):
            if st.session_state.heuristic_results:
                st.info("📊 상세 분석을 표시합니다.")
            else:
                st.warning("⚠️ 분석할 결과가 없습니다.")

def start_optimization():
    """최적화 시작"""
    try:
        st.session_state.heuristic_running = True
        st.session_state.optimization_progress = 0

        # 최적화 객체 생성
        st.session_state.heuristic_optimizer = HeuristicVRPOptimizer()

        # 데이터 소스에 따른 메시지
        data_type = "실제 CSV 데이터 (50개 배송지)" if st.session_state.use_real_data else "샘플 데이터 (8개 배송지)"
        st.info(f"🔄 {data_type}로 휴리스틱 알고리즘을 실행하는 중...")

        # 모든 알고리즘 실행 (실제 데이터 사용 여부 전달)
        results = st.session_state.heuristic_optimizer.run_all_algorithms(
            use_real_data=st.session_state.use_real_data
        )

        st.session_state.heuristic_results = results
        st.session_state.heuristic_running = False
        st.session_state.optimization_progress = 100

        # 결과에 따른 성공 메시지
        num_deliveries = len(results['deliveries']) if results['deliveries'] else 0
        st.success(f"✅ 최적화가 완료되었습니다! ({num_deliveries}개 배송지 처리)")

    except Exception as e:
        st.error(f"❌ 최적화 중 오류가 발생했습니다: {str(e)}")
        st.session_state.heuristic_running = False

def reset_optimization_data():
    """최적화 데이터 초기화"""
    st.session_state.heuristic_results = None
    st.session_state.heuristic_running = False
    st.session_state.optimization_progress = 0

def create_optimization_status():
    """최적화 상태 표시"""
    st.markdown("### 📊 최적화 상태")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.session_state.heuristic_running:
            st.markdown('<p class="status-running">🟢 최적화 진행 중</p>', unsafe_allow_html=True)
        elif st.session_state.heuristic_results:
            st.markdown('<p class="status-completed">🔵 최적화 완료</p>', unsafe_allow_html=True)
        else:
            st.markdown("🔴 대기 중")

    with col2:
        if st.session_state.optimization_progress > 0:
            st.progress(st.session_state.optimization_progress / 100,
                       f"진행률: {st.session_state.optimization_progress}%")

def create_results_comparison():
    """결과 비교 섹션"""
    if not st.session_state.heuristic_results:
        st.info("📊 최적화를 실행하면 결과가 여기에 표시됩니다.")
        return

    st.markdown("### 📈 알고리즘 비교 결과")

    summary = st.session_state.heuristic_optimizer.get_solution_summary()

    if not summary:
        st.warning("⚠️ 결과 요약 데이터가 없습니다.")
        return

    # 최고 성능 알고리즘 찾기
    best_algorithm = max(summary.keys(), key=lambda x: summary[x]['score'])

    # 알고리즘별 카드 표시
    for algo_name, metrics in summary.items():
        card_class = "algorithm-card winner-card" if algo_name == best_algorithm else "algorithm-card"

        st.markdown(f"""
        <div class="{card_class}">
            <h4>{'🏆 ' if algo_name == best_algorithm else ''}
                {algo_name.replace('_', ' ').title()}
                {' (최우수)' if algo_name == best_algorithm else ''}
            </h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="🚛 사용 트럭",
                value=f"{metrics['trucks_used']}대"
            )

        with col2:
            st.metric(
                label="📦 완료 배송",
                value=f"{metrics['deliveries_completed']}개",
                delta=f"{metrics['completion_rate']:.1%}"
            )

        with col3:
            st.metric(
                label="📏 총 거리",
                value=f"{metrics['total_distance']:.0f}m"
            )

        with col4:
            st.metric(
                label="⏱️ 소요 시간",
                value=f"{metrics['total_time']:.0f}분"
            )

        with col5:
            st.metric(
                label="🎯 종합 점수",
                value=f"{metrics['score']:.1f}"
            )

        st.markdown("---")

def create_performance_charts():
    """성능 비교 차트"""
    if not st.session_state.heuristic_results:
        return

    st.markdown("### 📊 성능 비교 차트")

    summary = st.session_state.heuristic_optimizer.get_solution_summary()

    if not summary:
        return

    # 데이터 준비
    algorithms = list(summary.keys())
    algorithm_labels = [algo.replace('_', ' ').title() for algo in algorithms]

    trucks_used = [summary[algo]['trucks_used'] for algo in algorithms]
    completion_rates = [summary[algo]['completion_rate'] * 100 for algo in algorithms]
    total_distances = [summary[algo]['total_distance'] for algo in algorithms]
    scores = [summary[algo]['score'] for algo in algorithms]

    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('사용 트럭 수', '배송 완료율 (%)', '총 이동 거리 (m)', '종합 점수'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 색상 팔레트
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    # 사용 트럭 수
    fig.add_trace(
        go.Bar(x=algorithm_labels, y=trucks_used,
               name='사용 트럭', marker_color=colors[0], showlegend=False),
        row=1, col=1
    )

    # 배송 완료율
    fig.add_trace(
        go.Bar(x=algorithm_labels, y=completion_rates,
               name='완료율', marker_color=colors[1], showlegend=False),
        row=1, col=2
    )

    # 총 이동 거리
    fig.add_trace(
        go.Bar(x=algorithm_labels, y=total_distances,
               name='총 거리', marker_color=colors[2], showlegend=False),
        row=2, col=1
    )

    # 종합 점수
    fig.add_trace(
        go.Bar(x=algorithm_labels, y=scores,
               name='점수', marker_color=colors[3], showlegend=False),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="휴리스틱 알고리즘 성능 비교"
    )

    st.plotly_chart(fig, use_container_width=True)

def create_route_visualization():
    """경로 시각화"""
    if not st.session_state.heuristic_results:
        return

    st.markdown("### 🗺️ 최적 경로 시각화")

    results = st.session_state.heuristic_results

    if 'best_solution' not in results or not results['best_solution']:
        st.warning("⚠️ 최적 해가 없습니다.")
        return

    best_solution = results['best_solution']
    deliveries = results['deliveries']
    hub = results['hub']

    # 지도 데이터 준비
    fig = go.Figure()

    # 허브 표시
    fig.add_trace(go.Scatter(
        x=[hub.x], y=[hub.y],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='허브',
        text=['허브'],
        hovertemplate='<b>%{text}</b><br>좌표: (%{x}, %{y})<extra></extra>'
    ))

    # 배송지 표시
    delivery_x = [d.x for d in deliveries]
    delivery_y = [d.y for d in deliveries]
    delivery_text = [f"{d.delivery_id}<br>{d.area}" for d in deliveries]

    fig.add_trace(go.Scatter(
        x=delivery_x, y=delivery_y,
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='배송지',
        text=delivery_text,
        hovertemplate='<b>%{text}</b><br>좌표: (%{x}, %{y})<extra></extra>'
    ))

    # 경로 표시
    colors = ['green', 'orange', 'purple', 'brown', 'pink']

    for i, route_info in enumerate(best_solution['routes']):
        route_x = [hub.x]  # 허브에서 시작
        route_y = [hub.y]

        for delivery_idx in route_info.route:
            delivery = deliveries[delivery_idx]
            route_x.append(delivery.x)
            route_y.append(delivery.y)

        route_x.append(hub.x)  # 허브로 복귀
        route_y.append(hub.y)

        fig.add_trace(go.Scatter(
            x=route_x, y=route_y,
            mode='lines+markers',
            line=dict(color=colors[i % len(colors)], width=3),
            name=f'{route_info.truck_id} 경로',
            hovertemplate='<b>%{fullData.name}</b><extra></extra>'
        ))

    fig.update_layout(
        title=f"최적 경로 - {best_solution.get('best_algorithm', '알 수 없음').title()}",
        xaxis_title="X 좌표",
        yaxis_title="Y 좌표",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

def create_detailed_results():
    """상세 결과 표시"""
    if not st.session_state.heuristic_results:
        return

    st.markdown("### 📋 상세 결과")

    results = st.session_state.heuristic_results

    if 'best_solution' not in results or not results['best_solution']:
        return

    best_solution = results['best_solution']

    # 트럭별 경로 정보
    st.markdown("#### 🚛 트럭별 경로 정보")

    for route_info in best_solution['routes']:
        with st.expander(f"{route_info.truck_id} - {route_info.deliveries_count}개 배송지"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**기본 정보**")
                st.write(f"• 배송지 수: {route_info.deliveries_count}개")
                st.write(f"• 총 거리: {route_info.total_distance:.1f}m")
                st.write(f"• 소요 시간: {route_info.total_time:.1f}분")
                st.write(f"• 총 무게: {route_info.total_weight:.1f}kg")
                st.write(f"• 총 아이템: {route_info.total_items}개")

            with col2:
                st.write("**배송 경로**")
                deliveries = results['deliveries']
                route_text = "허브 → "

                for delivery_idx in route_info.route:
                    delivery = deliveries[delivery_idx]
                    route_text += f"{delivery.delivery_id}({delivery.area}) → "

                route_text += "허브"
                st.write(route_text)

def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_heuristic_session_state()

    # 메인 헤더
    st.markdown('<h1 class="main-header">🧠 휴리스틱 알고리즘 기반 VRPTW 해법</h1>', unsafe_allow_html=True)

    # 알고리즘 정보
    create_algorithm_info()

    # 알고리즘 선택
    selected_algorithms = create_algorithm_selection()

    st.markdown("---")

    # 최적화 제어
    create_optimization_controls()

    st.markdown("---")

    # 최적화 상태
    create_optimization_status()

    st.markdown("---")

    # 결과 비교
    create_results_comparison()

    st.markdown("---")

    # 성능 차트
    create_performance_charts()

    st.markdown("---")

    # 경로 시각화
    create_route_visualization()

    st.markdown("---")

    # 상세 결과
    create_detailed_results()

    # 자동 새로고침 (최적화 중일 때)
    if st.session_state.heuristic_running:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()