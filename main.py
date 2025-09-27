import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import sys

# 페이지 설정
st.set_page_config(
    page_title="VRPTW 솔루션 비교 대시보드",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
        border-left: 4px solid #F18F01;
        padding-left: 1rem;
    }

    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }

    .status-online {
        color: #28a745;
        font-weight: bold;
    }

    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }

    .algorithm-card {
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .algorithm-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }

    .comparison-table {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """세션 상태 초기화"""
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'current_episode' not in st.session_state:
        st.session_state.current_episode = 0
    if 'rl_performance' not in st.session_state:
        st.session_state.rl_performance = {
            'rewards': [],
            'completion_times': [],
            'distances': []
        }
    if 'heuristic_performance' not in st.session_state:
        st.session_state.heuristic_performance = {
            'completion_times': [],
            'distances': [],
            'success_rates': []
        }

def create_sidebar():
    """사이드바 생성"""
    st.sidebar.title("🚚 VRPTW 대시보드")
    st.sidebar.markdown("---")

    # SUMO 연결 상태
    st.sidebar.subheader("🔗 SUMO 연결 상태")
    sumo_status = st.sidebar.empty()

    # 임시로 연결 상태 시뮬레이션
    if st.session_state.simulation_running:
        sumo_status.markdown('<p class="status-online">🟢 연결됨</p>', unsafe_allow_html=True)
    else:
        sumo_status.markdown('<p class="status-offline">🔴 연결 안됨</p>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # 시뮬레이션 제어
    st.sidebar.subheader("⚙️ 시뮬레이션 제어")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ 시작", use_container_width=True):
            st.session_state.simulation_running = True
            st.rerun()

    with col2:
        if st.button("⏹️ 정지", use_container_width=True):
            st.session_state.simulation_running = False
            st.rerun()

    # 설정 옵션
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ 설정")

    simulation_speed = st.sidebar.slider("시뮬레이션 속도", 0.1, 3.0, 1.0, 0.1)
    update_interval = st.sidebar.selectbox("업데이트 간격", [1, 2, 5, 10], index=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 표시 옵션")
    show_real_time = st.sidebar.checkbox("실시간 업데이트", True)
    show_statistics = st.sidebar.checkbox("통계 정보", True)
    show_comparison = st.sidebar.checkbox("비교 차트", True)

    return {
        'simulation_speed': simulation_speed,
        'update_interval': update_interval,
        'show_real_time': show_real_time,
        'show_statistics': show_statistics,
        'show_comparison': show_comparison
    }

def create_kpi_dashboard():
    """주요 KPI 대시보드 생성"""
    st.markdown('<h2 class="sub-header">📊 실시간 성능 지표</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # 임시 데이터 생성
    if st.session_state.simulation_running:
        delivery_rate = np.random.uniform(75, 95)
        avg_time = np.random.uniform(120, 180)
        fuel_efficiency = np.random.uniform(8.5, 12.0)
        customer_satisfaction = np.random.uniform(4.2, 4.8)
    else:
        delivery_rate = 0
        avg_time = 0
        fuel_efficiency = 0
        customer_satisfaction = 0

    with col1:
        st.metric(
            label="배송 완료율",
            value=f"{delivery_rate:.1f}%",
            delta=f"{np.random.uniform(-2, 3):.1f}%"
        )

    with col2:
        st.metric(
            label="평균 배송 시간",
            value=f"{avg_time:.0f}분",
            delta=f"{np.random.uniform(-10, 5):.0f}분"
        )

    with col3:
        st.metric(
            label="연료 효율성",
            value=f"{fuel_efficiency:.1f}km/L",
            delta=f"{np.random.uniform(-0.5, 0.8):.1f}km/L"
        )

    with col4:
        st.metric(
            label="고객 만족도",
            value=f"{customer_satisfaction:.1f}★",
            delta=f"{np.random.uniform(-0.2, 0.3):.1f}★"
        )

def create_algorithm_comparison():
    """알고리즘 비교 섹션"""
    st.markdown('<h2 class="sub-header">🤖 vs 🧠 알고리즘 비교</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="algorithm-card">
            <h3>🤖 강화학습 (DQN)</h3>
            <p><strong>장점:</strong></p>
            <ul>
                <li>동적 환경 적응성 우수</li>
                <li>장기적 최적화 가능</li>
                <li>학습을 통한 성능 개선</li>
            </ul>
            <p><strong>현재 상태:</strong></p>
            <p>에피소드: {episode} / 1000</p>
            <p>평균 보상: {reward:.2f}</p>
        </div>
        """.format(
            episode=st.session_state.current_episode,
            reward=np.random.uniform(-100, 500) if st.session_state.simulation_running else 0
        ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="algorithm-card">
            <h3>🧠 휴리스틱 방법</h3>
            <p><strong>장점:</strong></p>
            <ul>
                <li>빠른 실행 속도</li>
                <li>안정적인 성능</li>
                <li>해석 가능한 결과</li>
            </ul>
            <p><strong>현재 상태:</strong></p>
            <p>실행 시간: {time:.1f}초</p>
            <p>최적화율: {opt:.1f}%</p>
        </div>
        """.format(
            time=np.random.uniform(0.5, 2.0) if st.session_state.simulation_running else 0,
            opt=np.random.uniform(85, 95) if st.session_state.simulation_running else 0
        ), unsafe_allow_html=True)

def create_live_charts():
    """실시간 차트 생성"""
    st.markdown('<h2 class="sub-header">📈 실시간 성능 차트</h2>', unsafe_allow_html=True)

    if st.session_state.simulation_running:
        # 임시 데이터 생성
        episodes = list(range(max(0, st.session_state.current_episode - 50), st.session_state.current_episode + 1))
        rl_rewards = [np.random.uniform(-50, 300) for _ in episodes]
        heuristic_scores = [np.random.uniform(200, 400) for _ in episodes]

        # 성능 비교 차트
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=episodes,
            y=rl_rewards,
            mode='lines+markers',
            name='강화학습 보상',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=episodes,
            y=heuristic_scores,
            mode='lines+markers',
            name='휴리스틱 점수',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="실시간 성능 비교",
            xaxis_title="에피소드/실행 횟수",
            yaxis_title="성능 점수",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # 에피소드 카운터 업데이트
        if st.session_state.current_episode < 1000:
            st.session_state.current_episode += 1

def create_current_status():
    """현재 상태 정보"""
    st.markdown('<h2 class="sub-header">🚦 현재 시뮬레이션 상태</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.simulation_running:
            st.success("✅ 시뮬레이션 실행 중")
            st.info(f"🔄 진행률: {(st.session_state.current_episode / 1000 * 100):.1f}%")
        else:
            st.error("❌ 시뮬레이션 정지됨")

    with col2:
        active_vehicles = np.random.randint(3, 8) if st.session_state.simulation_running else 0
        st.metric("활성 차량 수", active_vehicles)

    with col3:
        pending_deliveries = np.random.randint(10, 50) if st.session_state.simulation_running else 0
        st.metric("대기 중인 배송", pending_deliveries)

def main():
    """메인 함수"""
    initialize_session_state()

    # 메인 헤더
    st.markdown('<h1 class="main-header">🚚 VRPTW 솔루션 비교 대시보드</h1>', unsafe_allow_html=True)

    # 사이드바 생성
    settings = create_sidebar()

    # 프로젝트 소개
    st.markdown("""
    ### 📋 프로젝트 개요
    이 대시보드는 **VRPTW(Vehicle Routing Problem with Time Windows)** 문제를 해결하는
    **강화학습**과 **휴리스틱** 방법론을 실시간으로 비교 분석할 수 있는 플랫폼입니다.

    **SUMO(Simulation of Urban MObility)** 시뮬레이터와 연동하여 실제 도로 환경에서의
    성능을 측정하고 시각화합니다.
    """)

    st.markdown("---")

    # KPI 대시보드
    create_kpi_dashboard()

    st.markdown("---")

    # 현재 상태
    create_current_status()

    st.markdown("---")

    # 알고리즘 비교
    create_algorithm_comparison()

    st.markdown("---")

    # 실시간 차트 (설정에 따라 표시)
    if settings['show_real_time'] and settings['show_comparison']:
        create_live_charts()

    # 자동 새로고침 (시뮬레이션이 실행 중일 때)
    if st.session_state.simulation_running and settings['show_real_time']:
        time.sleep(settings['update_interval'])
        st.rerun()

if __name__ == "__main__":
    main()