import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsões de Seleção - Random Forest", layout="wide")

# --- FUNÇÕES DE PROCESSAMENTO (CACHED) ---
@st.cache_data
def carregar_e_processar_dados():
    try:
        df = pd.read_csv('all_matches.csv')
    except FileNotFoundError:
        return None

    # Tratamento de Datas
    df['date'] = pd.to_datetime(df['date'])
    date = df['date'].dt.strftime('%Y-%m-%d')
    splitted_date = date.str.split('-')
    df['year'] = [int(x[0]) for x in splitted_date]

    # Agregação anual
    def get_annual_stats(df):
        home_stats = df[['date', 'year', 'home_team', 'home_score', 'away_score']].rename(
            columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'}
        )
        home_stats['result'] = np.where(home_stats['goals_for'] > home_stats['goals_against'], 3,
                                np.where(home_stats['goals_for'] == home_stats['goals_against'], 1, 0))

        away_stats = df[['date', 'year', 'away_team', 'away_score', 'home_score']].rename(
            columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'}
        )
        away_stats['result'] = np.where(away_stats['goals_for'] > away_stats['goals_against'], 3,
                                np.where(away_stats['goals_for'] == away_stats['goals_against'], 1, 0))

        all_stats = pd.concat([home_stats, away_stats])

        annual = all_stats.groupby(['team', 'year']).agg(
            games=('result', 'count'),
            points=('result', 'sum'),
            goals_scored=('goals_for', 'mean'),
            goals_conceded=('goals_against', 'mean')
        ).reset_index()

        annual['performance_score'] = annual['points'] / annual['games']
        return annual

    df_annual = get_annual_stats(df)

    # Criação de Lags (Features)
    df_annual = df_annual.sort_values(['team', 'year'])
    df_annual['prev_score_1'] = df_annual.groupby('team')['performance_score'].shift(1)
    df_annual['prev_score_2'] = df_annual.groupby('team')['performance_score'].shift(2)
    df_annual['prev_goals_1'] = df_annual.groupby('team')['goals_scored'].shift(1)
    
    # Remove linhas vazias geradas pelos shifts
    df_model = df_annual.dropna().copy()
    
    return df_model

# --- INTERFACE: BARRA LATERAL ---
st.sidebar.title("Configuração da Simulação")

# 1. Carrega os dados
df_features = carregar_e_processar_dados()

if df_features is None:
    st.error("ERRO: O arquivo 'all_matches.csv' não foi encontrado na pasta.")
    st.stop()

# 2. Seletor de Ano
anos_disponiveis = sorted(df_features['year'].unique())
ano_padrao = int(anos_disponiveis[-1]) 

target_year = st.sidebar.selectbox(
    "Escolha o Ano de Previsão:", 
    options=anos_disponiveis, 
    index=len(anos_disponiveis)-1
)

cutoff_year = target_year - 1

st.sidebar.markdown(f"""
---
**Resumo do Treino:**
* **Treinamento:** Dados até {cutoff_year}
* **Previsão:** Ano de {target_year}
""")

#run_btn = st.sidebar.button("Rodar Modelo Agora", type="primary")

# --- LÓGICA DE TREINAMENTO (ON-THE-FLY) ---
st.title(f"Previsão de Performance: {target_year}")

# Só roda se clicar ou se for a primeira vez
#if run_btn or 'df_results' not in st.session_state or st.session_state.get('last_year') != target_year:
if 'df_results' not in st.session_state or st.session_state.get('last_year') != target_year:

    with st.spinner(f'Treinando Random Forest com dados históricos até {cutoff_year}...'):
        
        features = ['prev_score_1', 'prev_score_2', 'prev_goals_1']
        target = 'performance_score'

        # Filtros de tempo
        X_train = df_features[df_features['year'] <= cutoff_year][features]
        y_train = df_features[df_features['year'] <= cutoff_year][target]

        test_data = df_features[df_features['year'] == target_year].copy()
        X_test = test_data[features]
        y_test = test_data[target]

        if not X_train.empty and not X_test.empty:
            # Treinamento
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            # Previsão e Erro
            test_data['predicted_score'] = rf.predict(X_test)
            test_data['abs_error'] = abs(test_data['performance_score'] - test_data['predicted_score'])
            
            # Salvar no session_state
            st.session_state['df_results'] = test_data
            st.session_state['last_year'] = target_year
            
        else:
            st.error("Dados insuficientes para realizar o treinamento neste ano.")
            st.stop()

# Recupera os dados calculados
df_res = st.session_state['df_results']

# --- DASHBOARD VISUAL ---

# 1. KPIs Globais
st.divider()
col1, col2, col3, col4 = st.columns(4)

mae = mean_absolute_error(df_res['performance_score'], df_res['predicted_score'])
rmse = np.sqrt(mean_squared_error(df_res['performance_score'], df_res['predicted_score']))
correlacao = df_res['performance_score'].corr(df_res['predicted_score'])

col1.metric(
    "Erro Médio (MAE)", 
    f"{mae:.4f}",
    help="Representa a média da diferença absoluta entre o valor Real e o Previsto."
)

col2.metric(
    "RMSE", 
    f"{rmse:.4f}",
    help="Similar ao MAE, porém penaliza erros grandes de forma mais severa, elevando ao quadrado cada Erro Médio."
)

col3.metric(
    "Correlação (R)", 
    f"{correlacao:.4f}",
    help="Varia de 0 a 1. Quanto mais próximo de 1, melhor o modelo entendeu a hierarquia de força das seleções, elencando elas como se fosse uma tabela."
)
col4.metric("Seleções Analisadas", len(df_res))

# 2. Seletor de Time
st.markdown("---")
times_no_ano = sorted(df_res['team'].unique())
col_sel_1, col_sel_2 = st.columns([1, 3])
with col_sel_1:
    time_selecionado = st.selectbox("Analisar Seleção Específica:", times_no_ano)

df_team = df_res[df_res['team'] == time_selecionado].iloc[0]

# 3. Métricas do Time
st.subheader(f"Detalhes: {time_selecionado}")
c1, c2, c3 = st.columns(3)
real = df_team['performance_score']
pred = df_team['predicted_score']
err = df_team['abs_error']

diferenca = pred - real 

c1.metric("Real", f"{real:.2f}")

# delta_color="normal":
# - Se diferenca > 0 (Previsão maior): Verde com seta para CIMA
# - Se diferenca < 0 (Previsão menor): Vermelho com seta para BAIXO
c2.metric(
    "Previsto (RF)", 
    f"{pred:.2f}", 
    delta=f"{diferenca:.2f}", 
    delta_color="normal",
    help="""
    Indica a diferença em relação ao Real:
    🟢 Seta Verde (Cima): O modelo calculou um valor maior que o real (Superestimou).
    🔴 Seta Vermelha (Baixo): O modelo calculou um valor menor que o real (Subestimou).
    """
)
c3.metric("Jogos no Ano", df_team['games'])

# 4. Gráficos e Tabelas
tab1, tab2 = st.tabs(["Comparativo Individual", "Dispersão Global & Histórico"])

with tab1:
    st.markdown("##### Comparativo: Real x Previsto")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=['Real', 'Previsto'], y=[real, pred], 
                             marker_color=['#2ca02c', '#1f77b4'], 
                             text=[f"{real:.2f}", f"{pred:.2f}"], textposition='auto'))
    fig_bar.update_layout(height=400, yaxis_range=[0, 3.2])
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # --- Parte 1: Gráfico Global ---
    st.markdown("##### 1. Comparação de todas as seleções (Global)")
    st.markdown("Cada ponto representa um país. A linha tracejada é o acerto obtido.")
    
    # Destacar o time selecionado no gráfico global com uma cor diferente ou tamanho maior
    colors = ['red' if t == time_selecionado else 'blue' for t in df_res['team']]
    sizes = [10 if t == time_selecionado else 5 for t in df_res['team']]
    
    fig_scatter = go.Figure()
    
    # Todos os times (azul)
    df_others = df_res[df_res['team'] != time_selecionado]
    fig_scatter.add_trace(go.Scatter(
        x=df_others['performance_score'], 
        y=df_others['predicted_score'],
        mode='markers',
        name='Outras Seleções',
        marker=dict(color='#A6C8FF', size=8, opacity=0.6),
        text=df_others['team']
    ))
    
    # Time selecionado (destaque)
    fig_scatter.add_trace(go.Scatter(
        x=[real], y=[pred],
        mode='markers',
        name=time_selecionado,
        marker=dict(color='red', size=12, line=dict(width=2, color='black')),
        text=[time_selecionado]
    ))

    # Linha Ideal
    fig_scatter.add_shape(type="line", line=dict(dash='dash', color='gray'), x0=0, y0=0, x1=3, y1=3)
    
    fig_scatter.update_layout(
        xaxis_title="Performance Real",
        yaxis_title="Performance Prevista",
        height=500
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # --- Parte 2: Tabela de Features ---
    st.markdown(f"##### 2. Dados anteriores da seleção: **{time_selecionado}**")

    features_data = {
        'Métrica': [
            'Performance No Ano Anterior',
            'Performance de 2 Anos Atrás',
            'Média De Gols Marcados No Ano Anterior'
        ],
        'Valor': [
            f"{df_team['prev_score_1']:.4f}",
            f"{df_team['prev_score_2']:.4f}",
            f"{df_team['prev_goals_1']:.2f}"
        ]
    }
    df_display_features = pd.DataFrame(features_data)
    
    df_display_features = df_display_features.set_index('Métrica')
    
    st.table(df_display_features)