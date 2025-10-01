# app.py — versão com visual aprimorado

import os
import io
import math
import pandas as pd
import streamlit as st
from agent import (
    initialize_openai_api,
    classify_intent,
    get_analysis_code,
    get_chat_response,
    execute_code,
)

# ========================= Configuração de página =========================
st.set_page_config(
    page_title='Agente de Análise de Dados',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ========================= Estilos (CSS) =========================
st.markdown("""
<style>
:root {
  --bg: #0f1115;
  --panel: #151922;
  --panel-2: #1a1f2b;
  --text: #e7eaf0;
  --muted: #b4bdd1;
  --primary: #7c9aff;    /* acento */
  --success: #2ecc71;
  --warning: #f1c40f;
  --danger:  #ff6b6b;
  --card-radius: 14px;
  --shadow: 0 10px 30px rgba(0,0,0,0.25);
}
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #0f1115 0%, #0f1115 30%, #0b0d13 100%) !important;
  color: var(--text) !important;
}
section[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
.block-container {
  padding-top: 1rem !important;
}

.hero {
  margin-top: 30px;
  border-radius: 18px;
  padding: 22px 24px;
  background: radial-gradient(1200px 400px at 10% -10%, rgba(124,154,255,0.18), transparent),
              linear-gradient(180deg, #161b25, #121722);
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: var(--shadow);
}
.hero h1 {
  font-size: 1.75rem;
  line-height: 1.15;
  margin: 0;
}
.hero p {
  margin: 6px 0 0 0;
  color: var(--muted);
}

.badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.badge {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(124,154,255,0.12);
  border: 1px solid rgba(124,154,255,0.25);
  font-size: 12px;
  color: var(--text);
}

.card {
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: var(--card-radius);
  padding: 18px;
  box-shadow: var(--shadow);
}
.card h3 {
  margin-top: 0;
  font-size: 1rem;
}

.metric {
  display: grid;
  gap: 6px;
}
.metric .label {
  color: var(--muted);
  font-size: 12px;
}
.metric .value {
  font-size: 1.25rem;
  font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}
.stTabs [data-baseweb="tab"] {
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.06);
  border-bottom: 2px solid transparent;
  border-radius: 10px 10px 0 0;
  padding: 10px 14px;
  color: var(--text);
}
.stTabs [aria-selected="true"] {
  border-bottom: 2px solid var(--primary) !important;
}

.kicker {
  color: var(--muted);
  font-size: 12px;
  letter-spacing: .08em;
  text-transform: uppercase;
}

pre, code, .stCodeBlock {
  background: var(--panel-2) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* chat bubbles */
[data-testid="stChatMessage"] {
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 14px;
  box-shadow: var(--shadow);
}
[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) {
  background: linear-gradient(180deg, #151a25, #141925);
}
[data-testid="stChatMessage"]:has(div[aria-label="user"]) {
  background: linear-gradient(180deg, #19202c, #161c28);
}

/* file/dataframe containers */
[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* buttons */
.stButton>button, .stDownloadButton>button {
  background: linear-gradient(180deg, #7c9aff, #6488ff);
  color: #0b0d13;
  border: none;
  border-radius: 12px;
  padding: 10px 14px;
  font-weight: 700;
  box-shadow: 0 6px 18px rgba(124,154,255,0.35);
}
.stButton>button:hover, .stDownloadButton>button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}

/* expander */
.streamlit-expanderHeader {
  background: var(--panel) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* footnote */
.footnote {
  color: var(--muted);
  font-size: 12px;
  text-align: right;
  padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ========================= Estado da sessão =========================
st.session_state.setdefault('df', None)
st.session_state.setdefault('file_meta', None)  # (name, size)
st.session_state.setdefault('sample_rendered', False)
st.session_state.setdefault('chat_history', [])
st.session_state.setdefault('insights', [])

# ========================= Sidebar =========================
with st.sidebar:
    st.markdown('### ⚙️ Configurações')
    st.caption('Defina sua chave de API da OpenAI como variável de ambiente `OPENAI_API_KEY`.')

    uploaded_file = st.file_uploader('📥 Envie um CSV', type=['csv'])

    st.markdown('---')
    st.markdown('#### Aparência')
    show_schema = st.toggle('Mostrar aba **Esquema**', value=True)
    show_code_expander = st.toggle('Mostrar expander de código', value=True)
    sample_rows = st.slider('Linhas da amostra', 5, 50, 10, 5)
    st.markdown('---')
    st.caption('💡 Dica: peça coisas como _"histograma de Amount"_ ou _"correlação entre X e Y"_.')

# ========================= Chave de API =========================
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error('A chave de API da OpenAI não foi encontrada. Defina `OPENAI_API_KEY`.')
    st.stop()
else:
    initialize_openai_api(api_key)

# ========================= Funções auxiliares =========================
def human_size(nbytes: int | None) -> str:
    if not nbytes and nbytes != 0:
        return '—'
    if nbytes == 0:
        return '0 B'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    idx = int(math.floor(math.log(nbytes, 1024)))
    p = math.pow(1024, idx)
    s = round(nbytes / p, 2)
    return f'{s} {units[idx]}'

def get_meta(f):
    if f is None:
        return None
    try:
        size = f.size
    except Exception:
        try:
            size = len(f.getbuffer())
        except Exception:
            size = None
    return (f.name, size)

# ========================= Hero / Header =========================
st.markdown("""
<div class="hero">
  <div class="kicker">OpenAI Data Agent</div>
  <h1>🎲 Análise Explorátoria guiada por IA</h1>
  <p>Envie um CSV, faça perguntas em linguagem natural, gere gráficos e salve conclusões automaticamente.</p>
  <div class="badges">
    <span class="badge">GPT-4o-mini</span>
    <span class="badge">Streamlit</span>
    <span class="badge">Plotly / Matplotlib</span>
    <span class="badge">Memória de INSIGHTs</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.write('')

# ========================= Upload & DF handling =========================
new_meta = get_meta(uploaded_file)

if uploaded_file is not None:
    if st.session_state.file_meta != new_meta:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_meta = new_meta
            st.session_state.sample_rendered = False
            st.session_state.chat_history = [
                {'role': 'assistant', 'content': '📁 Arquivo recebido! Pronto para analisar. Faça uma pergunta ou peça um gráfico.'}
            ]
            st.session_state.insights = []
            st.toast('CSV carregado com sucesso!', icon='✅')
        except Exception as e:
            st.error(f'Falha ao ler CSV: {e}')
            st.stop()
else:
    st.info('Envie um CSV na barra lateral para começar.')
    st.stop()

df = st.session_state.df

# ========================= Cards de métricas =========================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="card metric"><div class="label">Arquivo</div>'
                f'<div class="value">{st.session_state.file_meta[0]}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card metric"><div class="label">Tamanho</div>'
                f'<div class="value">{human_size(st.session_state.file_meta[1])}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card metric"><div class="label">Linhas</div>'
                f'<div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="card metric"><div class="label">Colunas</div>'
                f'<div class="value">{df.shape[1]}</div></div>', unsafe_allow_html=True)

st.write('')

# ========================= Abas de conteúdo =========================
tabs = ['🔎 Amostra']
if show_schema:
    tabs.append('🧬 Esquema')
tabs.append('🧠 Conclusões')
tab_objs = st.tabs(tabs)

# Amostra
with tab_objs[0]:
    st.markdown('#### Amostra do DataFrame')
    st.dataframe(df.head(sample_rows), use_container_width=True)

# Esquema (dtypes / nulos)
if show_schema:
    with tab_objs[1]:
        st.markdown('#### Esquema e Qualidade')
        colL, colR = st.columns([1.2, 1])
        with colL:
            dtypes_df = pd.DataFrame({
                'coluna': df.columns,
                'dtype': [str(t) for t in df.dtypes.values],
                'n_nulos': df.isna().sum().values,
                'pct_nulos': (df.isna().mean().round(4) * 100).values
            })
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
        with colR:
            st.markdown('**Resumo**')
            st.write(f'- Colunas numéricas: **{df.select_dtypes("number").shape[1]}**')
            st.write(f'- Colunas categóricas: **{df.select_dtypes("object").shape[1]}**')
            st.write(f'- Colunas de data: **{df.select_dtypes("datetime").shape[1]}** (verificar conversão)')
            top_nulls = dtypes_df.sort_values('pct_nulos', ascending=False).head(5)[['coluna','pct_nulos']]
            st.write('**Top 5 % nulos:**')
            st.dataframe(top_nulls, use_container_width=True, hide_index=True)

# Conclusões
with tab_objs[-1]:
    st.markdown('#### Memória de conclusões')
    if st.session_state.insights:
        for i, ins in enumerate(st.session_state.insights, 1):
            st.markdown(f'- {ins}')
    else:
        st.caption('Sem conclusões registradas ainda. Peça análises e gráficos.')

st.write('')

# ========================= Conversa =========================
st.markdown('### 💬 Conversa')

# Render histórico
for msg in st.session_state.chat_history:
    role = msg['role']
    avatar = 'assistant' if role == 'assistant' else 'user'
    with st.chat_message(role, avatar=None):
        st.markdown(msg['content'])

# Entrada
user_input = st.chat_input('Pergunte algo (ex.: "histograma de Amount", "tendência temporal de X", "quais conclusões?").')

def push_assistant(text: str):
    st.session_state.chat_history.append({'role': 'assistant', 'content': text})
    with st.chat_message('assistant'):
        st.markdown(text)

if user_input:
    # Mostra pergunta
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Comandos rápidos de conclusões
    normalized = user_input.strip().lower()
    if normalized in {
        'conclusoes', 'conclusões', 'resumo', 'insights',
        'quais conclusoes', 'quais conclusões até agora?',
        'quais conclusões até agora', 'quais conclusões?'
    }:
        if st.session_state.insights:
            bullets = '\n'.join([f'- {i}' for i in st.session_state.insights])
            push_assistant(f'Aqui estão as conclusões registradas até agora:\n\n{bullets}')
        else:
            push_assistant('Ainda não há conclusões registradas. Solicite alguma análise ou gráfico para começarmos.')
        st.stop()

    # Classificação de intenção
    intent = classify_intent(user_input)
    if intent not in {'analysis', 'chat'}:
        push_assistant('Não entendi a intenção. Reformule pedindo uma análise dos dados ou continue a conversa.')
        st.stop()

    # Roteamento
    if intent == 'chat':
        reply = get_chat_response(user_input)
        push_assistant(reply)
        st.stop()

    # === intent == analysis ===
    sample_text = df.head(20).to_markdown(index=False)
    code = get_analysis_code(user_input, sample_text)

    # Corrige linhas "INSIGHT:" sem print()
    fixed_lines = []
    for line in code.splitlines():
        if line.strip().startswith('INSIGHT:'):
            insight_txt = line.strip().replace('"', "'")
            fixed_lines.append(f'print("{insight_txt}")')
        else:
            fixed_lines.append(line)
    code = '\n'.join(fixed_lines)

    if show_code_expander:
        with st.expander('🧩 Código gerado pela IA', expanded=False):
            st.code(code, language='python')

    stdout_text, error_text = execute_code(code, df)

    if error_text:
        push_assistant(f'Ocorreu um erro na execução:\n\n```\n{error_text}\n```')
        st.stop()

    if stdout_text.strip():
        with st.chat_message('assistant'):
            st.markdown('**Resultado da análise:**')
            st.code(stdout_text)

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': '✅ Análise executada e gráficos/renderizações (se houver) exibidos acima.'
        })

        # Extrai e registra INSIGHTs
        new_insights = []
        for line in stdout_text.splitlines():
            line_stripped = line.strip()
            if line_stripped.upper().startswith('INSIGHT:') or 'INSIGHT:' in line_stripped.upper():
                if ':' in line_stripped:
                    insight = line_stripped.split(':', 1)[1].strip()
                    if insight:
                        new_insights.append(insight)

        if new_insights:
            st.session_state.insights.extend(new_insights)
            st.toast(f'{len(new_insights)} conclusão(ões) registrada(s) 📌', icon='✍️')
            push_assistant('Memória atualizada com novas conclusões.')
        else:
            push_assistant('Análise executada com sucesso.')
