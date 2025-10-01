# agent.py

import io
import contextlib
import traceback
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------- OpenAI SDK --------
from openai import OpenAI

_client = None

# ================= Inicialização =================
def initialize_openai_api(api_key: str):
    """Inicializa o cliente da OpenAI com a chave fornecida."""
    global _client
    _client = OpenAI(api_key=api_key)

# ================= Classificação de intenção =================
def classify_intent(user_prompt: str) -> str:
    """
    Usa IA para classificar intenção em "analysis" ou "chat".
    Retorna exatamente "analysis" ou "chat"; em erro, retorna "chat" como fallback seguro.
    """
    try:
        if _client is None:
            raise RuntimeError("OpenAI client não inicializado.")

        hint = (
            "Classifique a intenção do usuário somente como 'analysis' ou 'chat'. "
            "Use 'analysis' quando houver pedido para analisar dados, gerar gráfico ou código; "
            "caso contrário, 'chat'. "
            "Responda com apenas uma palavra: analysis ou chat."
        )

        out = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": hint},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        text = (out.choices[0].message.content or "").strip().lower()

        if "analysis" in text and "chat" not in text:
            return "analysis"
        if "chat" in text and "analysis" not in text:
            return "chat"

        # fallback simples por palavras-chave
        if any(
            k in user_prompt.lower()
            for k in [
                "analis", "plot", "gráfico", "grafico", "describe", "correla",
                "hist", "box", "scatter", "média", "media", "mediana", "moda",
                "variância", "variancia", "desvio padrão", "std"
            ]
        ):
            return "analysis"
        return "chat"

    except Exception:
        return "chat"

# ================= Resposta de chat =================
def get_chat_response(user_prompt: str) -> str:
    try:
        if _client is None:
            raise RuntimeError("OpenAI client não inicializado.")

        sys_msg = (
            "Você é o OpenAI Data Agent. Responda de forma clara e amigável. "
            "Se a pergunta não exigir análise de dados, seja breve."
        )

        out = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return (out.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Não foi possível gerar uma resposta de chat: {e}"

# ================= Geração de código de análise =================
def get_analysis_code(user_prompt: str, sample_markdown: str) -> str:
    """
    Retorna APENAS código Python que usa o DataFrame 'df' já existente.
    Deve imprimir alguma saída textual e, quando possível, um 'INSIGHT: ...' ao final.
    """
    try:
        if _client is None:
            raise RuntimeError("OpenAI client não inicializado.")

        sys_analyst = (
            "Você é um assistente especialista em EDA e visualização. "
            "Você gera somente código Python que será executado em Streamlit. "
            "O DataFrame alvo está disponível na variável df. "
            "Responda apenas com código Python válido (sem cercas de código). "
            "Quando chegar a uma conclusão, imprima uma linha iniciando com 'INSIGHT: ' "
            "(ex.: print('INSIGHT: ...')). "
            "Prefira plotly.express (st.plotly_chart(fig, use_container_width=True)) "
            "ou matplotlib/seaborn (st.pyplot(plt.gcf())). "
            "Nunca leia arquivos. Use apenas a variável df. "
            "Trate NaN antes de astype(int). "
            "Evite chained assignment; use df.loc[...]. "
            "Se fizer séries temporais, tente detectar colunas como "
            "['date','data','dt','time','timestamp'] (case-insensitive) "
            "e converter com pd.to_datetime(errors='coerce')."
            "Para desvio padrão/variância, não trate 'std'/'var' como nomes de colunas do df."
            " - Use: num = df.select_dtypes('number'); resumo = num.agg(['std','var']).T  (ou construa DataFrame com {'std':..., 'var':...})"
        )

        user_msg = (
            "Você irá gerar APENAS código Python (sem explicações, sem cercas ```).\n\n"
            "Regras:\n"
            "- O DataFrame já existe como df (pandas). NÃO leia arquivos.\n"
            "- Se converter tipos, trate NaN previamente (ex.: fillna, dropna) antes de astype(int).\n"
            "- Para Matplotlib/Seaborn: chame st.pyplot(plt.gcf()) após o plot.\n"
            "- Para Plotly: use st.plotly_chart(fig, use_container_width=True).\n"
            "- Evite chained assignment; use df.loc[...].\n"
            "- Para desvio padrão/variância, NÃO trate 'std'/'var' como colunas do df; "
            "  use df.select_dtypes('number') e agregue (ex.: num.agg(['std','var']).T) ou monte um DataFrame com {'std':..., 'var':...}.\n"
            "- Mostre prints com resultados e métricas relevantes (print()).\n"
            "- Ao final, imprima uma linha começando com 'INSIGHT:' resumindo a principal conclusão (máx. 140 caracteres).\n"
            "- Você pode criar novos DataFrames auxiliares se necessário.\n\n"
            f"Contexto (amostra df em Markdown):\n{sample_markdown}\n\n"
            f"Tarefa do usuário:\n{user_prompt}"
        )

        out = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_analyst},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        code = (out.choices[0].message.content or "").strip()

        # Remoção defensiva de cercas
        if code.startswith("```"):
            code = code.strip("`")
        if code.lstrip().lower().startswith("python"):
            code = code.split("\n", 1)[-1]

        return code
    except Exception:
        return (
            "print('Falha ao gerar código. Mostrando preview do df:')\n"
            "print(df.head())\n"
            "print('INSIGHT: Não foi possível gerar a análise solicitada; verifique sua conexão ou reformule o pedido.')\n"
        )

# ================= Execução do código =================
def execute_code(code: str, df: pd.DataFrame):
    """
    Executa o código gerado em um namespace controlado com acesso a:
    df, st, pd, plt, sns, px.
    Retorna (stdout, error_text). Gráficos são exibidos via Streamlit no próprio código.
    """
    safe_globals = {
        "__builtins__": {
            "__import__": __import__,  # essencial para importações
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "round": round,
            "print": print,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
        },
        "pd": pd,
        "st": st,
        "plt": plt,
        "sns": sns,
        "px": px,
    }

    safe_locals = {"df": df}
    stdout_buffer = io.StringIO()
    error_text = ""

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, safe_globals, safe_locals)
    except Exception:
        error_text = traceback.format_exc()

    return stdout_buffer.getvalue(), error_text
