import streamlit as st
import json
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="Dashboard de Tráfego", layout="wide")
st.title("📊 Dashboard de Tráfego com YOLOv11")

def carregar_dados(caminho_json):
    try:
        with open(caminho_json, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar JSON: {e}")
        return []

def formatar_tempo(timestamp):
    if isinstance(timestamp, str):
        try:
            return timestamp
        except:
            return timestamp
    elif isinstance(timestamp, (int, float)):
        # Se for um número grande, converte como epoch
        if timestamp > 1e9:
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = datetime.utcfromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(timestamp)

# Lista arquivos válidos
arquivos_json = sorted([f for f in os.listdir() if f.startswith("resultados") and f.endswith(".json")])

if not arquivos_json:
    st.error("Nenhum arquivo JSON encontrado.")
    st.stop()

arquivo_selecionado = st.selectbox("Selecione o arquivo de resultados:", arquivos_json)
dados_raw = carregar_dados(arquivo_selecionado)

if not dados_raw:
    st.warning("O arquivo está vazio ou inválido.")
    st.stop()

# Processa dados
dados_por_tempo = {}
for entrada in dados_raw:
    if "timestamp" not in entrada or "areas" not in entrada:
        continue
    tempo = formatar_tempo(entrada["timestamp"])
    dados_por_tempo[tempo] = entrada

tempos = sorted(dados_por_tempo.keys())

if not tempos:
    st.warning("Nenhum dado de tempo encontrado.")
    st.stop()

indice = st.slider("Escolha o tempo", 0, len(tempos) - 1, 0)
tempo_selecionado = tempos[indice]
dados_atual = dados_por_tempo[tempo_selecionado]
areas_dado = dados_atual["areas"]

st.subheader(f"⏱️ Dados para o tempo: {tempo_selecionado}")

col1, col2 = st.columns(2)
for i, (nome_area, info) in enumerate(areas_dado.items()):
    with (col1 if i % 2 == 0 else col2):
        st.markdown(f"### 🛣️ {nome_area}")
        st.metric("Veículos Totais", info.get("veiculos", 0))
        st.metric("Veículos Parados", info.get("parados", 0))

        if info.get("tipos"):
            df_tipos = pd.DataFrame(list(info["tipos"].items()), columns=["Tipo", "Quantidade"])
            fig = px.bar(df_tipos, x="Tipo", y="Quantidade", title="Tipos de veículos")
            fig.update_yaxes(tickmode="linear", dtick=1)
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{nome_area}")
        else:
            st.info("Nenhum veículo detectado nesta área.")

# GRÁFICOS TEMPORAIS
with st.expander("📈 Evolução Temporal"):
    linhas = []
    for entrada in dados_raw:
        try:
            tempo = formatar_tempo(entrada["timestamp"])
            for area, info in entrada["areas"].items():
                linhas.append({
                    "tempo": tempo,
                    "área": area,
                    "veículos": info["veiculos"],
                    "parados": info["parados"]
                })
        except Exception:
            continue

    df = pd.DataFrame(linhas)
    if not df.empty:
        fig1 = px.line(df, x="tempo", y="veículos", color="área", title="Veículos ao longo do tempo")
        fig1.update_yaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df, x="tempo", y="parados", color="área", title="Veículos parados ao longo do tempo")
        fig2.update_yaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar gráficos.")
