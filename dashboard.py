import streamlit as st
import json
import pandas as pd
import plotly.express as px

def carregar_dados(caminho_json):
    with open(caminho_json, 'r') as f:
        return json.load(f)

def formatar_tempo(segundos):
    minutos = int(segundos // 60)
    segundos = int(segundos % 60)
    return f"{minutos:02d}:{segundos:02d}"

st.set_page_config(page_title="Dashboard de Tráfego", layout="wide")
st.title("📊 Dashboard de Tráfego com YOLOv11")

dados_raw = carregar_dados("resultados.json")

dados_por_segundo = {}
for entrada in dados_raw:
    t_seg = int(entrada["timestamp"])
    if t_seg not in dados_por_segundo:
        dados_por_segundo[t_seg] = entrada

tempos = sorted(dados_por_segundo.keys())

indice = st.slider("Escolha o tempo do vídeo (em segundos)", min_value=tempos[0], max_value=tempos[-1], step=1)
dados_atual = dados_por_segundo[indice]

tempo_atual = dados_atual["timestamp"]
areas_dado = dados_atual["areas"]

st.subheader(f"⏱️ Dados para o tempo: {formatar_tempo(tempo_atual)}")

col1, col2 = st.columns(2)
for i, (nome_area, info) in enumerate(areas_dado.items()):
    with (col1 if i % 2 == 0 else col2):
        st.markdown(f"### 🛣️ {nome_area}")
        st.metric("Veículos Totais", info["veiculos"])
        st.metric("Veículos Parados", info["parados"])
        if info["tipos"]:
            df_tipos = pd.DataFrame(list(info["tipos"].items()), columns=["Tipo", "Quantidade"])
            fig = px.bar(df_tipos, x="Tipo", y="Quantidade", title="Tipos de veículos")
            fig.update_yaxes(tickmode="linear", dtick=1)  
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum veículo detectado nesta área.")

with st.expander("📈 Evolução Temporal"):
    linhas = []
    for entrada in dados_raw:
        t = formatar_tempo(entrada["timestamp"])
        for area, info in entrada["areas"].items():
            linhas.append({
                "tempo": t,
                "área": area,
                "veículos": info["veiculos"],
                "parados": info["parados"]
            })

    df = pd.DataFrame(linhas)
    fig1 = px.line(df, x="tempo", y="veículos", color="área", title="Veículos ao longo do tempo")
    fig1.update_yaxes(tickmode="linear", dtick=1) 

    fig2 = px.line(df, x="tempo", y="parados", color="área", title="Veículos parados ao longo do tempo")
    fig2.update_yaxes(tickmode="linear", dtick=1)  

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
