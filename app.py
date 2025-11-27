
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import inflection
import io

# =============================================================================
# CONFIGURAÇÕES DA PÁGINA E ESTILO
# =============================================================================
st.set_page_config(
    page_title="Análise de Propensão de Pagamento",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Resultados da Análise de Propensão")

# =============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO (DO NOTEBOOK)
# =============================================================================

def clean_column_names(df):
    """Padroniza os nomes das colunas para snake_case."""
    df_copy = df.copy()
    df_copy.columns = [inflection.underscore(c).strip() for c in df_copy.columns]
    return df_copy

def engineer_features_master(df):
    """Executa a engenharia de features de forma consistente (V3)."""
    df_eng = df.copy()
    
    # Conversão de colunas monetárias
    cols_to_convert = [
        'total_financiado', 'saldo_vencido', 'recebido', 
        'saldo_vencido_com_juros', 'total_pago_com_juros', 
        'vencidos_sem_juros_tmb', 'recebido_sem_juros_tmb'
    ]
    
    for col in cols_to_convert:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].astype(str)
            df_eng[col] = df_eng[col].str.replace('R$', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).str.strip()
            df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce')

    # Correção de tipos (V3)
    if 'pdd' in df_eng.columns:
        df_eng['pdd'] = df_eng['pdd'].astype(str)

    # Features de Score e Idade (V3)
    df_eng['score_x_idade'] = df_eng['score'] * df_eng['idade']
    df_eng['score_ao_quadrado'] = df_eng['score'] ** 2
    df_eng['score_por_idade'] = df_eng['score'] / (df_eng['idade'] + 1)
    
    # Faixa etária (V3 - bins atualizados)
    df_eng['faixa_etaria'] = pd.cut(
        df_eng['idade'], 
        bins=[0, 25, 35, 45, 55, 65, 120], 
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    ).astype(str)
    
    # Faixa de score (V3 - nova feature)
    df_eng['faixa_score'] = pd.cut(
        df_eng['score'],
        bins=[0, 300, 500, 700, 850, 1000],
        labels=['Muito_Baixo', 'Baixo', 'Medio', 'Bom', 'Excelente']
    ).astype(str)
    
    # Região
    mapa_regioes = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }
    df_eng['regiao'] = df_eng['endereco_estado'].str.upper().str.strip().map(mapa_regioes).fillna('Outra')
    
    # Features financeiras (V3 - corrigidas)
    df_eng['valor_medio_parcela'] = np.where(
        df_eng['quantidade_parcelas'] > 0,
        df_eng['total_financiado'] / df_eng['quantidade_parcelas'],
        0
    )
    
    df_eng['percentual_vencido'] = np.where(
        df_eng['total_financiado'] > 0,
        df_eng['saldo_vencido'] / df_eng['total_financiado'],
        0
    )
    
    df_eng['percentual_pago'] = np.where(
        df_eng['total_financiado'] > 0,
        df_eng['recebido'] / df_eng['total_financiado'],
        0
    )
    
    df_eng['percentual_parcelas_vencidas'] = np.where(
        df_eng['quantidade_parcelas'] > 0,
        df_eng['quantidade_parcelas_vencidas'] / df_eng['quantidade_parcelas'],
        0
    )
    
    # Severidade do atraso (V3)
    df_eng['severidade_atraso'] = df_eng['dias_em_atraso'] * df_eng['percentual_vencido']
    
    # Indicadores binários (V3)
    df_eng['ja_pagou_algo'] = (df_eng['recebido'] > 0).astype(int)
    df_eng['atraso_severo'] = (df_eng['dias_em_atraso'] > 90).astype(int)
    df_eng['alto_comprometimento'] = (df_eng['percentual_vencido'] > 0.5).astype(int)
    
    # Features de interação (V3)
    df_eng['score_x_percentual_pago'] = df_eng['score'] * df_eng['percentual_pago']
    df_eng['score_x_atraso'] = df_eng['score'] / (df_eng['dias_em_atraso'] + 1)
    
    # Log de valores monetários (V3)
    df_eng['total_financiado_log'] = np.log1p(df_eng['total_financiado'].clip(lower=0))
    df_eng['saldo_vencido_log'] = np.log1p(df_eng['saldo_vencido'].clip(lower=0))
    
    # Limpeza final
    df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    numeric_cols = df_eng.select_dtypes(include=np.number).columns
    df_eng[numeric_cols] = df_eng[numeric_cols].fillna(0)
    
    object_cols = df_eng.select_dtypes(include=['object']).columns
    df_eng[object_cols] = df_eng[object_cols].fillna('Desconhecido')
    
    return df_eng

# =============================================================================
# CARREGAMENTO DOS MODELOS (COM CACHE)
# =============================================================================

@st.cache_resource
def load_models():
    """Carrega os modelos V3 de ML, usando cache para performance."""
    models = {}
    # Caminhos dos modelos V3 (ajuste conforme o diretório dos seus modelos)
    model_files = {
        "Alto": "modelo_Alto.joblib",
        "Médio": "modelo_Médio.joblib",
        "Baixo": "modelo_Baixo.joblib"
    }
    for risk, filename in model_files.items():
        try:
            model_data = joblib.load(filename)
            # V3: modelo é um dicionário com 'pipeline', 'optimal_threshold', 'features'
            models[risk] = model_data
            st.sidebar.success(f"✅ Modelo {risk} carregado (threshold: {model_data.get('optimal_threshold', 0.5):.2f})")
        except FileNotFoundError:
            st.sidebar.warning(f"⚠️ Modelo '{filename}' não encontrado.")
    
    if not models:
        st.error("ERRO CRÍTICO: Nenhum modelo foi carregado. Verifique os arquivos .joblib.")
        return None
    return models

models = load_models()

# =============================================================================
# LÓGICA DA APLICAÇÃO
# =============================================================================

def run_prediction(df, models_dict, risk_category=None):
    """
    Prepara os dados, executa as predições usando modelos V3 com threshold otimizado.
    
    :param df: DataFrame de entrada.
    :param models_dict: Dicionário com os modelos V3 carregados.
    :param risk_category: (Opcional) Se especificado, processa o DF inteiro para essa categoria.
    :return: DataFrame com os resultados ou None se houver erro.
    """
    st.info("Iniciando preparação dos dados...")
    
    # 1. Padroniza nomes das colunas
    df_clean = clean_column_names(df)
    
    # 2. Validação de colunas essenciais
    essential_cols = ['idade', 'score', 'endereco_estado', 'total_financiado', 
                      'saldo_vencido', 'recebido', 'quantidade_parcelas', 'dias_em_atraso']
    missing_cols = [col for col in essential_cols if col not in df_clean.columns]
    if missing_cols:
        st.error(f"Colunas essenciais não encontradas: **{', '.join(missing_cols)}**")
        return None

    # 3. Engenharia de Features (V3)
    df_featured = engineer_features_master(df_clean)
    st.write("✅ Engenharia de features V3 concluída.")

    # 4. Lógica de Predição com modelos V3
    st.info("Executando modelos de Machine Learning V3...")
    
    results_list = []
    
    if risk_category:  # Modo de risco específico
        st.write(f"Aplicando o modelo de risco **'{risk_category}'** para todo o arquivo.")
        model_data = models_dict.get(risk_category)
        if model_data:
            pipeline = model_data['pipeline']
            threshold = model_data.get('optimal_threshold', 0.5)
            features = model_data.get('features', [])
            
            st.write(f"   → Usando threshold otimizado: {threshold:.2f}")
            
            # Verifica e cria features faltantes
            for f in features:
                if f not in df_featured.columns:
                    df_featured[f] = 0 if f not in ['status_cobranca', 'faixa_etaria', 'faixa_score', 'regiao', 'segmento', 'categoria_risco_score', 'modalidade', 'pdd'] else 'Desconhecido'
            
            X = df_featured[features]
            probabilities = pipeline.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            temp_df = df.copy()
            temp_df['previsao'] = predictions
            temp_df['prob_pagar'] = probabilities * 100
            temp_df['threshold_usado'] = threshold
            results_list.append(temp_df)
        else:
            st.error(f"Modelo para risco '{risk_category}' não encontrado.")
            return None
            
    else:  # Modo de roteamento automático
        if 'categoria_risco_score' not in df_featured.columns:
            st.error("Coluna **'categoria_risco_score'** necessária para roteamento automático.")
            return None
        
        # Mapeia valores para garantir correspondência
        risk_mapping = {'Alto': 'Alto', 'Médio': 'Medio', 'Medio': 'Medio', 'Baixo': 'Baixo'}
        
        for risk_level, model_data in models_dict.items():
            # Normaliza o nome do risco
            risk_values = [risk_level, risk_mapping.get(risk_level, risk_level)]
            df_subset_idx = df_featured[df_featured['categoria_risco_score'].isin(risk_values)].index
            
            if len(df_subset_idx) > 0:
                pipeline = model_data['pipeline']
                threshold = model_data.get('optimal_threshold', 0.5)
                features = model_data.get('features', [])
                
                st.write(f"  - Processando {len(df_subset_idx)} registros para **'{risk_level}'** (threshold: {threshold:.2f})...")
                
                # Prepara subset
                df_subset = df_featured.loc[df_subset_idx].copy()
                
                # Verifica features
                for f in features:
                    if f not in df_subset.columns:
                        df_subset[f] = 0 if f not in ['status_cobranca', 'faixa_etaria', 'faixa_score', 'regiao', 'segmento', 'categoria_risco_score', 'modalidade', 'pdd'] else 'Desconhecido'
                
                X = df_subset[features]
                probabilities = pipeline.predict_proba(X)[:, 1]
                predictions = (probabilities >= threshold).astype(int)
                
                temp_df = df.loc[df_subset_idx].copy()
                temp_df['previsao'] = predictions
                temp_df['prob_pagar'] = probabilities * 100
                temp_df['threshold_usado'] = threshold
                results_list.append(temp_df)

    if not results_list:
        st.warning("Nenhum registro correspondente às categorias de risco foi encontrado.")
        return None

    # 5. Consolidação dos Resultados
    df_final_results = pd.concat(results_list).sort_index()
    df_final_results['previsao_label'] = df_final_results['previsao'].map({
        0: 'NÃO Protestar', 
        1: 'PROTESTAR'
    })
    
    st.success("🎉 Previsões V3 concluídas com sucesso!")
    return df_final_results


# =============================================================================
# INTERFACE DO USUÁRIO (CORPO PRINCIPAL)
# =============================================================================

st.header("1. Configure sua Análise")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Escolha um arquivo Excel (.xlsx) ou CSV (.csv)", 
        type=['xlsx', 'csv']
    )
    processing_mode = st.radio(
        "Escolha o Modo de Processamento",
        ('Roteamento Automático por Risco', 'Aplicar Risco Específico a Todos'),
        help="""
        - **Roteamento Automático:** Usa a coluna 'categoria_risco_score' do arquivo para aplicar o modelo correto (Alto, Médio ou Baixo) a cada linha.
        - **Aplicar Risco Específico:** Aplica um único modelo para todas as linhas do arquivo, ignorando a coluna 'categoria_risco_score'.
        """
    )

with col2:
    specific_risk = None
    if processing_mode == 'Aplicar Risco Específico a Todos':
        specific_risk = st.selectbox(
            "Selecione o Risco",
            ("Alto", "Médio", "Baixo")
        )
    
    st.write("\n") # Spacer
    st.write("\n") # Spacer
    process_button = st.button("🚀 Processar Arquivo", disabled=(uploaded_file is None), use_container_width=True)

st.divider()

# --- Corpo principal para outputs ---
if process_button and uploaded_file is not None:
    if models is None:
        st.stop()

    try:
        if uploaded_file.name.endswith('.csv'):
            # Read a small sample to sniff the delimiter
            sample = uploaded_file.read(2048)
            uploaded_file.seek(0)  # Reset file pointer to the beginning
            
            import csv
            try:
                # Decode sample for sniffing
                sample_str = sample.decode('utf-8')
                dialect = csv.Sniffer().sniff(sample_str)
                delimiter = dialect.delimiter
                st.write(f"INFO: Delimitador CSV detectado automaticamente: '{delimiter}'")
            except (csv.Error, UnicodeDecodeError):
                st.warning("Não foi possível detectar o delimitador do CSV. Usando ';' como padrão, pois é comum no Brasil.")
                delimiter = ';' # Fallback to a common delimiter
            
            df_input = pd.read_csv(uploaded_file, sep=delimiter)
        else:
            df_input = pd.read_excel(uploaded_file)
            
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    with st.spinner('Aguarde, os robôs estão trabalhando...'):
        
        risk_to_apply = specific_risk if processing_mode == 'Aplicar Risco Específico a Todos' else None
        
        df_results = run_prediction(df_input, models, risk_category=risk_to_apply)

    if df_results is not None:
        st.header("Resultados da Análise")

        # --- Métricas e Gráficos ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Registros Processados", f"{len(df_results)}")

        prediction_counts = df_results['previsao_label'].value_counts()
        col2.metric("Deve protestar", f"{prediction_counts.get('PROTESTAR', 0)}")
        col3.metric("Não deve protestar", f"{prediction_counts.get('NÃO Protestar', 0)}")

        st.subheader("Distribuição das Previsões")
        st.bar_chart(prediction_counts)
        
        # --- Tabela de Resultados ---
        st.subheader("Resultado Final (amostra de 300 linhas)")

        # Formata as colunas de resultado
        df_results['probabilidade_reverter'] = df_results['prob_pagar'].astype(int)
        
        # Remove a coluna de previsão numérica (0/1) para evitar duplicatas
        if 'previsao' in df_results.columns:
            df_results.drop(columns=['previsao'], inplace=True)

        # Renomeia a coluna com o texto da previsão para 'previsao'
        df_results.rename(columns={'previsao_label': 'previsao'}, inplace=True)

        # Define a ordem das colunas para o display (resultados primeiro)
        cols_to_display = ['previsao', 'probabilidade_reverter'] + [col for col in df_input.columns if col in df_results.columns]
        # Garante que não haja duplicatas na lista de colunas
        cols_to_display = list(dict.fromkeys(cols_to_display)) 

        # Mostra apenas o head na tela
        st.dataframe(df_results[cols_to_display].head(300))

        # --- Botão de Download ---
        # O Excel para download deve conter o dataset completo com os resultados
        @st.cache_data
        def convert_df_to_excel(df):
            output = io.BytesIO()
            # Prepara o dataframe para download (todas as linhas)
            df_for_download = df[cols_to_display]
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_for_download.to_excel(writer, index=False, sheet_name='Resultados')
            processed_data = output.getvalue()
            return processed_data

        excel_output = convert_df_to_excel(df_results)
        
        st.download_button(
           label="📥 Baixar Resultados Completos em Excel",
           data=excel_output,
           file_name=f"resultados_previsao_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Aguardando o upload de um arquivo e o comando para processar.")

