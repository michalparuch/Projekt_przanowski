import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_missingness(df, feature_cols):
    """Analiza brakujących wartości"""
    missing_summary = df[feature_cols].isnull().mean() * 100
    return missing_summary.sort_values(ascending=False)

def analyze_outliers(df, feature_cols):
    """Analiza wartości odstających w zmiennych"""
    outlier_summary = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_summary[col] = outliers / len(df) * 100
    return pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier_Percentage']).sort_values(by='Outlier_Percentage', ascending=False)

def analyze_stability(df1, df2, feature_cols):
    """Analiza stabilności zmiennych między zbiorami"""
    stability_summary = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df1[col]):
            std = df1[col].std()
            if std == 0:  # Jeśli odchylenie standardowe wynosi 0, pomijamy zmienną
                stability_summary[col] = np.nan
            else:
                mean_diff = abs(df1[col].mean() - df2[col].mean())
                stability_summary[col] = mean_diff / std
    return pd.DataFrame.from_dict(stability_summary, orient='index', columns=['Stability_Index']).sort_values(by='Stability_Index', ascending=False)

# Wczytywanie danych
train_path = "C:/Users/xmari/OneDrive/Pulpit/dane_sas/abt_sam_beh_train.sas7bdat"
valid_path = "C:/Users/xmari/OneDrive/Pulpit/dane_sas/abt_sam_beh_valid.sas7bdat"
data_train = pd.read_sas(train_path, format='sas7bdat')
data_valid = pd.read_sas(valid_path, format='sas7bdat')

# Identyfikacja typów zmiennych
ident_cols = ['cid', 'period']
target_cols = [col for col in data_train.columns if col.startswith('default_cus')]
feature_cols = [col for col in data_train.columns if col not in ident_cols + target_cols]

# Interfejs użytkownika
st.title("Porównanie zmiennych między zbiorami danych")
st.sidebar.header("Opcje")

# Wybór trybu analizy
analysis_mode = st.sidebar.radio("Tryb analizy", ["Pojedyncza zmienna", "Scatterplot"])

if analysis_mode == "Pojedyncza zmienna":
    # Wybór zmiennej
    selected_column = st.sidebar.selectbox("Wybierz zmienną", feature_cols)

    # Wybór typu zmiennej
    variable_type = st.sidebar.radio("Typ zmiennej", ["Liczbowa", "Kategorialna"])

    # Wyświetlenie wykresów
    st.header(f"Analiza zmiennej: {selected_column}")

    if variable_type == "Liczbowa":

        if pd.api.types.is_numeric_dtype(data_train[selected_column]):
            stability_index = analyze_stability(data_train, data_valid, [selected_column])
            st.subheader("Stabilność zmiennej")
            st.write(stability_index)

        # Tworzenie histogramów
        fig_train = px.histogram(
            data_train, x=selected_column, title=f"Rozkład zmiennej {selected_column} - Zbiór Train",
            nbins=30
        )
        fig_valid = px.histogram(
            data_valid, x=selected_column, title=f"Rozkład zmiennej {selected_column} - Zbiór Valid",
            nbins=30
        )

        # Wyświetlenie wykresów w dwóch kolumnach
        col1, col2 = st.columns(2)
        with col1:
            if pd.api.types.is_numeric_dtype(data_train[selected_column]):
                outliers_train = analyze_outliers(data_train, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Train:")
                st.write(outliers_train)

            missingness_train = analyze_missingness(data_train, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Train:")
            st.write(missingness_train)

            st.plotly_chart(fig_train, use_container_width=True)


        with col2:
            if pd.api.types.is_numeric_dtype(data_train[selected_column]):
                outliers_valid = analyze_outliers(data_valid, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Valid:")
                st.write(outliers_valid)

            missingness_valid = analyze_missingness(data_valid, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Valid:")
            st.write(missingness_valid)

            st.plotly_chart(fig_valid, use_container_width=True)


        # Tworzenie boxplotów
        box_train = px.box(data_train, y=selected_column, title=f"Boxplot zmiennej {selected_column} - Zbiór Train")
        box_valid = px.box(data_valid, y=selected_column, title=f"Boxplot zmiennej {selected_column} - Zbiór Valid")

        # Wyświetlenie boxplotów w dwóch kolumnach
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(box_train, use_container_width=True)
        with col2:
            st.plotly_chart(box_valid, use_container_width=True)

    elif variable_type == "Kategorialna":

        if pd.api.types.is_numeric_dtype(data_train[selected_column]):
            stability_index = analyze_stability(data_train, data_valid, [selected_column])
            st.subheader("Stabilność zmiennej")
            st.write(stability_index)

        # Tworzenie countplotów
        fig_train = px.histogram(
            data_train, x=selected_column, title=f"Countplot zmiennej {selected_column} - Zbiór Train",
            color=selected_column
        )
        fig_valid = px.histogram(
            data_valid, x=selected_column, title=f"Countplot zmiennej {selected_column} - Zbiór Valid",
            color=selected_column
        )

        # Wyświetlenie countplotów w dwóch kolumnach
        col1, col2 = st.columns(2)
        with col1:
            if pd.api.types.is_numeric_dtype(data_train[selected_column]):
                outliers_train = analyze_outliers(data_train, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Train:")
                st.write(outliers_train)

            missingness_train = analyze_missingness(data_train, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Train:")
            st.write(missingness_train)

            st.plotly_chart(fig_train, use_container_width=True)
        with col2:
            if pd.api.types.is_numeric_dtype(data_train[selected_column]):
                outliers_valid = analyze_outliers(data_valid, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Valid:")
                st.write(outliers_valid)

            missingness_valid = analyze_missingness(data_valid, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Valid:")
            st.write(missingness_valid)

            st.plotly_chart(fig_valid, use_container_width=True)

        # Heatmapa rozkładu wartości kategorycznych
        st.subheader("Heatmapa rozkładu wartości kategorycznych")

        # Tworzenie pivot table dla heatmapy
        train_counts = data_train[selected_column].value_counts().reset_index()
        valid_counts = data_valid[selected_column].value_counts().reset_index()

        train_counts.columns = [selected_column, "Train"]
        valid_counts.columns = [selected_column, "Valid"]

        combined_counts = pd.merge(train_counts, valid_counts, on=selected_column, how="outer").fillna(0)
        combined_counts.set_index(selected_column, inplace=True)

        # Tworzenie heatmapy
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            combined_counts,
            fmt=".0f",
            cmap="coolwarm",
            cbar=True,
            ax=ax
        )
        ax.set_title("Porównanie liczności zmiennych kategorycznych (Train vs Valid)")
        st.pyplot(fig)

elif analysis_mode == "Scatterplot":
    # Wybór zmiennych dla scatterplota
    x_column = st.sidebar.selectbox("Wybierz zmienną (oś X)", feature_cols)
    y_column = st.sidebar.selectbox("Wybierz zmienną (oś Y)", feature_cols)

    st.header(f"Scatterplot: {x_column} vs {y_column}")

    # Scatterplot dla zbioru Train
    scatter_train = px.scatter(
        data_train, x=x_column, y=y_column, title=f"Scatterplot - Zbiór Train",
        color=y_column, marginal_x="box", marginal_y="box"
    )

    # Scatterplot dla zbioru Valid
    scatter_valid = px.scatter(
        data_valid, x=x_column, y=y_column, title=f"Scatterplot - Zbiór Valid",
        color=y_column, marginal_x="box", marginal_y="box"
    )

    # Wyświetlenie scatterplotów w dwóch kolumnach
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(scatter_train, use_container_width=True)
    with col2:
        st.plotly_chart(scatter_valid, use_container_width=True)
