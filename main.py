import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_excel(file_path)

train_data = load_data(r'C:/Users/xmari/OneDrive/Pulpit/dane_sas/excel_out_train.xlsx')
valid_data = load_data(r'C:/Users/xmari/OneDrive/Pulpit/dane_sas/excel_out_valid.xlsx')

train_stability_data = load_data(r'C:/Users/xmari/OneDrive/Pulpit/dane_sas/stab_train.xlsx')
valid_stability_data = load_data(r'C:/Users/xmari/OneDrive/Pulpit/dane_sas/stab_valid.xlsx')

def analyze_missingness(df, feature_cols):
    """Analiza brakujących wartości"""
    missing_summary = df[feature_cols].isnull().mean() * 100
    return missing_summary.sort_values(ascending=False)

def analyze_outliers(df, feature_cols):
    q1 = df[feature_cols].quantile(0.25)
    q3 = df[feature_cols].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = ((df[feature_cols] < lower_bound) | (df[feature_cols] > upper_bound)).sum()
    return (outliers / len(df) * 100).sort_values(ascending=False).to_frame(name="Outlier_Percentage")

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

def convert_bytes_to_str(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df

data_train = convert_bytes_to_str(data_train)
data_valid = convert_bytes_to_str(data_valid)

def process_period_column(df):
    df[['year', 'month']] = df['period'].astype(str).str.extract(r'(\d{4})(\d{2})').astype(int)
    return df

data_train = process_period_column(data_train)
data_valid = process_period_column(data_valid)

# Identyfikacja typów zmiennych
ident_cols = ['cid', 'period']
target_cols = [col for col in data_train.columns if col.startswith('default_cus')]
feature_cols = [col for col in data_train.columns if col not in ident_cols + target_cols]

# Interfejs użytkownika
st.title("Porównanie zmiennych między zbiorami danych")
st.sidebar.header("Opcje")

analysis_mode = st.sidebar.radio("Tryb analizy", ["Pojedyncza zmienna", "Scatterplot", "Analiza wartości odstających"])

if analysis_mode != "Analiza wartości odstających":
    st.sidebar.subheader("Filtrowanie po roku")
    unique_years = sorted(list(set(data_train['year'].unique()) | set(data_valid['year'].unique())))
    year_options = ["All"] + unique_years  # Dodaj opcję "All" na początek listy
    selected_year = st.sidebar.selectbox("Wybierz rok", year_options, index=year_options.index(2004))

    if selected_year == "All":
        data_train_filtered = data_train
        data_valid_filtered = data_valid
    else:
        data_train_filtered = data_train[data_train['year'] == selected_year]
        data_valid_filtered = data_valid[data_valid['year'] == selected_year]

# # Wybór trybu analizy
# analysis_mode = st.sidebar.radio("Tryb analizy", ["Pojedyncza zmienna", "Scatterplot", "Analiza wartości odstających"])

if analysis_mode == "Pojedyncza zmienna":
    # Wybór zmiennej
    selected_column = st.sidebar.selectbox("Wybierz zmienną", feature_cols)

    # Wybór typu zmiennej
    variable_type = st.sidebar.radio("Typ zmiennej", ["Ilosciowa", "Kategorialna"])

    # Wyświetlenie wykresów
    st.header(f"Analiza zmiennej: {selected_column}")

    # Wybór liczby binów dla histogramów (tylko dla zmiennych liczbowych)
    if variable_type == "Ilosciowa":

        # if pd.api.types.is_numeric_dtype(data_train[selected_column]):
        #     stability_index = analyze_stability(data_train, data_valid, [selected_column])
        #     st.subheader("Stabilność zmiennej")
        #     st.write(stability_index)

        bins = st.sidebar.slider("Liczba binów dla histogramu", min_value=5, max_value=100, value=30, step=1)

        # Tworzenie histogramów z wybraną liczbą binów
        fig_train = px.histogram(
            data_train_filtered, x=selected_column, title=f"Rozkład zmiennej {selected_column} - Zbiór Train",
            nbins=bins
        )
        fig_valid = px.histogram(
            data_valid_filtered, x=selected_column, title=f"Rozkład zmiennej {selected_column} - Zbiór Valid",
            nbins=bins
        )

        # Wyświetlenie histogramów w dwóch kolumnach
        col1, col2 = st.columns(2)
        with col1:
            if pd.api.types.is_numeric_dtype(data_train_filtered[selected_column]):
                outliers_train = analyze_outliers(data_train_filtered, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Train:")
                st.write(outliers_train)

            missingness_train = analyze_missingness(data_train_filtered, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Train:")
            st.write(missingness_train)

            st.plotly_chart(fig_train, use_container_width=True)

        with col2:
            if pd.api.types.is_numeric_dtype(data_train_filtered[selected_column]):
                outliers_valid = analyze_outliers(data_valid_filtered, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Valid:")
                st.write(outliers_valid)

            missingness_valid = analyze_missingness(data_valid_filtered, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Valid:")
            st.write(missingness_valid)

            st.plotly_chart(fig_valid, use_container_width=True)

        # Tworzenie boxplotów
        box_train = px.box(data_train_filtered, y=selected_column, title=f"Boxplot zmiennej {selected_column} - Zbiór Train")
        box_valid = px.box(data_valid_filtered, y=selected_column, title=f"Boxplot zmiennej {selected_column} - Zbiór Valid")

        # Wyświetlenie boxplotów w dwóch kolumnach
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(box_train, use_container_width=True)
        with col2:
            st.plotly_chart(box_valid, use_container_width=True)

        if selected_year == "All":
            st.header("Trendy w czasie dla zmiennych liczbowych")

            # Wybór zmiennej do analizy trendu

            if pd.api.types.is_numeric_dtype(data_train[selected_column]):
                # Grupowanie danych po roku
                train_yearly_stats = data_train.groupby('year')[selected_column].agg(['mean', 'std']).reset_index()
                valid_yearly_stats = data_valid.groupby('year')[selected_column].agg(['mean', 'std']).reset_index()

                # Wykres dla zbioru Train
                fig_train = px.line(
                    train_yearly_stats, x='year', y='mean',
                    error_y='std', title=f"Trend zmiennej {selected_column} w czasie - Zbiór Train",
                    labels={'mean': 'Średnia', 'std': 'Odchylenie standardowe', 'year': 'Rok'}
                )

                # Wykres dla zbioru Valid
                fig_valid = px.line(
                    valid_yearly_stats, x='year', y='mean',
                    error_y='std', title=f"Trend zmiennej {selected_column} w czasie - Zbiór Valid",
                    labels={'mean': 'Średnia', 'std': 'Odchylenie standardowe', 'year': 'Rok'}
                )

                # Wyświetlenie wykresów w dwóch kolumnach
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_train, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_valid, use_container_width=True)

        if selected_year == "All" and pd.api.types.is_numeric_dtype(data_train[selected_column]):
            stability_index = analyze_stability(data_train, data_valid, [selected_column])
            st.subheader("Stabilność zmiennej (dla wszystkich lat)")
            image_path = r"C:/Users/xmari/OneDrive/Pulpit/dane_sas/wzor.png"
            st.image(image_path, caption="Wzór na podstawie którego liczony jest index stabilności", use_container_width=True)
            st.write("""
                Interpretacja wskaźnika:
                - < 0.5: Zmienna jest stabilna – różnice między zbiorami są niewielkie.
                - 0.5 ≤ wskaźnik < 1.0: Zmienna jest umiarkowanie niestabilna – warto przeanalizować.
                - ≥ 1.0: Zmienna jest niestabilna – różnice są znaczące i mogą wymagać uwagi.
                """)
            st.subheader("Stabilność zmiennej (dla wszystkich lat)")
            st.write(stability_index)

        if selected_year == "All":
            # Obliczanie stabilności dla każdego roku
            stability_per_year = {}
            for year in unique_years:
                train_year = data_train[data_train['year'] == year]
                valid_year = data_valid[data_valid['year'] == year]
                stability = analyze_stability(train_year, valid_year, [selected_column])
                stability_per_year[year] = stability.iloc[0, 0]  # Pobierz wartość wskaźnika stabilności

            # Tworzenie DataFrame z wynikami
            stability_df = pd.DataFrame.from_dict(
                stability_per_year, orient='index', columns=['Stability_Index']
            ).reset_index().rename(columns={'index': 'Year'})

            # Wyświetlenie tabeli stabilności
            st.write(stability_df)

            # Wykres stabilności w czasie
            fig_stability = px.line(
                stability_df, x='Year', y='Stability_Index',
                title=f"Stabilność zmiennej {selected_column} w czasie",
                labels={'Stability_Index': 'Wskaźnik stabilności', 'Year': 'Rok'}
            )
            st.plotly_chart(fig_stability, use_container_width=True)


    elif variable_type == "Kategorialna":

        # if pd.api.types.is_numeric_dtype(data_train_filtered[selected_column]):
        #     stability_index = analyze_stability(data_train_filtered, data_valid_filtered, [selected_column])
        #     st.subheader("Stabilność zmiennej")
        #     st.write(stability_index)

        # Tworzenie countplotów
        fig_train = px.histogram(
            data_train_filtered, x=selected_column, title=f"Countplot zmiennej {selected_column} - Zbiór Train",
            color=selected_column
        )
        fig_valid = px.histogram(
            data_valid_filtered, x=selected_column, title=f"Countplot zmiennej {selected_column} - Zbiór Valid",
            color=selected_column
        )

        # Wyświetlenie countplotów w dwóch kolumnach
        col1, col2 = st.columns(2)
        with col1:
            if pd.api.types.is_numeric_dtype(data_train_filtered[selected_column]):
                outliers_train = analyze_outliers(data_train_filtered, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Train:")
                st.write(outliers_train)

            missingness_train = analyze_missingness(data_train_filtered, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Train:")
            st.write(missingness_train)

            st.plotly_chart(fig_train, use_container_width=True)
        with col2:
            if pd.api.types.is_numeric_dtype(data_train_filtered[selected_column]):
                outliers_valid = analyze_outliers(data_valid_filtered, [selected_column])

                st.subheader("Wartości odstające")
                st.write("Zbiór Valid:")
                st.write(outliers_valid)

            missingness_valid = analyze_missingness(data_valid_filtered, [selected_column])
            st.subheader("Brakujące wartości")
            st.write("Zbiór Valid:")
            st.write(missingness_valid)

            st.plotly_chart(fig_valid, use_container_width=True)

        # Heatmapa rozkładu wartości kategorycznych
        st.subheader("Heatmapa rozkładu wartości kategorycznych")

        # Tworzenie pivot table dla heatmapy
        train_counts = data_train_filtered[selected_column].value_counts().reset_index()
        valid_counts = data_valid_filtered[selected_column].value_counts().reset_index()

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

        if selected_year == "All" and pd.api.types.is_numeric_dtype(data_train[selected_column]):
            stability_index = analyze_stability(data_train, data_valid, [selected_column])
            st.subheader("Stabilność zmiennej (dla wszystkich lat)")
            st.write(stability_index)

        if selected_year == "All":
            # Obliczanie stabilności dla każdego roku
            stability_per_year = {}
            for year in unique_years:
                train_year = data_train[data_train['year'] == year]
                valid_year = data_valid[data_valid['year'] == year]
                stability = analyze_stability(train_year, valid_year, [selected_column])
                stability_per_year[year] = stability.iloc[0, 0]  # Pobierz wartość wskaźnika stabilności

            # Tworzenie DataFrame z wynikami
            stability_df = pd.DataFrame.from_dict(
                stability_per_year, orient='index', columns=['Stability_Index']
            ).reset_index().rename(columns={'index': 'Year'})

            # Wyświetlenie tabeli stabilności
            st.write(stability_df)

            # Wykres stabilności w czasie
            fig_stability = px.line(
                stability_df, x='Year', y='Stability_Index',
                title=f"Stabilność zmiennej {selected_column} w czasie",
                labels={'Stability_Index': 'Wskaźnik stabilności', 'Year': 'Rok'}
            )
            st.plotly_chart(fig_stability, use_container_width=True)

elif analysis_mode == "Scatterplot":
    # Wybór zmiennych dla scatterplota
    x_column = st.sidebar.selectbox("Wybierz zmienną (oś X)", feature_cols)
    y_column = st.sidebar.selectbox("Wybierz zmienną (oś Y)", feature_cols)

    st.header(f"Scatterplot: {x_column} vs {y_column}")

    # Scatterplot dla zbioru Train
    scatter_train = px.scatter(
        data_train_filtered, x=x_column, y=y_column, title=f"Scatterplot - Zbiór Train",
        color=y_column, marginal_x="box", marginal_y="box"
    )

    # Scatterplot dla zbioru Valid
    scatter_valid = px.scatter(
        data_valid_filtered, x=x_column, y=y_column, title=f"Scatterplot - Zbiór Valid",
        color=y_column, marginal_x="box", marginal_y="box"
    )

    # Wyświetlenie scatterplotów w dwóch kolumnach
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(scatter_train, use_container_width=True)
    with col2:
        st.plotly_chart(scatter_valid, use_container_width=True)

elif analysis_mode == "Analiza wartości odstających":
    # Wybór zmiennych dla wartości odstających
    feature_cols = train_data['Variable'].tolist()
    selected_features = st.sidebar.multiselect("Wybierz zmienne do analizy odstających", feature_cols, default=feature_cols[:3])

    if selected_features:
        st.write("### Porównanie wartości odstających")

        # Filtrowanie danych dla wybranych zmiennych
        train_filtered = train_data[train_data['Variable'].isin(selected_features)]
        valid_filtered = valid_data[valid_data['Variable'].isin(selected_features)]

        # Dodanie kolumny identyfikującej dane (Train/Valid)
        train_filtered['Dataset'] = 'Train'
        valid_filtered['Dataset'] = 'Valid'

        # Łączenie danych
        combined_data = pd.concat([train_filtered, valid_filtered])

        # Wykres wartości odstających
        fig_outliers = px.bar(
            combined_data,
            x='Variable',
            y='Outliers_Percent',
            color='Dataset',
            barmode='group',
            title="Porównanie wartości odstających (% odstających)",
            labels={"Outliers_Percent": "% Odstających", "Variable": "Zmienna"},
        )

        st.plotly_chart(fig_outliers, use_container_width=True)

        # Analiza stabilności w czasie
        st.write("### Stabilność w czasie")

        # Ograniczenie zmiennych do tych w danych stabilności
        stability_cols = list(set(train_stability_data.columns).intersection(set(valid_stability_data.columns)))
        stability_cols.remove('Period')  # Pomijamy kolumnę 'Period'

        # Wybór zmiennych do analizy stabilności
        selected_stability_features = st.sidebar.multiselect(
            "Wybierz zmienne do analizy stabilności",
            stability_cols,
            default=stability_cols[:3]
        )

        if selected_stability_features:
            # Filtrujemy dane na podstawie wybranych zmiennych
            train_stability = train_stability_data[['Period'] + selected_stability_features]
            valid_stability = valid_stability_data[['Period'] + selected_stability_features]

            # # Konwersja kolumn na wartości numeryczne
            # for feature in selected_stability_features:
            #     train_stability[feature] = pd.to_numeric(train_stability[feature], errors='coerce')
            #     valid_stability[feature] = pd.to_numeric(valid_stability[feature], errors='coerce')
            #
            # # Usuwanie wierszy z brakującymi wartościami
            # train_stability = train_stability.dropna(subset=selected_stability_features)
            # valid_stability = valid_stability.dropna(subset=selected_stability_features)

            # Dodajemy identyfikator zbioru
            train_stability['Dataset'] = 'Train'
            valid_stability['Dataset'] = 'Valid'

            # Łączenie danych
            combined_stability = pd.concat([train_stability, valid_stability])

            # Tworzymy wykres liniowy z użyciem matplotlib
            plt.figure(figsize=(12, 6))

            for feature in selected_stability_features:
                train_data_feature = combined_stability[combined_stability['Dataset'] == 'Train']
                plt.plot(
                    train_data_feature['Period'],
                    train_data_feature[feature],
                    label=f'Train - {feature}',
                    linestyle='-',
                    marker='o'
                )

                valid_data_feature = combined_stability[combined_stability['Dataset'] == 'Valid']
                plt.plot(
                    valid_data_feature['Period'],
                    valid_data_feature[feature],
                    label=f'Valid - {feature}',
                    linestyle='--',
                    marker='x'
                )

            plt.title("Stabilność zmiennych w czasie")
            plt.xlabel("Okres")
            plt.ylabel("Wartość")
            plt.legend()
            plt.grid(True)

            # Wyświetlamy wykres w Streamlit
            st.pyplot(plt)
