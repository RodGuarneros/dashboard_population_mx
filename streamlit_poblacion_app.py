#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import io
import geopandas as gpd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.io as pio
import altair_viewer as altviewer
import logging


# Page configuration
st.set_page_config(
    page_title="Tendencias Poblacionales",
    page_icon="👩‍👧‍👦",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: -10rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


# Load data
datos = pd.read_csv('data/datos.csv', encoding='Latin1') # piramides
input_datos = datos
df_reshaped = pd.read_csv('data/result_sorted_final.csv', encoding='utf-8')
input_df = df_reshaped.sort_values(by="ENTIDAD", ascending=True)
input_df = df_reshaped # cholopleth, ranking_edad evolucion_poblacion
df_reshaped_2 = pd.read_csv('data/result_sorted2_hist.csv', encoding='Latin1') # histograma
input_hist = df_reshaped_2
# df_calculos = pd.read_csv('calculos.csv', encoding='Latin1')


# Sidebar
with st.sidebar:
    # st.title('Tendencias poblacionales 1970-2050 <br> México')
    st.markdown("<h1 style='text-align: center;'>Tendencias poblacionales 1970-2050 <br> México</h1>", unsafe_allow_html=True)
    st.sidebar.image("https://img.cdn-pictorem.com/uploads/collection/L/LD5RFK7TTK/900_Grasshopper-Geography_Elevation_map_of_Mexico_with_black_background.jpg", use_column_width=True)
    
    # Año
    # year_list = list(df_reshaped.AÑO.unique())[:19]
    year_list = list(range(1970, 2051))
    selected_year = st.selectbox('Seleccione el año:', sorted(year_list, reverse=False))
    df_selected_year = df_reshaped[df_reshaped.AÑO == selected_year]
    input_year = df_selected_year
    df_selected_year_sorted = df_selected_year.sort_values(by="POBLACION", ascending=False)

    # Entidad
    entidad_list = list(df_reshaped.ENTIDAD.unique())
    selected_entidad = st.selectbox('Seleccione la entidad o República Mexicana:', sorted(entidad_list, reverse=False))
    df_selected_entidad = df_reshaped[df_reshaped.ENTIDAD == selected_entidad]
    input_entidad = df_selected_entidad
    df_selected_entidad_sorted = df_selected_entidad.sort_values(by="POBLACION", ascending=False)

    # Género
    genero_list = list(df_reshaped.SEXO.unique())
    selected_genero = st.selectbox('Seleccione por género o datos totales:', sorted(genero_list, reverse=True))
    df_selected_genero = df_reshaped[df_reshaped.SEXO == selected_genero]
    input_genero = df_selected_genero
    df_selected_genero_sorted = df_selected_genero.sort_values(by="POBLACION", ascending=False)

    with st.expander('Filosofía del panel de control', expanded=False):
        st.write('''
            - Se basa en un enfoque de <span style="color:#C2185B">"Programación Orientada a Objetos"</span>.
            - La población se puede modelar a partir de sus atributos y funciones que en escencia definen sus características y capacidades, respectivamente. 
            - En este ejemplo, se parte de la pregunta básica <span style="color:#C2185B">¿Cuál es la tendencia de crecimiento poblacional a nivel nacional y por entidad federativa entre 1970 y 2050, y cómo varía esta tendencia según el género y la edad de la población?</span>
            - Este aplicativo incluye atributos de la población mexicana como:
                1. El año en el que se sitúa.
                2. La Entidad Federativa a la que pertenece. 
                3. El género de la población disponible en los datos (Femenino y Masculino).
                4. La edad promedio y su distribución.
            - Con base en estas características, el usuario puede generar combinaciones de interés para conocer las perspectivas sobre:
                1. La evolución de la población entre 1970 y 2050. 
                2. La pirámide poblacional. 
                3. La distribución de la población por edad.
            - Es posible también generar perspectivas sobre la distribución geográfica y ranking en dos dimensiones:
                1. Población total por entidad federativa y nacional.
                2. Edad promedio por estado y nacional.
            - La ventaja de un panel de control como este consiste en sus <span style="color:#C2185B">economías de escala y la capacidad que tiene para presentar insights más profundos respecto a la población y sus funciones o actividades, tales como capacidad adquisitiva, preferencias, crédito al consumo, acceso a servicios de conectividad, empleo, sequías y hasta modelos predictivos.</span> 
            ''', unsafe_allow_html=True)



    with st.expander('Fuentes y detalles técnicos', expanded=False):
        st.write('''
            - Fuente: [Consejo Nacional de Población (CONAPO), consultado el 3 de febrero de 2024.](https://www.gob.mx/conapo).
            - Tecnologías y lenguajes: Python 3.10, Streamlit 1.30.0, CSS 3.0, HTML5, Google Colab y GitHub. 
            - Autor: Rodrigo Guarneros ([LinkedIn](https://www.linkedin.com/in/guarneros/) y [X](https://twitter.com/RodGuarneros)).
            - Basado en [@DataProfessor](https://github.com/dataprofessor/population-dashboard/tree/master).
            - Comentarios al correo electrónico rodrigo.guarneros@gmail.com.
            ''', unsafe_allow_html=True)


# Ranking Edad
def ranking_edad(input_df, input_year, input_genero, entidad_seleccionada=None):
    año_seleccionado_dato_rank_edad = input_df[(input_df['AÑO'] == input_year) & (input_df['SEXO'] == input_genero)].reset_index(drop=True)
    
    result_df_2 = pd.DataFrame({
        "ENTIDAD": año_seleccionado_dato_rank_edad['ENTIDAD'],
        "Edad promedio": np.round(año_seleccionado_dato_rank_edad['EDAD_PROMEDIO'], 2)
    }).sort_values(by='Edad promedio', ascending=True)

    fig = px.bar(result_df_2, x='Edad promedio', y='ENTIDAD', orientation='h',
                 title=f'Ranking Nacional Edad Promedio {input_year}, {input_genero}',
                 labels={'Edad promedio': 'Edad promedio', 'ENTIDAD': 'Estado'},
                 template='plotly_dark',
                 color='Edad promedio',
                 color_continuous_scale='reds',
                 custom_data=[result_df_2['Edad promedio']])  # Use custom_data to store 'Edad promedio' values
    
    fig.update_layout(title_x=0.15)
    fig.update_layout(margin=dict(b=30, t=30))
    fig.update_layout(height=600)  # Adjust the height as needed
    fig.update_layout(xaxis=dict(tickfont=dict(size=8)))  # Adjust the font size as needed
    fig.update_traces(width=0.5)  # Adjust the width as needed
    fig.update_traces(marker=dict(line=dict(width=7, color='DarkSlateGray', shape='spline')), selector=dict(mode='markers'), Width=0.1)


    if entidad_seleccionada:
        selected_entity_color = '#1DD5EE'
        other_entity_color = '#D82C20'
        fig.update_traces(marker=dict(color=[selected_entity_color if entidad == entidad_seleccionada else other_entity_color for entidad in result_df_2['ENTIDAD']]),
                          selector=dict(type='bar'))
        
    
    fig.update_layout({'plot_bgcolor': 'black', 'paper_bgcolor': 'black'})

    # Add 'Edad promedio' values to the tooltip
    fig.update_traces(hovertemplate='<b>%{y}</b><br>Edad promedio=%{customdata[0]:,.2f}')
    fig.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'), yaxis_title_font=dict(color='#FFD86C'))
    
    return fig

fig_ranking = ranking_edad(input_df, selected_year, selected_genero, entidad_seleccionada=selected_entidad)


def ranking_poblacion2(input_df, input_year, input_genero, entidad_seleccionada=None):
    año_seleccionado_dato_rank_edad = input_df[(input_df['AÑO'] == input_year) & (input_df['SEXO'] == input_genero)].reset_index(drop=True)
    año_seleccionado_dato_rank_edad = año_seleccionado_dato_rank_edad.sort_values(by='POBLACION', ascending=True)
    año_seleccionado_dato_rank_edad = año_seleccionado_dato_rank_edad[año_seleccionado_dato_rank_edad['ENTIDAD'] != 'República Mexicana']

    fig5 = px.bar(año_seleccionado_dato_rank_edad, x='POBLACION', y='ENTIDAD', orientation='h',
                 title=f'Ranking Nacional Población Total {input_year}, {input_genero}',
                 labels={'POBLACION': 'Población', 'ENTIDAD': 'Estado'},
                 template='plotly_dark',
                #  color='POBLACIÓN',
                 color_continuous_scale='reds',
                 custom_data=[año_seleccionado_dato_rank_edad['POBLACION']])  # Use custom_data to store 'Edad promedio' values
    
    fig5.update_layout(title_x=0.15)
    fig5.update_layout(margin=dict(b=30, t=30))
    fig5.update_layout(height=600)  # Adjust the height as needed
    fig5.update_layout(xaxis=dict(tickfont=dict(size=8)))  # Adjust the font size as needed
    fig5.update_traces(width=0.5)  # Adjust the width as needed
    fig5.update_traces(marker=dict(line=dict(width=7, color='DarkSlateGray', shape='spline')), selector=dict(mode='markers'), Width=0.1)

    if entidad_seleccionada:
        selected_entity_color = '#1DD5EE'
        other_entity_color = '#D82C20'
        fig5.update_traces(marker=dict(color=[selected_entity_color if entidad == entidad_seleccionada else other_entity_color for entidad in año_seleccionado_dato_rank_edad['ENTIDAD']]),
                          selector=dict(type='bar'))
        
    
    fig5.update_layout({'plot_bgcolor': 'black', 'paper_bgcolor': 'black'})

    # Add 'Edad promedio' values to the tooltip
    fig5.update_traces(hovertemplate='<b>%{y}</b><br>POBLACIÓN=%{customdata[0]:,.0f}')
    fig5.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'), yaxis_title_font=dict(color='#FFD86C'))
    
    return fig5

fig_ranking2 = ranking_poblacion2(input_df, selected_year, selected_genero, selected_entidad)

##### EVOLUCIÓN poblacional ######
def evolucion_poblacion(input_df, entidad_seleccionada, highlight_year=None):
    df = input_df[input_df['ENTIDAD'] == entidad_seleccionada].sort_values(by='AÑO')
    
    # Pivote
    df_pivoted = df.pivot_table(index=['AÑO', 'ENTIDAD', 'Clave_Entidad', 'EDAD_PROMEDIO'], columns='SEXO', values='POBLACION', aggfunc='sum').reset_index()

    # Renombrando las columnas
    df_pivoted.columns.name = None  # ya no hay sexo name
    df_pivoted = df_pivoted.rename(columns={'Hombres': 'POBLACION_Hombres', 'Mujeres': 'POBLACION_Mujeres', 'Total': 'POBLACION_Total'})

    # Pivote
    df_merged = pd.merge(df, df_pivoted, on=['AÑO', 'ENTIDAD', 'Clave_Entidad', 'EDAD_PROMEDIO']).groupby(['AÑO', 'ENTIDAD']).sum().reset_index()
    
    # Create the line chart
    fig = px.line(df_merged, x='AÑO', y=['POBLACION_Hombres', 'POBLACION_Mujeres', 'POBLACION_Total'],
                  color_discrete_map={ 'POBLACION_Total': '#1DD5EE', 'POBLACION_Mujeres': '#D82C20', 'POBLACION_Hombres': 'lightgreen'},
                  labels={'AÑO': 'Año', 'value': 'Población'},
                  title=f'Dinámica poblacional,<br> {entidad_seleccionada}',
                  template='plotly_dark')
    fig.update_traces(marker=dict(size=5))  # Adjust the size as needed

# Reduce font size of y-axis labels
    fig.update_layout(yaxis=dict(tickfont=dict(size=20)))  # Adjust the font size as needed

    fig.update_layout(title_x=0.15)

    fig.update_layout(legend=dict(traceorder='reversed'))

    # Customize legends manually
    fig.for_each_trace(lambda t: t.update(name='Total') if 'Total' in (t.name or '') else None)
    fig.for_each_trace(lambda t: t.update(name='Mujeres') if 'Mujeres' in (t.name or '') else None)
    fig.for_each_trace(lambda t: t.update(name='Hombres') if 'Hombres' in (t.name or '') else None)
    

    # Add a circular marker for the highlighted year
    if highlight_year:
        year_marker_data = df_merged[df_merged['AÑO'] == highlight_year]
        fig.add_trace(go.Scatter(x=[highlight_year] * 3, y=year_marker_data[['POBLACION_Hombres', 'POBLACION_Mujeres', 'POBLACION_Total']].values.flatten(),
                                 mode='markers',
                                 marker=dict(color='yellow', size=10, opacity=0.3),
                                 showlegend=False))
    circle_size = 12
    fig.add_trace(go.Scatter(x=[highlight_year] * 3, y=year_marker_data[['POBLACION_Hombres', 'POBLACION_Mujeres', 'POBLACION_Total']].values.flatten(),
                             mode='markers',
                             marker=dict(color='green', size=circle_size, opacity=0.3, line=dict(color='white', width=2)),
                             showlegend=False))
    # Update layout
    fig.update_layout(legend_title_text=f'Género', legend=dict(itemsizing='constant', title_font=dict(color='white'), font=dict(color='white')))

    fig.update_layout({'plot_bgcolor': 'black', 'paper_bgcolor': 'black'})
    fig.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'), yaxis_title_font=dict(color='#FFD86C'))

    return fig


# Display evolution plot
fig_evolucion = evolucion_poblacion(input_df, selected_entidad, highlight_year=selected_year)

# Pirámide poblacional ######################

def piramide_poblacional(input_datos, input_year, entidad_seleccionada = None):
    age_bins = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 150]
    age_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
                  '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90+']

    df = input_datos[(input_datos['AÑO'] == input_year) & (input_datos['ENTIDAD'] == entidad_seleccionada)]

    # Incluimos una columna de Age en df using .loc
    df.loc[:, 'Age_Group'] = pd.cut(df['EDAD'], bins=age_bins, labels=age_labels, right=False)

    # Extraer datos de Hombres
    df_hombres = df[df['SEXO'] == 'Hombres'].groupby('Age_Group')['POBLACION'].sum().reset_index()
    df_hombres = df_hombres.rename(columns={'Age_Group': 'Edad', 'POBLACION': 'Hombres'})
    df_hombres['Hombres'] = df_hombres['Hombres'] * -1

    # Extremos datos de mujeres
    df_mujeres = df[df['SEXO'] == 'Mujeres'].groupby('Age_Group')['POBLACION'].sum().reset_index()
    df_mujeres = df_mujeres.rename(columns={'Age_Group': 'Edad', 'POBLACION': 'Mujeres'})
    
    # Fusionamos sobre la columna Age
    result_df = pd.merge(df_hombres, df_mujeres, on='Edad', how='outer')

    # Calculate absolute values for Hombres and Mujeres columns
    result_df['Hombres_abs'] = result_df['Hombres'].abs()
    result_df['Mujeres_abs'] = result_df['Mujeres'].abs()

    # Plotting using Plotly Express
    fig = px.bar(result_df, x=['Hombres', 'Mujeres'], y='Edad', orientation='h',
                 title=f'Pirámide Poblacional,<br> {selected_entidad} {selected_year}', labels={'Edad': 'Edad promedio', 'value': 'Población'},
                 template='plotly_dark',
                 color_discrete_map={'Mujeres': '#D82C20', 'Hombres': 'lightgreen'})  

    # Set background color to black
    fig.update_layout({'plot_bgcolor': 'black', 'paper_bgcolor': 'black'})
    fig.update_layout(title_x=0.15)

    fig.update_layout(legend=dict(traceorder='reversed'))

    # Adjust orientation of x-axis labels for 'Mujeres' and 'Hombres' columns
    fig.update_xaxes(tickangle=0, side='top', col=0)  # For 'Hombres' column
    fig.update_xaxes(tickangle=0, side='top', col=1)  # For 'Mujeres' column


    # Customize legends
    fig.update_layout(legend_title_text=f'Género', legend=dict(itemsizing='constant', title_font=dict(color='white'), font=dict(color='white')))

    # Remove x-axis labels
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[]))

    # Customize tooltips to show absolute values for x-axis
    fig.update_traces(hovertemplate='Edad promedio: %{y}<br>Población Hombres: %{customdata[0]:+}<br>Población Mujeres: %{customdata[1]:+}',
                      customdata=result_df[['Hombres_abs', 'Mujeres_abs']].to_numpy())
    
    fig.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'), yaxis_title_font=dict(color='#FFD86C'))
    
    return fig

# Assuming datos is your DataFrame
fig_piramide = piramide_poblacional(input_datos, selected_year, entidad_seleccionada=selected_entidad)

# Histograma #####
def calculate_statistics(df):
    ages = np.repeat(df['EDAD'], df['POBLACION'])
    mean = np.average(ages)
    median = np.median(ages)
    std = np.sqrt(np.average((df['EDAD'] - np.average(df['EDAD'], weights=df['POBLACION']))**2, weights=df['POBLACION']))

    statistics_text = f"Media: {mean:.2f}<br>Mediana: {median}<br>Desviación estándar: {std:.2f}"
    return statistics_text


# estadisticas

def histograma_poblacional(input_df, input_year, input_entidad, input_genero):
    df = input_df[(input_df['AÑO'] == input_year) & (input_df['ENTIDAD'] == input_entidad) & (input_df['SEXO'] == input_genero)].groupby('EDAD')['POBLACION'].sum().reset_index()

    # Create a weighted array for the ages
    ages = np.repeat(df['EDAD'], df['POBLACION'])

    fig = px.histogram(df, x='EDAD', y='POBLACION', nbins=25,
                       labels={'EDAD': 'Edad', 'POBLACION': 'Población total'},
                       title=f'Población por edad,<br> {input_entidad} {selected_year} ({selected_genero})',
                       template='plotly_dark',
                       color_discrete_sequence=px.colors.qualitative.Set1,
                       opacity=0.8
                       )
    
    # Set the y-axis range based on the data
    fig.update_layout(
        yaxis=dict(range=[0, df['POBLACION'].max() * 5], title='Frecuencia absoluta'),  # Adjust the multiplier as needed
        autosize=True
    )

    # Update layout for white lines
    fig.update_layout(bargap=0.02,  # Set small gap between bars
                      bargroupgap=0.1)  # Set gap between groups of bars
    
    fig.update_layout(bargap=0.02,  # Set small gap between bars
                    bargroupgap=0.1,  # Set gap between groups of bars
                    xaxis=dict(gridcolor='gray'),  # Set x-axis grid color
                    yaxis=dict(gridcolor='gray')  # Set y-axis grid color
                    )

    # Get statistics text
    statistics_text = calculate_statistics(df)

    # Add annotation with statistics text
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref='paper',
        yref='paper',
        text=statistics_text,
        showarrow=False,
        align='right',
        font=dict(color='#FFD86C')
    )
    
    fig.update_layout(title_x=0.14)

    # Update legend title
    fig.update_layout(legend_title_text='Grupos de Edad')

    fig.update_layout({'plot_bgcolor': 'black', 'paper_bgcolor': 'black'})

    fig.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'), yaxis_title_font=dict(color='#FFD86C'))

    # Show the plot
    return fig

# Example usage
fig_hist = histograma_poblacional(input_hist, selected_year, selected_entidad, selected_genero)

#### Cálculos poblacionales ######
def calculos_pob(input_df, input_year, input_genero):
    
    año_seleccionado_dato = input_df[(input_df['AÑO'] == input_year) & (input_df['SEXO'] == input_genero)].reset_index(drop=True)

    input_df_pre = input_df[input_df['AÑO'] == input_year - 1]
        # Select data for the previous year
    año_previo_dato = input_df_pre[input_df_pre['SEXO'] == input_genero].reset_index(drop=True)
        
        # Calculate the difference in population
    diferencia_poblacional = año_seleccionado_dato['POBLACION'] - año_previo_dato['POBLACION']
        
        # Calculate the absolute difference in population
    población_diferencia_absoluta = abs(diferencia_poblacional)
        
        # Calculate the growth rate with a check for zero difference
    growth_rate = np.where(
        año_previo_dato['POBLACION'] == 0,
        0,  # Set growth rate to 0 if the previous year's population is 0
        (diferencia_poblacional / np.where(año_previo_dato['POBLACION'] == 0, 1, año_previo_dato['POBLACION'])) * 100
    )

    growth_rate_round = np.round(growth_rate, 2)
    
        # Create a new DataFrame with selected columns
    result_df = pd.DataFrame({
        'AÑO': año_seleccionado_dato['AÑO'],
        "name": año_seleccionado_dato['ENTIDAD'],
        'Categoría': año_seleccionado_dato['SEXO'],
        'POBLACIÓN': año_seleccionado_dato['POBLACION'],
        'POBLACION_2023': año_previo_dato['POBLACION'],
        'POBLACION_2024': año_seleccionado_dato['POBLACION'],
        'diferencia_poblacional': diferencia_poblacional,
        'población_diferencia_absoluta': población_diferencia_absoluta,
        'Crecimiento': growth_rate_round,
        'Edad promedio':año_seleccionado_dato['EDAD_PROMEDIO']
    })
        
        # Sort the DataFrame based on the population difference
    df_población_diferencia_ordenada = result_df

    df_población_diferencia_ordenada.drop(df_población_diferencia_ordenada[df_población_diferencia_ordenada['name'] == 'República Mexicana'].index, inplace=True)
    df_población_diferencia_ordenada.loc[(df_población_diferencia_ordenada['AÑO'] == 1970) & (df_población_diferencia_ordenada['Crecimiento'] == -99.28), 'Crecimiento'] = 1.1

    return df_población_diferencia_ordenada

calculos_df = calculos_pob(input_df, selected_year, selected_genero) 


### MAPA por población ###

def mapa_poblacional(input_calculos):

    with open('mexico-with-regions_.geojson', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Ordenando por el orden en geojson
    name_to_index = {feature['properties']['name']: idx for idx, feature in enumerate(geojson_data['features'])}

    # Assuming input_calculos is your DataFrame
    # Add a new column 'index' based on the 'name' column's index in the GeoJSON file
    input_calculos['index'] = input_calculos['name'].map(name_to_index)

    # Sort the DataFrame based on the 'index' column
    input_calculos_sorted = input_calculos.sort_values(by='index')

    # Drop the 'index' column if you don't need it anymore
    input_calculos_sorted = input_calculos_sorted.drop(columns=['index'])

    state_ids = [feature['id'] for feature in geojson_data['features']]

    data = input_calculos_sorted['Edad promedio']


    fig2 = go.Figure(go.Choroplethmapbox(
        geojson=geojson_data,
        locations=state_ids,  # Use state IDs instead of names
        z=data,  # Use the generated random data for coloring
        colorscale='reds',
        zmin=min(data),
        zmax=max(data),
        marker_opacity=0.8,
        marker_line_width=0,
        customdata=input_calculos_sorted[['name', 'Edad promedio']],  # Include additional data for tooltip
    ))

    # Define the title text with a line break and font color
    title_text = f'Estados de la República Mexicana,<br>{data.name} en {selected_year} ({selected_genero})'
    title_font_color = '#FFD86C'  # Change the color to your preference

    fig2.update_layout(
        title={
            'text': title_text,
            'font': {'color': title_font_color}
        },
        mapbox_style="carto-darkmatter",
        mapbox_zoom=3,
        mapbox_center={"lat": 23.6345, "lon": -102.5528},  # Centered on Mexico
    )

    # Define the hovertemplate for the tooltip
    hover_template = (
        "<b>%{customdata[0]}</b><br>" +  # Include "Entidad" in bold
        "<extra>Edad promedio: %{customdata[1]}</extra>"  # Include "Edad promedio"
    )

    # Update traces to set hovertemplate
    fig2.update_traces(hovertemplate=hover_template)

    return fig2

mapa_poblacion_render = mapa_poblacional(calculos_df)

#### MAPA Población total ############3

def mapa_poblacional2(input_calculos):

    with open('mexico-with-regions_.geojson', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Ordenando por el orden en geojson
    name_to_index = {feature['properties']['name']: idx for idx, feature in enumerate(geojson_data['features'])}

    # Assuming input_calculos is your DataFrame
    # Add a new column 'index' based on the 'name' column's index in the GeoJSON file
    input_calculos['index'] = input_calculos['name'].map(name_to_index)

    # Sort the DataFrame based on the 'index' column
    input_calculos_sorted = input_calculos.sort_values(by='index')

    # Drop the 'index' column if you don't need it anymore
    input_calculos_sorted = input_calculos_sorted.drop(columns=['index'])

    state_ids = [feature['id'] for feature in geojson_data['features']]

    data = input_calculos_sorted['POBLACIÓN']


    fig3 = go.Figure(go.Choroplethmapbox(
        geojson=geojson_data,
        locations=state_ids,  # Use state IDs instead of names
        z=data,  # Use the generated random data for coloring
        colorscale='reds',
        zmin=min(data),
        zmax=max(data),
        marker_opacity=0.8,
        marker_line_width=0,
        customdata=input_calculos_sorted[['name', 'POBLACIÓN']],  # Include additional data for tooltip
    ))

    # Define the title text with a line break and font color
    title_text = f'Estados de la República Mexicana,<br>{data.name} en {selected_year} ({selected_genero})'
    title_font_color = '#FFD86C'  # Change the color to your preference

    fig3.update_layout(
        title={
            'text': title_text,
            'font': {'color': title_font_color}
        },
        mapbox_style="carto-darkmatter",
        mapbox_zoom=3,
        mapbox_center={"lat": 23.6345, "lon": -102.5528},  # Centered on Mexico
    )

    # Define the hovertemplate for the tooltip
    hover_template = (
        "<b>%{customdata[0]}</b><br>" +  # Include "Entidad" in bold
        "<extra>Población Total: %{customdata[1]:,.0f}</extra>"  # Include "Población" with commas
    )
    fig3.update_traces(hovertemplate=hover_template)

    return fig3

mapa_poblacion_render2 = mapa_poblacional2(calculos_df)


#### Título Dinámico ######

def titulo_dinamico(input_df, input_year, input_genero, input_entidad):
    # Select data for the current year
    cifra_poblacion = input_df[(input_df['SEXO'] == input_genero) & (input_df['AÑO'] == input_year) & (input_df['ENTIDAD'] == input_entidad)].reset_index()['POBLACION'].iloc[0]

    # Format cifra_poblacion with commas
    formatted_cifra_poblacion = '{:,}'.format(cifra_poblacion)

    ano = input_year
    genero = input_genero
    entidad = input_entidad

    # Set a yellow color for the title
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Población de {entidad} en {ano} ({genero}: {formatted_cifra_poblacion})</span>'

    return styled_title

Titulo_dinamico = titulo_dinamico(input_df, selected_year, selected_genero, selected_entidad)

# Dashboard Main Panel
st.markdown(Titulo_dinamico, unsafe_allow_html=True)
# calculos_df
# Define the tabs
tab1, tab2, tab3 = st.tabs(["Gráficas", "Mapa (Población Total)", "Mapa (Edad Promedio)"])
#C2185B
with tab1:

    chart1_col, chart2_col, chart3_col = st.columns((1, 1, 1))  # Three columns for Tab1

    with chart1_col:
        with st.expander('Perspectivas', expanded=False):
            st.markdown(f'La población de <span style="color:#C2185B">{selected_entidad}</span> seguirá enfrentando cambios radicales. La tasa de crecimiento anual en <span style="color:#C2185B">{selected_year}</span> es de <span style="color:#C2185B">{calculos_df.Crecimiento.iloc[0]}%</span>.', unsafe_allow_html=True)
            st.markdown(f'Las entidades que claramente han alcanzado su máximo poblacional y revertido su tendencia para registrar tasas decrecientes son: Ciudad de México (2019), Guerrero (2016) y Veracruz (2016).', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#C2185B">Nuevo León</span> es una de las entidades federativas que <span style="color:#C2185B">se pronostica que no alcanzará su máximo histórico en 2070 y seguirá creciendo aunque con menor aceleración</span>.', unsafe_allow_html=True)
            st.markdown(f'Se han encontrado tendencias que requieren más atención por considerarse un fenómeno atípico (no atribuible al procesamiento de los datos) o ajustes en la medición. Como son los casos de: <span style="color:#C2185B">Campeche, Chiapas, Nayarit, Durango, Quintana Roo, Sinaloa, Sonora, Tabasco, Tamaulipas y Zacatecas</span>.', unsafe_allow_html=True)

        st.plotly_chart(fig_evolucion, use_container_width=True, height=500)

    with chart2_col:
        with st.expander('Perspectivas', expanded=False):
            st.markdown(f'La edad promedio en <span style="color:#C2185B">{selected_entidad}</span> para el año <span style="color:#C2185B">{selected_year}</span> se registra en <span style="color:#C2185B">{calculos_df.loc[0, "Edad promedio"]} años</span>.', unsafe_allow_html=True)
            st.markdown(f'Claramente y sin excepción, las mujeres superan a los hombres en número.', unsafe_allow_html=True)
            st.markdown(f'Todos los estados proyectan un giro en la pirámide poblacional donde las personas más jóvenes comienzan a reducirse año con año y la población adulta, incluidos los mayores de 65 años, comienza a aumentar, lo que <span style="color:#C2185B">incrementa la tasa de dependencia (número de personas que no trabaja y tiene más de 65 años, comparada con aquellos que están en edad de trabajar)</span>.', unsafe_allow_html=True)
        
        st.plotly_chart(fig_piramide, use_container_width=True, height=500)

    with chart3_col:
        with st.expander('Perspectivas', expanded=False):
            st.markdown(f'La edad promedio en <span style="color:#C2185B">{selected_entidad}</span> para el año <span style="color:#C2185B">{selected_year}</span> es de <span style="color:#C2185B">{calculos_df.loc[0, "Edad promedio"]} años</span>. Se trata de un estadístico de tendencia central útil. No obstante, ante la existencia de datos aberrantes, se sugiere la mediana de la edad disponible en las última sección de este tablero, cuya cualidad es que es menos sensible a los datos extremos.', unsafe_allow_html=True)
            st.markdown(f'Si bien excedemos el objetivo de esta app, vale la pena señalar que la distribución por edad tiende a reducir su sesgo y comportarse como una distribución normal en periodos posteriores a 2030. Lo anterior es atribuible a factores tales como: <span style="color:#C2185B">(i) Reducción de las tasas de nacimiento; (ii) Incremento en la expectativa de vida; (iii) Reducción de las tasas de mortalidad; (iv) Factores sociales y económicos; (v) Impacto migratorio</span>.', unsafe_allow_html=True)

        
        st.plotly_chart(fig_hist, use_container_width=True, height=500)

# Define the content for tab2
with tab2:
    with st.expander('Perspectivas', expanded=False):
        st.write('''
                 - En 1970, las cinco entidades federativas más pobladas fueron: Ciudad de México (3.5 M), Estado de México (2.08 M), Veracruz (2.06 M), Jalisco (1.7 M) y Puebla (1.4 M).
                 - En 2024, la lista de las entidades federativas más pobladas es la siguiente: Estado de México (8.5 M), Ciudad de México (4.4 M), Jalisco (4.3 M), Veracruz (3.9 M) y Puebla (3.4 M).
                 - Para 2050, las trayectorias poblacionales sugieren que la lista será encabezada por: Estado de México (18.1 M), Jalisco (10.05 M), Nuevo León (8.4 M), Puebla (8.3 M) y Ciudad de México (8.01 M).
                 - Si nos preguntamos cuál debería ser la tasa de crecimiento anual promedio que cada estado debería experimentar en su población para alcanzar las predicciones de los próximos 26 años, la respuesta es la siguiente: Estado de México (2.9%), Jalisco (3.3%), Nuevo León (1.11%), Puebla (3.5%) y Ciudad de México (2.3%).
                 - Estas tasas de crecimiento poblacionales son considerablemente altas si se comparan con la media de la tasa de crecimiento anual a nivel mundial, que se espera sea del 1% durante el mismo período.                    
                 ''')

    chart1_col, chart2_col = st.columns((1, 1))  # Two columns for Tab2

    with chart1_col:
        st.plotly_chart(mapa_poblacion_render2, use_container_width=True, height=500)

    with chart2_col:
        st.plotly_chart(fig_ranking2, use_container_width=True)

# Define the content for tab3
with tab3:
    with st.expander('Perspectivas', expanded=False):
        st.write('''
                 - La mediana de la edad en 2050, a nivel mundial, se estima en 41 años.  
                 - En México, en 1970, las cinco entidades federativas con la mediana de edad más alta registrada son: Yucatán (23.4 años), Ciudad de México (22.7 años), Tlaxcala (22.5 años), Nuevo León (22.4 años) y Tamaulipas (22.3 años).
                 - En 2024, la lista de las entidades federativas con mayor mediana de edad es la siguiente: Ciudad de México (37.3 años), Veracruz (34.1 años), Morelos (33.6 años), Colima (33.6 años) y Tamaulipas (33.3 años).
                 - Para 2050, las predicciones poblacionales sugieren que la lista estará encabezada por: Ciudad de México (47.8 años), Colima (43.8 años), Veracruz (43.5 años), Morelos (43 años) y Yucatán (43.6 años), mientras que la mediana de la población en todo el país será de 40.9 años.
                 ''')

    chart1_col, chart2_col = st.columns((1, 1))  # Two columns for Tab3

    with chart1_col:
        st.plotly_chart(mapa_poblacion_render, use_container_width=True, height=500)

    with chart2_col:
        st.plotly_chart(fig_ranking, use_container_width=True)
