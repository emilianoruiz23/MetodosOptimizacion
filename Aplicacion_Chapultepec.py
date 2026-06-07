import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.optimize import linprog

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Optimización Chapultepec - MAC UNAM", layout="wide")
st.title("🌲 Optimización de Rutas: Bosque de Chapultepec")
st.markdown("**Proyecto de Análisis de Redes | Emiliano Ruiz Sánchez y Ricardo López Ramírez**")

# --- DATOS DEL MODELO ---
@st.cache_data
def cargar_grafo():
    nodos_info = {
        'N1': 'Lago', 'N2': 'Casa del Lago', 'N3': 'Zoo Aventuras',
        'N4': 'Zoológico', 'N5': 'Museo Axolote', 'N6': 'Herpetario',
        'N7': 'Jardín Botánico', 'N8': 'Orquideario', 'N9': 'Castillo',
        'N10': 'Ahuehuete', 'N11': 'Semi Lago', 'N12': 'F. Quijote',
        'N13': 'Sor Juana', 'N14': 'F. Ranas', 'N15': 'Aviario'
    }
    
    aristas = [
        ('N1', 'N2', 43), ('N2', 'N3', 91), ('N3', 'N4', 250), ('N4', 'N5', 51),
        ('N5', 'N6', 154), ('N6', 'N7', 670), ('N7', 'N8', 90), ('N8', 'N9', 270),
        ('N9', 'N10', 100), ('N10', 'N11', 190), ('N11', 'N12', 240), ('N12', 'N13', 200),
        ('N13', 'N14', 380), ('N14', 'N15', 300), ('N2', 'N4', 200), ('N3', 'N5', 120), 
        ('N4', 'N6', 180), ('N5', 'N7', 220), ('N6', 'N8', 150), ('N7', 'N9', 200), 
        ('N8', 'N10', 210), ('N9', 'N11', 170), ('N10', 'N12', 160), ('N11', 'N13', 140), 
        ('N12', 'N14', 180), ('N13', 'N15', 220), ('N1', 'N3', 130), ('N2', 'N5', 160), 
        ('N4', 'N7', 300), ('N6', 'N9', 250), ('N8', 'N11', 230), ('N10', 'N13', 210), 
        ('N11', 'N14', 190), ('N12', 'N15', 260)
    ]
    
    G = nx.Graph()
    for origen, destino, peso in aristas:
        G.add_edge(origen, destino, weight=peso)
    return G, nodos_info

G, dict_nodos = cargar_grafo()

mapa_colores = {
    'N1': '#87CEFA', 'N2': '#87CEFA', 'N11': '#87CEFA',
    'N3': '#FFB347', 'N4': '#FFB347', 'N5': '#FFB347', 'N6': '#FFB347', 'N15': '#FFB347',
    'N7': '#98FB98', 'N8': '#98FB98', 'N10': '#98FB98',
    'N9': '#DDA0DD', 'N12': '#DDA0DD', 'N13': '#DDA0DD', 'N14': '#DDA0DD'
}

posiciones = {
    'N1': (0, 10), 'N2': (3, 15), 'N3': (5, 20), 'N4': (6, 10),
    'N5': (10, 18), 'N6': (14, 20), 'N7': (19, 14), 'N8': (18, 8),
    'N9': (15, 0), 'N10': (11, 6), 'N11': (8, 2), 'N12': (5, 5),
    'N13': (2, 0), 'N14': (0, -6), 'N15': (6, -8)
}

# --- SIDEBAR: NAVEGACIÓN ---
menu = st.sidebar.radio(
    "Menú de Proyecto:",
    (
        "1. Animación de la Red", 
        "2. Ruta Más Corta (Dijkstra)", 
        "3. Matriz de Rutas (Floyd-Warshall)", 
        "4. Análisis de Sensibilidad", 
        "5. Programación Lineal Entera (PLE)",
        "6. Conclusiones y Referencias"
    )
)

# --- REUTILIZABLE: SELECTOR DE NODOS ---
def selector_nodos(clave_origen, clave_destino, idx_origen=0, idx_destino=8):
    opciones = [f"{k} - {v}" for k, v in dict_nodos.items()]
    c1, c2 = st.columns(2)
    with c1: or_sel = st.selectbox("Origen:", opciones, index=idx_origen, key=clave_origen)
    with c2: des_sel = st.selectbox("Destino:", opciones, index=idx_destino, key=clave_destino)
    return or_sel.split(" - ")[0], des_sel.split(" - ")[0]

# --- 1. ANIMACIÓN DE LA RED ---
if menu == "1. Animación de la Red":
    st.header("Visualización Animada de la Red")
    st.write("Construcción secuencial de los arcos y nodos que componen la topología del Bosque de Chapultepec.")
    
    col_mapa, col_leyenda = st.columns([3, 1])
    with col_mapa:
        if st.button("▶️ Iniciar Animación de Red"):
            plot_placeholder = st.empty()
            nodos_ordenados = sorted(G.nodes(), key=lambda x: int(x[1:]))
            
            for i in range(1, len(nodos_ordenados) + 1):
                nodos_visibles = nodos_ordenados[:i]
                nodo_actual = nodos_ordenados[i-1]
                
                for alfa in [0.2, 0.6, 1.0]:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    nodos_previos = nodos_ordenados[:i-1]
                    
                    if nodos_previos:
                        nx.draw_networkx_nodes(G, posiciones, nodelist=nodos_previos, 
                                               node_color=[mapa_colores[n] for n in nodos_previos],
                                               node_size=800, edgecolors='black', ax=ax)
                    
                    nx.draw_networkx_nodes(G, posiciones, nodelist=[nodo_actual], 
                                           node_color=[mapa_colores[nodo_actual]],
                                           node_size=800, edgecolors='black', alpha=alfa, ax=ax)
                    
                    G_sub = G.subgraph(nodos_visibles)
                    nx.draw_networkx_edges(G_sub, posiciones, edge_color='gray', width=1.2, alpha=alfa * 0.5, ax=ax)
                    nx.draw_networkx_labels(G_sub, posiciones, font_size=10, font_weight='bold', ax=ax)
                    
                    labels = nx.get_edge_attributes(G_sub, 'weight')
                    nx.draw_networkx_edge_labels(G_sub, posiciones, edge_labels=labels, font_size=8, font_color='red', ax=ax)
                    
                    ax.set_xlim(-2, 21); ax.set_ylim(-10, 22); plt.axis('off')
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(0.02)
            st.success("Red completamente renderizada.")
        else:
            st.info("Haz clic en el botón superior para generar la red.")

    with col_leyenda:
        st.subheader("Nomenclatura")
        st.dataframe(pd.DataFrame(list(dict_nodos.items()), columns=["ID", "Lugar"]), hide_index=True)

# --- 2. DIJKSTRA ---
elif menu == "2. Ruta Más Corta (Dijkstra)":
    st.header("📍 Algoritmo de Dijkstra")
    st.write("Determinación de la ruta de costo mínimo basada en etiquetado permanente.")
    
    u, v = selector_nodos("dijk_or", "dijk_des")
    
    if u != v:
        try:
            ruta = nx.shortest_path(G, source=u, target=v, weight='weight')
            costo = nx.shortest_path_length(G, source=u, target=v, weight='weight')
            
            st.success(f"**Distancia mínima óptima:** {costo} metros")
            st.write("**Secuencia:** " + " ➡️ ".join([f"**{n} ({dict_nodos[n]})**" for n in ruta]))
            
            fig, ax = plt.subplots(figsize=(14, 9))
            nx.draw_networkx_nodes(G, posiciones, node_color='#E0E0E0', node_size=500, edgecolors='white', ax=ax)
            nx.draw_networkx_edges(G, posiciones, edge_color='#E0E0E0', width=1.0, ax=ax)
            nx.draw_networkx_labels(G, posiciones, font_size=9, font_color='gray', ax=ax)
            
            aristas_ruta = list(zip(ruta, ruta[1:]))
            nx.draw_networkx_nodes(G, posiciones, nodelist=ruta, node_color=[mapa_colores[n] for n in ruta], node_size=900, edgecolors='black', linewidths=2, ax=ax)
            nx.draw_networkx_edges(G, posiciones, edgelist=aristas_ruta, edge_color='red', width=3.5, ax=ax)
            nx.draw_networkx_labels(G, posiciones, labels={n: n for n in ruta}, font_size=11, font_weight='bold', ax=ax)
            
            edge_labels = nx.get_edge_attributes(G, 'weight')
            path_edge_labels = {(o, d): edge_labels.get((o, d), edge_labels.get((d, o))) for o, d in aristas_ruta}
            nx.draw_networkx_edge_labels(G, posiciones, edge_labels=path_edge_labels, font_size=10, font_color='red', font_weight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'), ax=ax)
            
            ax.set_xlim(-2, 21); ax.set_ylim(-10, 22); plt.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        except nx.NetworkXNoPath:
            st.error("No existe interconexión viable entre los puntos seleccionados.")
    else:
        st.warning("El nodo de origen debe ser distinto al de destino.")

# --- 3. FLOYD-WARSHALL ---
elif menu == "3. Matriz de Rutas (Floyd-Warshall)":
    st.header("📊 Algoritmo de Floyd-Warshall")
    st.write("Cálculo simultáneo de caminos mínimos entre todos los pares de nodos de la red.")
    
    nodos_lista = sorted(G.nodes(), key=lambda x: int(x[1:]))
    n = len(nodos_lista)
    nodo_to_idx = {nodo: i for i, nodo in enumerate(nodos_lista)}
    
    D = np.full((n, n), np.inf)
    P = np.full((n, n), "", dtype=object)
    
    for i in range(n):
        D[i][i] = 0
        P[i][i] = "-"
        
    for u_edge, v_edge, data in G.edges(data=True):
        i, j = nodo_to_idx[u_edge], nodo_to_idx[v_edge]
        peso = data['weight']
        D[i][j] = peso; D[j][i] = peso
        P[i][j] = v_edge; P[j][i] = u_edge
        
    historial_D, historial_P = [D.copy()], [P.copy()]
    
    for k in range(n):
        D_k, P_k = historial_D[-1].copy(), historial_P[-1].copy()
        for i in range(n):
            for j in range(n):
                if D_k[i][k] + D_k[k][j] < D_k[i][j]:
                    D_k[i][j] = D_k[i][k] + D_k[k][j]
                    P_k[i][j] = P_k[i][k]
        historial_D.append(D_k); historial_P.append(P_k)
        
    k_sel = st.slider("Iteración matricial ($k$):", min_value=0, max_value=n, value=0)
    
    col_D, col_P = st.columns(2)
    with col_D:
        st.subheader(f"Matriz de Costos $D^{{({k_sel})}}$")
        df_D_disp = pd.DataFrame(historial_D[k_sel], index=nodos_lista, columns=nodos_lista).replace(np.inf, np.nan)
        st.dataframe(df_D_disp.style.format(na_rep='inf', precision=0).background_gradient(cmap='Greens', axis=None).highlight_null(color='lightgray'), use_container_width=True, height=450)
    with col_P:
        st.subheader(f"Matriz de Secuencias $P^{{({k_sel})}}$")
        df_P = pd.DataFrame(historial_P[k_sel], index=nodos_lista, columns=nodos_lista)
        def color_rutas(val): return f'background-color: {mapa_colores[val]}; color: black' if val in mapa_colores else 'background-color: #f0f2f6; color: gray'
        try: styler_P = df_P.style.map(color_rutas)
        except AttributeError: styler_P = df_P.style.applymap(color_rutas)
        st.dataframe(styler_P, use_container_width=True, height=450)

# --- 4. ANALISIS DE SENSIBILIDAD ---
elif menu == "4. Análisis de Sensibilidad":
    st.header("⚠️ Análisis de Sensibilidad ('What-If')")
    
    escenario = st.radio("Configuración del Escenario:", ["Caso 1: Bloqueo Estructural (Mantenimiento N6 - N7)", "Caso 2: Fluctuación de Costos (Saturación Peatonal N1 - N3)"])
    
    if escenario == "Caso 1: Bloqueo Estructural (Mantenimiento N6 - N7)":
        st.subheader("🚧 Interrupción del Arco N6 - N7")
        bloqueo = st.checkbox("Simular Clausura Física del Sendero")
        G_s = G.copy()
        
        c_base = nx.shortest_path_length(G, 'N6', 'N9', weight='weight')
        if bloqueo: 
            G_s.remove_edge('N6', 'N7')
        
        try:
            ruta = nx.shortest_path(G_s, source='N6', target='N9', weight='weight')
            costo = nx.shortest_path_length(G_s, source='N6', target='N9', weight='weight')
            
            c1, c2 = st.columns(2)
            c1.metric("Longitud del Recorrido (N6 ➡️ N9)", f"{costo} m", delta=f"+{costo - c_base} m (Desvío)" if bloqueo else "0 m")
            c2.write(f"**Camino Activo:** " + " ➡️ ".join([f"**{n}**" for n in ruta]))
            
            fig, ax = plt.subplots(figsize=(14, 9))
            nx.draw_networkx_nodes(G, posiciones, node_color='#E0E0E0', node_size=500, edgecolors='white', ax=ax)
            nx.draw_networkx_edges(G, posiciones, edge_color='#E0E0E0', width=1.0, ax=ax)
            
            aristas_ruta = list(zip(ruta, ruta[1:]))
            colores_ruta = [mapa_colores[n] for n in ruta]
            nx.draw_networkx_nodes(G, posiciones, nodelist=ruta, node_color=colores_ruta, node_size=800, edgecolors='black', linewidths=2, ax=ax)
            nx.draw_networkx_edges(G, posiciones, edgelist=aristas_ruta, edge_color='red', width=3.5, ax=ax)
            nx.draw_networkx_labels(G, posiciones, labels={n: n for n in ruta}, font_size=10, font_weight='bold', ax=ax)
            
            edge_labels = nx.get_edge_attributes(G_s, 'weight')
            path_edge_labels = { (u, v): edge_labels.get((u,v), edge_labels.get((v,u))) for u, v in aristas_ruta }
            nx.draw_networkx_edge_labels(G_s, posiciones, edge_labels=path_edge_labels, font_color='red', font_weight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'), ax=ax)
            
            if bloqueo and G.has_edge('N6', 'N7'):
                nx.draw_networkx_edges(G, posiciones, edgelist=[('N6', 'N7')], edge_color='red', width=2.0, style='dashed', ax=ax)
                x_mid, y_mid = (posiciones['N6'][0] + posiciones['N7'][0])/2, (posiciones['N6'][1] + posiciones['N7'][1])/2 + 0.8
                ax.text(x_mid, y_mid, "❌ CERRADO", color='red', fontsize=11, ha='center', va='center', backgroundcolor='white', fontweight='bold')
            
            ax.set_xlim(-2, 21); ax.set_ylim(-10, 22); plt.axis('off')
            st.pyplot(fig)
            
        except nx.NetworkXNoPath: st.error("Incomunicación crítica.")
        
    else:
        st.subheader("🚶‍♂️ Incremento del Costo en Arco N1 - N3")
        congestion = st.checkbox("Simular Saturación de Fin de Semana (Costo 130m ➡️ 450m)")
        G_s = G.copy()
        
        c_base = nx.shortest_path_length(G, 'N1', 'N5', weight='weight')
        if congestion: 
            G_s['N1']['N3']['weight'] = 450
            
        ruta = nx.shortest_path(G_s, source='N1', target='N5', weight='weight')
        costo = nx.shortest_path_length(G_s, source='N1', target='N5', weight='weight')
        
        c1, c2 = st.columns(2)
        c1.metric("Longitud del Recorrido (N1 ➡️ N5)", f"{costo} m", delta=f"+{costo - c_base} m (Desvío Inteligente)" if congestion else "0 m")
        c2.write(f"**Camino Activo:** " + " ➡️ ".join([f"**{n}**" for n in ruta]))

        fig, ax = plt.subplots(figsize=(14, 9))
        nx.draw_networkx_nodes(G, posiciones, node_color='#E0E0E0', node_size=500, edgecolors='white', ax=ax)
        nx.draw_networkx_edges(G, posiciones, edge_color='#E0E0E0', width=1.0, ax=ax)
        
        if congestion:
            nx.draw_networkx_edges(G, posiciones, edgelist=[('N1', 'N3')], edge_color='darkorange', width=4.0, ax=ax)
            x_mid, y_mid = (posiciones['N1'][0] + posiciones['N3'][0])/2, (posiciones['N1'][1] + posiciones['N3'][1])/2 + 1.2
            ax.text(x_mid, y_mid, "⚠️ SATURADO (450m)", color='darkorange', fontsize=11, ha='center', va='center', backgroundcolor='white', fontweight='bold')
        
        aristas_ruta = list(zip(ruta, ruta[1:]))
        colores_ruta = [mapa_colores[n] for n in ruta]
        
        nx.draw_networkx_nodes(G, posiciones, nodelist=ruta, node_color=colores_ruta, node_size=800, edgecolors='black', linewidths=2, ax=ax)
        nx.draw_networkx_edges(G, posiciones, edgelist=aristas_ruta, edge_color='red', width=3.5, ax=ax)
        nx.draw_networkx_labels(G, posiciones, labels={n: n for n in ruta}, font_size=10, font_weight='bold', ax=ax)
        
        edge_labels = nx.get_edge_attributes(G_s, 'weight')
        path_edge_labels = { (u, v): edge_labels.get((u,v), edge_labels.get((v,u))) for u, v in aristas_ruta }
        nx.draw_networkx_edge_labels(G_s, posiciones, edge_labels=path_edge_labels, font_color='red', font_weight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'), ax=ax)
        
        ax.set_xlim(-2, 21); ax.set_ylim(-10, 22); plt.axis('off')
        st.pyplot(fig)

# --- 5. PROGRAMACION LINEAL ENTERA (PLE) ---
elif menu == "5. Programación Lineal Entera (PLE)":
    st.header("📐 Programación Lineal Entera (PLE)")
    st.write("Resolución formal del problema empleando ecuaciones algebraicas explícitas y variables estrictamente enteras ($x_{ij} \in \{0,1\}$).")
    
    u_ple, v_ple = selector_nodos("ple_or", "ple_des")
    
    if u_ple != v_ple:
        # Estructuración matemática del modelo de flujo
        edges_dir = []
        for u_edge, v_edge, data in G.edges(data=True):
            edges_dir.append((u_edge, v_edge, data['weight']))
            edges_dir.append((v_edge, u_edge, data['weight']))
            
        num_edges = len(edges_dir)
        nodos_lista = sorted(list(G.nodes()), key=lambda x: int(x[1:]))
        nodo_to_idx = {nodo: i for i, nodo in enumerate(nodos_lista)}
        
        # Función Objetivo: c^T * x
        c = [peso for _, _, peso in edges_dir]
        
        # Restricciones de Igualdad: A_eq * x = b_eq
        A_eq = np.zeros((len(nodos_lista), num_edges))
        b_eq = np.zeros(len(nodos_lista))
        
        for idx_n, nodo in enumerate(nodos_lista):
            for idx_e, (o, d, _) in enumerate(edges_dir):
                if o == nodo: A_eq[idx_n, idx_e] = 1   # Flujo de Salida
                if d == nodo: A_eq[idx_n, idx_e] = -1  # Flujo de Entrada
                
        b_eq[nodo_to_idx[u_ple]] = 1
        b_eq[nodo_to_idx[v_ple]] = -1
        
        # Restricción de Integraleza: Variables binarias {0, 1}
        integrality = np.ones(num_edges) 
        bounds = [(0, 1) for _ in range(num_edges)]
        
        # Ejecución del Optimizador (Método Highs)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality, method='highs')
        
        if res.success:
            costo_ple = int(round(res.fun))
            st.success(f"**Solución entera óptima calculada:** {costo_ple} metros")
            
            # Reconstrucción del camino óptimo a partir del vector x
            aristas_activas = [edges_dir[i] for i in range(num_edges) if res.x[i] > 0.5]
            
            st.markdown("### 🔍 Variables de Decisión Activas ($x_{ij} = 1$):")
            for o, d, w in aristas_activas:
                st.markdown(f"- $x_{{{o},{d}}} = 1$: Se transita del nodo **{o}** al nodo **{d}** (Distancia: {w}m)")
                
            st.markdown("### 📝 Formulación Estructural Ejecutada:")
            st.latex(r"\min Z = \sum_{(i,j) \in A} c_{ij} x_{ij}")
            st.markdown("**Sujeto a restricciones de conservación:**")
            st.latex(r"\sum_{j: (i,j) \in A} x_{ij} - \sum_{j: (j,i) \in A} x_{ji} = \begin{cases} 1, & i = \text{Origen} \\ -1, & i = \text{Destino} \\ 0, & \text{en otro caso} \end{cases}")
            st.latex(r"x_{ij} \in \{0, 1\} \quad \forall (i,j) \in A")
        else:
            st.error("El solucionador de programación entera no encontró una solución factible.")
    else:
        st.warning("Selecciona nodos distintos para estructurar las restricciones de flujo.")

# --- 6. CONCLUSIONES Y REFERENCIAS ---
elif menu == "6. Conclusiones y Referencias":
    st.header("🏁 Evaluación General del Proyecto")
    
    st.subheader("Tabla Comparativa de Técnicas")
    data_comp = {
        "Métrica / Criterio": ["Tipo de Algoritmo", "Complejidad Operativa", "Garantía de Optimalidad", "Sensibilidad al Escenario", "Formato de Salida"],
        "Dijkstra": ["Algoritmo Codicioso (Greedy)", "Baja: $O(V^2)$ o $O(E + V \log V)$", "Exacta (Sin pesos negativos)", "Requiere recalcular desde cero", "Secuencia de nodos única"],
        "Floyd-Warshall": ["Programación Dinámica", "Media-Alta: $O(V^3)$", "Exacta (Permite pesos negativos)", "Muestra impactos globales", "Matrices de distancias completas"],
        "Programación Entera (PLE)": ["Optimización Matemática", "Alta (NP-hard en general)", "Exacta Global", "Modificación matemática de vectores", "Flujo binario por arcos"]
    }
    st.dataframe(pd.DataFrame(data_comp), hide_index=True, use_container_width=True)
    
    st.subheader("Conclusiones Académicas")
    st.markdown("""
    El análisis comparativo de la red del Bosque de Chapultepec demuestra que, para problemas de rutas uniorigen-unidestino estándar, el algoritmo de **Dijkstra** destaca por su eficiencia y velocidad computacional. Sin embargo, al enfrentar problemas de planeación logística integral donde se requiere conocer las interconexiones totales de la red, **Floyd-Warshall** proporciona una infraestructura matricial robusta ideal para decisiones centralizadas. 
    
    Por otro lado, la **Programación Lineal Entera (PLE)** abre las puertas a modelados mucho más complejos, ya que permite añadir restricciones operativas del mundo real (como capacidades de flujo en senderos, sentidos de circulación únicos o ventanas de tiempo) que los algoritmos tradicionales de grafos no pueden asimilar directamente, consolidándose como la herramienta más flexible y escalable de la optimización matemática contemporánea.
    """)
    
    st.subheader("📚 Referencias (Formato APA 7)")
    st.markdown("""
    * Bazaraa, M. S., Jarvis, J. J., & Sherali, H. D. (2010). *Linear programming and network flows* (4th ed.). John Wiley & Sons.
    * Hillier, F. S., & Lieberman, G. J. (2015). *Introducción a la investigación de operaciones* (10a ed.). McGraw-Hill.
    * Taha, H. A. (2017). *Investigación de operaciones* (10a ed.). Pearson Educación.
    """)
