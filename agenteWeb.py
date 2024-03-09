import numpy as np
import streamlit as st

# Definir diccionario de estados
estados = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'Ñ': 14, 'O': 15, 'P': 16,
    'Q': 17, 'R': 18, 'S': 19
}

# Parámetros de entrada (interacción con el usuario)
st.title("Aplicación de Ruta Óptima")

# Ruta de la imagen descargada en tu sistema
ruta_imagen = r"C:/Users/diego/OneDrive/Escritorio/AgenteWeb/AgenteWeb/foto12.png"

# Mostrar la imagen en Streamlit
st.image(ruta_imagen, caption='Descripción de la imagen', use_column_width=True)

st.write("Ingresa el lugar de donde quieres partir, a dónde quieres ir y algunos puntos intermedios:")

inicio = st.selectbox("Lugar de partida:", list(estados.keys()))
fin = st.selectbox("Destino:", list(estados.keys()))
puntos = st.slider("Número de puntos intermedios:", 0, 10, 1)
medio = []
if puntos > 0:
    for i in range(puntos):
        medioAux = st.selectbox(f"Punto intermedio {i+1}:", list(estados.keys()))
        medio.append(medioAux)


inicioValor=0
finValor=0
medioValor = []
#Colocar dos variables que contengan la posicion correspondiente a la letra de inicio y fin
#ejemplo: inicio = "A", inicioValor = 0

if len(medio) > 0:
    for est in medio:
        for estado in estados:
            if inicio == estado:
                inicioValor = estados[estado]
            
            if(fin == estado):
                finValor = estados[estado]
            
            if(est == estado):
                medioValor.append(estados[estado])
else:
    for estado in estados:
        if inicio == estado:
            inicioValor = estados[estado]
        
        if(fin == estado):
            finValor = estados[estado]

R = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
])
R[finValor][finValor] = 10

if len(medioValor) > 0:
    i=0
    for estado in R:
        for valor in medioValor:
            if(estado[valor] == 1):
                R[i][valor] = 7
        i = i + 1
# print(R)
# Configuración de los parámetros gamma y alfa para el Q-Learning
gamma = 0.75
alpha = 0.9
# Inicialización de los valores Q
Q = np.array(np.zeros([20,20]))
# Implementación del proceso de Q-Learning
#Separar el algoritmo RL con el ambiente y generar una función llamada ruta que al final llamara el algoritmo RL e imprmirá la ruta óptima a seguir.

#Ambiente
def ruta():
    estado_actual = np.random.randint(0,20)
    accion_realizable = []
    for j in range(20):
        if R[estado_actual, j] > 0:
            accion_realizable.append(j)
    estado_siguiente = np.random.choice(accion_realizable)
    return estado_actual, accion_realizable, estado_siguiente
#Algoritmo RL
for i in range(1000):
    estado_actual, accion_realizable, estado_siguiente = ruta()
    TD = R[estado_actual, estado_siguiente] + gamma*Q[estado_siguiente, np.argmax(Q[estado_siguiente,])]- Q[estado_actual, estado_siguiente]
    Q[estado_actual, estado_siguiente] = Q[estado_actual, estado_siguiente] + alpha*TD
#ruta óptima a seguir.
# print("Q-Values:")
# print(Q.astype(int))
#Esta lista guarda el numero que representa el estado por el que debe pasar
# cor = []

# #Ciclo que obtiene la ruta a seguir de la matriz Q
# while(inicioValor != finValor):
#     i=0
#     for estado in Q:
#         if inicioValor == finValor:
#             cor.append(inicioValor)
#             break
#         if(i == inicioValor):
#             cor.append(inicioValor)
#             # print(cor)
#             mayor = max(estado)
#             j=0
#             for valor in estado:
#                 if mayor == valor:
#                     inicioValor = j
#                 j = j + 1
#         i = i + 1

# #Ciclo que cambia los valores numericos de la ruta por su letra correspondiente
# val = []
# for valor2 in cor:
#     for estado, valor in estados.items():
#         if valor == valor2:
#             val.append(estado)

# ruta = "La ruta a seguir es: "
# for i, estado in enumerate(val):
#     if i < len(val) - 1:
#         ruta = ruta + estado + " -> "
#     else:
#         ruta = ruta + estado

# #impresion de ruta a seguir
# st.write("La ruta óptima a seguir es:", ruta)
            
if st.button("Mostrar Ruta Óptima"):
    cor = []

    # Cálculo de la ruta óptima
    error = 1
    while(inicioValor != finValor):
        if(error > 150):
           break
        i=0
        for estado in Q:
            if inicioValor == finValor:
                cor.append(inicioValor)
                break
            if(i == inicioValor):
                cor.append(inicioValor)
                mayor = max(estado)
                j=0
                for valor in estado:
                    if mayor == valor:
                        inicioValor = j
                    j = j + 1
            i = i + 1
        error = error + 1

    val = []

    # Conversión de valores numéricos a letras correspondientes
    for valor2 in cor:
        for estado, valor in estados.items():
            if valor == valor2:
                val.append(estado)

    ruta = ""
    for i, estado in enumerate(val):
        if i < len(val) - 1:
            ruta = ruta + estado + " -> "
        else:
            ruta = ruta + estado

    # Impresión de la ruta a seguir
    st.write("La ruta óptima a seguir es:", ruta)