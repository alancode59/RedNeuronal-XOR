#Importamos las librerias necesarias
import tensorflow as tf
import numpy as np


# Definimos los datos de entrenamiento mediante un arreglo e instanciamos los datos de la tabla XOR
DatosInXor = np.array([[0, 0, 0], 
                    [0, 0, 1], 
                    [0, 1, 0], 
                    [0, 1, 1], 
                    [1, 0, 0], 
                    [1, 0, 1], 
                    [1, 1, 0], 
                    [1, 1, 1]])

#Definimos un arreglo el cual le pasaremos los datos de la salida esperada
Salidas = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])

#Declaramos tres funciones.

#La primer funcion se encargara de definir y construir nuestro modelo de la neurona
#Mediante la funcion declarada le instanciamos el numero de entradas y salidas
def definir_modelo(num_entradas, num_salidas):
    #Creamos la variable modelo, en ella ocuparemos keras, el cual nos ayuda a comenzar a definir el modelo secuencial
    #Ocupamos layers para indicarle que tiene 8 capas ocultas, le pasamos el parametro input_shape e instanciamos el numero de las entradas,
    #Asi como utilizamos  la activacion relu, la cual activara la neurona y hara dar valores mas cercanos
    modelo = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(num_entradas,), activation='relu'),
             
             #Asi como ocupamos las capas e instanciamos el numero de salidas,para darle la funcion de activacion sigmoid
             tf.keras.layers.Dense(num_salidas, activation='sigmoid')])
    #retornamos el resultado de modelo
    return modelo 


#Declaramos la funcion para compilar el modelo, en esa misma funcion le instanciamos la variable modelo
def compilar_modelo(modelo):
    
    #Utilizamos .compile, para compilar dicho modelo, ademas de utilizar el optimizador adam,
    #el cual calculara la combinacion lineal entre el gradiente y las incrementaciones realizadas con esto obtiene diferentes tasas de aprendizaje
    #Mediante loss, es la perdida que tiene nuestra red neuronal
    #Y metrics calcula la frecuencia de las predicciones coinciden con las etiquetas binarias de la evalucion de los datos
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#Declaramos la funcion,la cual almacenara el entrenamiento de la red neurnal
    #Le instanciamos o le pasamos como parametros la variable modelo, datos_entrada,salidas, num_epocas
def entrenar_modelo(modelo, datos_entrada, salidas, num_epocas):
    
    #Ocupamos modelo.fit para entrenar el modelo, para entrenar el modelo se le debe de pasar los datos_entrada, salidas y
    #ocupamos epochs y lo igualamos a al numero de epocas
    modelo.fit(datos_entrada, salidas, epochs=num_epocas)
    print('\n')


# Usamos las funciones que declaremos

# Declaramos modelo y llamamos a la funcion modelo e instacioamos definir modelo
# la cual tendra 3 neuronas de entrada que definen a nuestro modelo y una neurona de salida 
modelo = definir_modelo(3, 1)
#Mandamos a llamar a nuestra funcion compilar modelo
compilar_modelo(modelo)

#Mandamos a llamar a la funcion entrenar_modelo, le instanciamos modelo, los datos in de la tabla xor y las epocas
entrenar_modelo(modelo, DatosInXor, Salidas, 2500)

#Imprimimos las predicciones del modelo
print(modelo.predict(DatosInXor))
print('\n')

while True:
    entrada = []
    print('Ingresar Datos mediante teclado, (Entradas Binarias 0 y 1: )')
    for i in range(3):
        entrada.append(int(input(f"Ingrese el número {i+1}: ")))
    prediccion = modelo.predict(np.array([entrada]))
    print("Predicción:", int(prediccion[0][0] + 0.5))
    
    print('\n')
    entradas = [0, 1]
    salidas = [0, 1]
    tabla_xor = []
    for entrada_a in entradas:
        for entrada_b in entradas:
            for entrada_c in entradas:
                xor_resultado = entrada_a ^ entrada_b ^ entrada_c
                tabla_xor.append([entrada_a, entrada_b, entrada_c, xor_resultado])

    # Imprimimos la tabla XOR, para comparar resultados
    print('Tabla de verificacion XOR')
    for entrada in tabla_xor:
        entrada_a, entrada_b, entrada_c, salida = entrada
        print(f"Entradas: ({entrada_a}, {entrada_b}, {entrada_c}) - Salida: {salida}")

    
    
    
    
    




