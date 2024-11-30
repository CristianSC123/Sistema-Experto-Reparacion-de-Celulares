import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Paso 1: Cargar y preprocesar los datos (datos ficticios de ejemplo)
# Crear un DataFrame con los síntomas y la solución (Ajustar según dataset real)
data = {
    'sintoma': [
        'no carga', 'pantalla rota', 'no se enciende', 'batería drenada',
        'altavoz sin sonido', 'cámara borrosa', 'pantalla parpadea',
        'sobrecalentamiento', 'vibración no funciona', 'conexión wifi débil',
        'botones no responden', 'sensor de proximidad falla'
    ],
    'solucion': [
        'reemplazar puerto de carga', 
        'cambiar pantalla completa',
        'revisar placa base y circuito de encendido',
        'reemplazar batería y verificar consumo de energía',
        'revisar y cambiar altavoz',
        'limpiar lente de cámara y revisar sensor',
        'revisar conectores de pantalla y cable flex',
        'limpiar ventilación y verificar disipador de calor',
        'revisar motor de vibración y conexiones',
        'revisar antena wifi y conectores',
        'revisar y limpiar contactos de botones',
        'recalibrar o reemplazar sensor de proximidad'
    ]
}

df = pd.DataFrame(data)

# Convertir síntomas y soluciones en números
le_sintoma = LabelEncoder()
le_solucion = LabelEncoder()
X = le_sintoma.fit_transform(df['sintoma'])
y = le_solucion.fit_transform(df['solucion'])

# Convertir X a matriz de características
X = np.array(X).reshape(-1, 1)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Definir la arquitectura de la red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])


# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Paso 3: Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

# Paso 4: Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Precisión del modelo: {accuracy:.2f}')

# Paso 5: Predicción de ejemplo
def predecir_solucion(sintoma):
    sintoma_codificado = le_sintoma.transform([sintoma])[0]
    prediccion = model.predict(np.array([sintoma_codificado]).reshape(-1, 1))
    solucion_codificada = np.argmax(prediccion)
    solucion = le_solucion.inverse_transform([solucion_codificada])[0]
    return solucion

while True:
    print("-----------------------------------")        
    print("1.-  No carga")
    print("2.-  Pantalla rota")
    print("3.-  No se enciende")
    print("4.-  Batería drenada")
    print("5.-  Altavoz sin sonido")
    print("6.-  Cámara borrosa")
    print("7.-  Pantalla parpadea")
    print("8.-  Sobrecalentamiento")
    print("9.-  Vibración no funciona")
    print("10.- Conexión wifi débil")
    print("11.- Botones no responden")
    print("12.- Sensor de proximidad falla")
    print("-----------------------------------")
    
    num = int(input("Ingrese un número: "))
    
    if num == 1:
        sintoma_entrada = 'no carga'
    elif num == 2:
        sintoma_entrada = 'pantalla rota'
    elif num == 3:
        sintoma_entrada = 'no se enciende'
    elif num == 4:
        sintoma_entrada = 'batería drenada'
    elif num == 5:
        sintoma_entrada = 'altavoz sin sonido'
    elif num == 6:
        sintoma_entrada = 'cámara borrosa'
    elif num == 7:
        sintoma_entrada = 'pantalla parpadea'
    elif num == 8:
        sintoma_entrada = 'sobrecalentamiento'
    elif num == 9: 
        sintoma_entrada = 'vibración no funciona'
    elif num == 10:
        sintoma_entrada = 'conexión wifi débil'
    elif num == 11:
        sintoma_entrada = 'botones no responden'
    elif num == 12:
        sintoma_entrada = 'sensor de proximidad falla'

    # Mostrar la solución sugerida
    print(f"Para el síntoma '{sintoma_entrada}', la solución sugerida es: {predecir_solucion(sintoma_entrada)}")
    
    yn = input("¿Desea repetir? (y/n): ").lower()
    
    if yn != 'y':
        print("Gracias por usar el sistema. ¡Hasta luego!")
        break