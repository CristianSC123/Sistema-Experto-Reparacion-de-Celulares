import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accurac1y_score

# Paso 1: Crear y ampliar el conjunto de datos (ejemplo simple)
data = {
    'sintoma': [
        'no carga', 'pantalla rota', 'no se enciende', 'batería drenada',
        'altavoz sin sonido', 'cámara borrosa', 'pantalla parpadea',
        'sobrecalentamiento', 'vibración no funciona', 'conexión wifi débil',
        'botones no responden', 'sensor de proximidad falla',
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
        'recalibrar o reemplazar sensor de proximidad',
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

# Paso 2: Convertir síntomas y soluciones a números
le_sintoma = LabelEncoder()
le_solucion = LabelEncoder()
X = le_sintoma.fit_transform(df['sintoma'])
y = le_solucion.fit_transform(df['solucion'])

# Convertir X a matriz de características (reshape para que sea un vector de características)
X = np.array(X).reshape(-1, 1)

# Paso 3: Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Usar un clasificador Random Forest (mejor opción para datos pequeños)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Paso 5: Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Paso 6: Función de predicción
def predecir_solucion(sintoma):
    sintoma_codificado = le_sintoma.transform([sintoma])[0]
    prediccion = model.predict(np.array([sintoma_codificado]).reshape(-1, 1))
    solucion_codificada = prediccion[0]
    solucion = le_solucion.inverse_transform([solucion_codificada])[0]
    return solucion

# Paso 7: Bucle para interacción con el usuario
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
