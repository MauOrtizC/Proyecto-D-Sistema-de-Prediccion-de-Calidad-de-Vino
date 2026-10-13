
import json
from api.predict import predictor

predictor.cargar()

base = {
    "tipo": "tinto",
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

# Generar diferentes vinos para encontrar una contradiccion
vinos = []

for alcohol_i in range(90, 151, 5):
    for acidez_i in range(1, 10):
        for sulfatos_i in range(3, 15):
            vino = base.copy()
            vino["alcohol"] = alcohol_i / 10
            vino["volatile_acidity"] = acidez_i / 10
            vino["sulphates"] = sulfatos_i / 10
            vinos.append(vino)

# Consultar los dos modelos reales
calidades = predictor.predecir_calidad(vinos)
probabilidades = predictor.predecir_calidad_alta(vinos)

encontrado = False

for vino, calidad, prob in zip(vinos, calidades, probabilidades):
    if calidad >= 6.8 and prob < 0.5:
        print("\nCASO ENCONTRADO")
        print("\nDatos del vino:")
        print(json.dumps(vino, indent=2, ensure_ascii=False))
        print(f"\nRegresion: {calidad:.2f}/10")
        print(f"Clasificacion: {prob * 100:.2f}%")
        print("Codigo anterior: Calidad alta")
        print("Codigo corregido: Calidad no alta")
        encontrado = True
        break

if not encontrado:
    print("No se encontro un caso en estas combinaciones.")
    print("Esto no significa que la condicion anterior fuera correcta.")
