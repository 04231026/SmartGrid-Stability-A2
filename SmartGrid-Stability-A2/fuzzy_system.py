
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =====================================================
# 1. MEMBUAT VARIABEL FUZZY
# =====================================================

probabilitas = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01),
    "probabilitas"
)

kesepakatan = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01),
    "kesepakatan"
)

risiko = ctrl.Consequent(
    np.arange(0, 101, 1),
    "risiko"
)


# =====================================================
# 2. MEMBERSHIP FUNCTION INPUT 1
# =====================================================

probabilitas["rendah"] = fuzz.trapmf(
    probabilitas.universe,
    [0, 0, 0.15, 0.35]
)

probabilitas["sedang"] = fuzz.trimf(
    probabilitas.universe,
    [0.20, 0.50, 0.80]
)

probabilitas["tinggi"] = fuzz.trapmf(
    probabilitas.universe,
    [0.65, 0.85, 1, 1]
)


# =====================================================
# 3. MEMBERSHIP FUNCTION INPUT 2
# =====================================================

kesepakatan["rendah"] = fuzz.trapmf(
    kesepakatan.universe,
    [0, 0, 0.20, 0.45]
)

kesepakatan["sedang"] = fuzz.trimf(
    kesepakatan.universe,
    [0.25, 0.50, 0.75]
)

kesepakatan["tinggi"] = fuzz.trapmf(
    kesepakatan.universe,
    [0.55, 0.80, 1, 1]
)


# =====================================================
# 4. MEMBERSHIP FUNCTION OUTPUT
# =====================================================

risiko["rendah"] = fuzz.trapmf(
    risiko.universe,
    [0, 0, 20, 45]
)

risiko["sedang"] = fuzz.trimf(
    risiko.universe,
    [30, 50, 70]
)

risiko["tinggi"] = fuzz.trapmf(
    risiko.universe,
    [55, 80, 100, 100]
)


# =====================================================
# 5. FUZZY RULE BASE
# =====================================================

rule1 = ctrl.Rule(
    probabilitas["rendah"] & kesepakatan["rendah"],
    risiko["sedang"]
)

rule2 = ctrl.Rule(
    probabilitas["rendah"] & kesepakatan["sedang"],
    risiko["rendah"]
)

rule3 = ctrl.Rule(
    probabilitas["rendah"] & kesepakatan["tinggi"],
    risiko["rendah"]
)

rule4 = ctrl.Rule(
    probabilitas["sedang"] & kesepakatan["rendah"],
    risiko["tinggi"]
)

rule5 = ctrl.Rule(
    probabilitas["sedang"] & kesepakatan["sedang"],
    risiko["sedang"]
)

rule6 = ctrl.Rule(
    probabilitas["sedang"] & kesepakatan["tinggi"],
    risiko["sedang"]
)

rule7 = ctrl.Rule(
    probabilitas["tinggi"] & kesepakatan["rendah"],
    risiko["tinggi"]
)

rule8 = ctrl.Rule(
    probabilitas["tinggi"] & kesepakatan["sedang"],
    risiko["tinggi"]
)

rule9 = ctrl.Rule(
    probabilitas["tinggi"] & kesepakatan["tinggi"],
    risiko["tinggi"]
)


# =====================================================
# 6. MEMBANGUN SISTEM MAMDANI
# =====================================================

risk_control = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9
])


# =====================================================
# 7. FUNGSI PREDIKSI FUZZY
# =====================================================

def fuzzy_prediction(probability, agreement):

    simulation = ctrl.ControlSystemSimulation(
        risk_control
    )

    simulation.input["probabilitas"] = float(
        np.clip(probability, 0, 1)
    )

    simulation.input["kesepakatan"] = float(
        np.clip(agreement, 0, 1)
    )

    simulation.compute()

    risk_value = simulation.output["risiko"]

    if risk_value < 40:
        category = "Risiko Rendah"

    elif risk_value < 70:
        category = "Risiko Sedang"

    else:
        category = "Risiko Tinggi"

    return {
        "probabilitas": float(probability),
        "kesepakatan": float(agreement),
        "risk_score": float(risk_value),
        "category": category
    }


# =====================================================
# 8. PENGUJIAN BEBERAPA SKENARIO
# =====================================================

if __name__ == "__main__":

    scenarios = [
        (0.10, 0.90),
        (0.45, 0.80),
        (0.85, 0.95),
        (0.80, 0.30),
        (0.50, 0.50)
    ]

    print("=" * 55)
    print("PENGUJIAN FUZZY LOGIC MAMDANI")
    print("=" * 55)

    for index, (prob, agree) in enumerate(
        scenarios, start=1
    ):

        result = fuzzy_prediction(prob, agree)

        print(f"\nSkenario {index}")
        print(
            f"Probabilitas tidak stabil: "
            f"{prob:.2f}"
        )

        print(
            f"Kesepakatan model: "
            f"{agree:.2f}"
        )

        print(
            f"Skor risiko: "
            f"{result['risk_score']:.2f}"
        )

        print(
            f"Keputusan: "
            f"{result['category']}"
        )

    print("\nPengujian Fuzzy selesai.")
