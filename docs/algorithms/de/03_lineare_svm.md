# Lineare SVM – ein Klassifikator, der die breiteste Grenze zieht

## Das Problem

Wie bei der [Logistischen Regression](02_logistische_regression.md): Jeder Prompt ist eine Liste von
Zahlen aus [TF-IDF](01_tfidf.md), und wir wollen entscheiden: **allow**, **warn** oder **block**.

## Die Idee in einem Satz

> Eine lineare Support Vector Machine (SVM) zieht eine **Grenze** zwischen den Klassen und legt sie so,
> dass der **Abstand zu den nächsten Beispielen auf beiden Seiten möglichst groß** ist.

## Ein Bild im Kopf

Stell dir rote Punkte (*block*) und grüne Punkte (*allow*) auf einem Blatt Papier vor.
Es gibt viele gerade Linien, die sie trennen. Die SVM wählt die Linie mit der **breitesten leeren
Straße** zwischen den beiden Gruppen.

```
   grün ●  ●                 
      ●   ●     ┆           ┆         
     ●  ●       ┆  Grenze   ┆   ▲ ▲      
                ┆     │     ┆  ▲  ▲ rot
                ┆     │     ┆ ▲   ▲      
                ← breiter Abstand →
```

Warum? Ein breiter Abstand macht das Modell **robust**: Ein neuer Prompt, der etwas anders aussieht,
landet trotzdem auf der richtigen Seite.

Die Beispiele, die am nächsten an der Grenze liegen, heißen **Support-Vektoren** (Stützvektoren). Nur
sie bestimmen, wo die Grenze liegt – daher der Name.

Bei drei Klassen trainiert scikit-learn **eine Grenze pro Klasse** („diese Klasse gegen alle
anderen“) und nimmt die Klasse mit dem höchsten Punktwert.

## Punktwert statt Wahrscheinlichkeit

Die Logistische Regression sagt „90 % block“. Die SVM gibt stattdessen einen **Punktwert** aus:
- Punktwert > 0 → auf der Block-Seite der Grenze
- Punktwert < 0 → auf der anderen Seite
- je weiter weg von 0, desto sicherer

## Die zwei Stellschrauben, die wir getuned haben

**1. Regularisierung `C`** – *Wie streng ist die Grenze?*
- kleines `C` → breiter Abstand, erlaubt beim Training einige Fehler (robuster)
- großes `C` → versucht jedes Trainingsbeispiel richtig einzuordnen (Gefahr von Overfitting)
- Unser bester Wert: **C = 1,0**

**2. Block-Schwelle** – *Ab welchem Punktwert blockieren wir?*
Wir blockieren schon ab einem Punktwert von **−0,4**, also kurz vor der Grenze. Wie die 20-%-Schwelle
bei der Logistischen Regression fängt das mehr gefährliche Prompts ab und nimmt ein paar mehr
Fehlalarme in Kauf.

## Unsere Ergebnisse (Validierungs-Set)

| Version | Macro F1 | Recall block | Latenz p95 |
|---|---|---|---|
| Lineare SVM getuned | 0,865 | **0,956** | 3,4 ms |
| Lineare SVM + harmlose Prompts | 0,862 | **0,957** | 3,5 ms |

Die SVM erkennt Block-Prompts etwas besser als die Logistische Regression, der Unterschied ist aber
klein. Beide nutzen dieselben TF-IDF-Zahlen und haben deshalb dieselben Grenzen.

## Logistische Regression vs. Lineare SVM

| | Logistische Regression | Lineare SVM |
|---|---|---|
| Ausgabe | Wahrscheinlichkeit (0–100 %) | Punktwert (Abstand zur Grenze) |
| Ziel beim Training | möglichst richtige Wahrscheinlichkeiten | Grenze mit breitestem Abstand |
| Schwelle | leicht zu erklären („block ab 20 %“) | weniger anschaulich („block ab −0,4“) |
| Geschwindigkeit | sehr schnell | sehr schnell |
| Unser Recall block | 0,955 | 0,956 |

## Stärken und Grenzen

| ✅ Stärken | ⚠️ Grenzen |
|---|---|
| Funktioniert sehr gut mit vielen Merkmalen (300.000 Stücke) | Keine echten Wahrscheinlichkeiten |
| Robust durch den breiten Abstand | Nur gerade Grenzen (linear) |
| Schnell | Wie TF-IDF: kein Verständnis von Kontext |
