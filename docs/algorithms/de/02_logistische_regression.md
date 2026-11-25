# Logistische Regression – ein Klassifikator, der Wahrscheinlichkeiten ausgibt

## Das Problem

Nach [TF-IDF](01_tfidf.md) ist jeder Prompt eine lange Liste von Zahlen. Jetzt brauchen wir ein
Modell, das sich diese Zahlen anschaut und entscheidet: **allow**, **warn** oder **block**.

## Die Idee in einem Satz

> Die Logistische Regression gibt jedem Textstück ein **Gewicht** (wie stark es auf eine Klasse
> hinweist), addiert die Gewichte eines Prompts und macht aus der Summe eine **Wahrscheinlichkeit**
> für jede Klasse.

## Schritt für Schritt

1. **Gewichte lernen.** Beim Training sieht das Modell viele Prompts mit der richtigen Antwort. Es
   lernt zum Beispiel:
   - Stücke wie `IBAN`, `Passw`, `PIN` → starkes Gewicht in Richtung **block**
   - Stücke wie `@`, `.de`, `Str.` → Gewicht in Richtung **warn**
   - Stücke wie `fasse`, `summar` → Gewicht in Richtung **allow**
2. **Addieren.** Bei einem neuen Prompt multipliziert das Modell jede TF-IDF-Zahl mit ihrem Gewicht
   und zählt alles zusammen. So entsteht ein Punktwert pro Klasse.
3. **In Wahrscheinlichkeiten umwandeln.** Eine mathematische Funktion (*Softmax*) macht aus den
   Punktwerten Prozentwerte, die zusammen 100 % ergeben.

Beispiel für „Meine IBAN ist DE89 3704 0044 0532 0130 00“:

| Klasse | Wahrscheinlichkeit |
|---|---|
| allow | 2 % |
| warn | 8 % |
| block | **90 %** |

Trotz des Namens „Regression“ ist es ein **Klassifikations**-Modell.

## Die zwei Stellschrauben, die wir getuned haben

**1. Regularisierung `C`** – *Wie sehr darf das Modell den Trainingsdaten vertrauen?*
- kleines `C` → einfaches, vorsichtiges Modell, übersieht evtl. Details
- großes `C` → detailliertes Modell, lernt die Trainingsdaten evtl. auswendig (*Overfitting*)
- Unser bester Wert: **C = 30**

**2. Block-Schwelle** – *Ab welcher Wahrscheinlichkeit blockieren wir?*
Normalerweise gewinnt die Klasse mit der höchsten Wahrscheinlichkeit. Zur Sicherheit blockieren wir
**schon ab 20 %** Block-Wahrscheinlichkeit (Schwelle 0,2). So übersehen wir weniger gefährliche
Prompts und nehmen dafür ein paar mehr Fehlalarme in Kauf.

Außerdem nutzen wir `class_weight="balanced"`: Seltene Klassen zählen beim Training stärker, damit
das Modell sie nicht ignoriert.

## Unsere Ergebnisse (Validierungs-Set)

| Version | Macro F1 | Recall block |
|---|---|---|
| Baseline (C = 1, keine Schwelle) | 0,870 | 0,918 |
| Getuned (C = 30, Schwelle 0,2) | 0,864 | **0,955** |

Mit dem Tuning erreicht das Modell unser Ziel, **mindestens 95 %** der Block-Prompts zu blockieren,
in etwa 3 ms pro Prompt.

## Stärken und Grenzen

| ✅ Stärken | ⚠️ Grenzen |
|---|---|
| Schnell zu trainieren und zu nutzen | Nur so gut wie die TF-IDF-Zahlen (kein Kontext) |
| Gibt Wahrscheinlichkeiten aus → Schwelle leicht einstellbar | Nur „linear“: Gewichte werden einfach addiert |
| Gut erklärbar: Wir können die wichtigsten Stücke pro Klasse zeigen | Kombinationen von Stücken werden nicht verstanden |
