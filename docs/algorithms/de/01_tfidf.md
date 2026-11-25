# TF-IDF – Text in Zahlen verwandeln

## Das Problem

Ein Computer kann keinen Text lesen. Ein Machine-Learning-Modell arbeitet nur mit **Zahlen**.
Bevor wir einen Prompt als *allow*, *warn* oder *block* einordnen können, müssen wir jeden Prompt in
eine Liste von Zahlen umwandeln. Dafür nutzen wir TF-IDF.

## Die Idee in einem Satz

> TF-IDF gibt Textstücken eine hohe Zahl, wenn sie **in diesem Prompt oft** vorkommen, aber **in
> allen anderen Prompts selten**. Denn genau diese Stücke sagen am meisten über den Prompt aus.

## Die zwei Teile

**TF = Term Frequency (Häufigkeit)** – *Wie oft kommt ein Stück in diesem Prompt vor?*
Steht „IBAN“ im Prompt, ist das ein Hinweis. Steht es dreimal drin, ist der Hinweis etwas stärker.

**IDF = Inverse Document Frequency (Seltenheit)** – *Wie selten ist das Stück in allen Prompts?*
Stücke wie „der“ oder „und“ stehen in fast jedem Prompt. Sie sagen nichts aus und bekommen ein
**niedriges** Gewicht. Ein Stück wie „IBAN“ oder „Passwort“ ist selten und bekommt ein **hohes**
Gewicht.

**TF × IDF** = die endgültige Zahl für jedes Stück in jedem Prompt.

## Kleines Beispiel

| Stück | Kommt im Prompt vor? | Kommt in vielen Prompts vor? | TF-IDF |
|---|---|---|---|
| „bitte“ | ja | ja, sehr oft | niedrig |
| „zusammenfassen“ | ja | recht oft | mittel |
| „IBAN“ | ja | selten | **hoch** |

Jeder Prompt wird so zu einer langen Liste (einem *Vektor*) mit einer Zahl pro Stück. Die meisten
Zahlen sind 0, weil ein einzelner Prompt nur wenige aller möglichen Stücke enthält.

## Was wir in PromptGuard machen

Wir schneiden den Text nicht in ganze Wörter, sondern in **kurze Buchstabenfolgen mit 2 bis 5
Zeichen** (Zeichen-N-Gramme), zum Beispiel `„IBAN“` → `IB`, `BA`, `AN`, `IBA`, `BAN`, `IBAN`.

Warum Buchstaben statt Wörter?

- **Schreibweise und Tippfehler:** „Passwort“, „passwort“ und „Passwrt“ teilen sich trotzdem viele
  Stücke.
- **Mehrere Sprachen:** funktioniert gleich für Deutsch, Englisch, Französisch und Italienisch.
- **Muster in Zahlen:** Eine IBAN wie `DE89 3704…` oder eine E-Mail mit `@` und `.de` erzeugt typische
  Stücke, auch wenn das Modell genau diese Nummer nie gesehen hat.

Unsere Einstellungen (scikit-learn `TfidfVectorizer`):

| Einstellung | Wert | Bedeutung in einfachen Worten |
|---|---|---|
| `analyzer` | `char_wb` | Buchstabenstücke, nur innerhalb von Wortgrenzen |
| `ngram_range` | `(2, 5)` | Stücke mit 2 bis 5 Zeichen |
| `min_df` | `5` | Stücke ignorieren, die in weniger als 5 Prompts vorkommen (Rauschen) |
| `max_features` | `300.000` | höchstens die 300.000 häufigsten Stücke behalten |
| `sublinear_tf` | `True` | 10 Wiederholungen zählen mehr als 1, aber nicht zehnmal so viel |

## Stärken und Grenzen

| ✅ Stärken | ⚠️ Grenzen |
|---|---|
| Sehr schnell (wenige Millisekunden) | Versteht keine Bedeutung und keinen Kontext |
| Leicht zu erklären | Wortreihenfolge geht größtenteils verloren („Sack voller Stücke“) |
| Kommt gut mit Tippfehlern und vielen Sprachen klar | Ein Name wie „Anna“ sieht aus wie jedes andere Wort |

Der letzte Punkt erklärt eines unserer Ergebnisse: TF-IDF erkennt nicht, ob „Anna“ eine Person im
Prompt ist oder nur ein normales Wort. Deshalb werden manche Prompts mit Namen noch durchgelassen.

**Wichtig:** TF-IDF trifft selbst keine Entscheidung. Es erzeugt nur die Zahlen. Die Entscheidung
trifft danach der Klassifikator: [Logistische Regression](02_logistische_regression.md) oder
[Lineare SVM](03_lineare_svm.md).
