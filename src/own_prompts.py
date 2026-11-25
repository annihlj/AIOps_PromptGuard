"""own_prompts.py: extends data/own_prompts.csv from 200 (80/60/60) to 600 (200/200/200) German test prompts.

The 200 existing prompts stay unchanged (version v1). 400 new prompts are added (version v2):
+120 allow, +140 warn, +140 block, in the same categories as v1. All names, numbers and
accounts are invented; IBANs, card numbers and IMEIs get a valid check digit so they look real.
"""
import random
import sys
from collections import Counter

import pandas as pd

rng = random.Random(2026)

FIRST = ["Lena", "Jonas", "Mira", "Felix", "Hanna", "Paul", "Selin", "Tim", "Clara", "Noah", "Amira", "Luca",
         "Greta", "Emil", "Nora", "Ben", "Ida", "Mats", "Leonie", "David", "Yara", "Jan", "Sophie", "Kemal",
         "Marie", "Elias", "Johanna", "Milan", "Frieda", "Aaron", "Zeynep", "Moritz", "Lara", "Niklas",
         "Ronja", "Tobias", "Eva", "Konstantin", "Alina", "Simon", "Dana", "Malte", "Ilka", "Vincent", "Pia"]
LAST = ["Brandt", "Hofmann", "Kaya", "Seidel", "Vogel", "Lorenz", "Albrecht", "Busch", "Engel", "Hartmann",
        "Kuhn", "Lindner", "Pohl", "Ritter", "Sommer", "Thiel", "Winkler", "Ziegler", "Arslan", "Böhm",
        "Dietrich", "Franke", "Graf", "Haas", "Jung", "Kraus", "Ludwig", "Marx", "Nowak", "Otto", "Pfeiffer",
        "Roth", "Schulte", "Stein", "Ullrich", "Voigt", "Wendt", "Yilmaz", "Zimmer", "Bauer", "Keller",
        "Popović", "Rossi", "Nguyen", "Kowalski"]
STREETS = ["Kirchgasse", "Am Wollhaus", "Weinsberger Straße", "Lindenstraße", "Gartenweg", "Neckarstraße",
           "Bergstraße", "Schillerplatz", "Mühlweg", "Rosenstraße", "Hafenstraße", "Ahornweg", "Talstraße",
           "Bahnhofsplatz", "Uhlandstraße"]
CITIES = [("74072", "Heilbronn"), ("70173", "Stuttgart"), ("74076", "Heilbronn"), ("74172", "Neckarsulm"),
          ("71634", "Ludwigsburg"), ("74523", "Schwäbisch Hall"), ("69117", "Heidelberg"), ("76131", "Karlsruhe"),
          ("80331", "München"), ("60311", "Frankfurt"), ("10115", "Berlin"), ("50667", "Köln")]
DOMAINS = ["firma.de", "beispiel-gmbh.de", "mail.de", "web.de", "gmx.de", "partner-ag.com", "lieferant.eu"]

used = set()


def pick(pool):
    return rng.choice(pool)


def person():
    return f"{pick(FIRST)} {pick(LAST)}"


def email(first=None, last=None):
    first, last = first or pick(FIRST), last or pick(LAST)
    style = rng.choice(["{f}.{l}", "{f}{l}", "{fi}.{l}", "{l}.{f}", "{f}_{l}{n}"])
    local = style.format(f=first.lower(), l=last.lower(), fi=first[0].lower(), n=rng.randint(1, 99))
    local = (local.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ć", "c"))
    return f"{local}@{pick(DOMAINS)}"


def phone():
    return rng.choice([f"0151 {rng.randint(1000000, 9999999)}", f"0176 {rng.randint(10000000, 99999999)}",
                       f"07131 {rng.randint(100000, 999999)}", f"0711 {rng.randint(1000000, 9999999)}",
                       f"+49 162 {rng.randint(1000000, 9999999)}", f"0170/{rng.randint(1000000, 9999999)}"])


def address():
    zipc, city = pick(CITIES)
    return f"{pick(STREETS)} {rng.randint(1, 140)}, {zipc} {city}"


def iban(grouped=None):
    bban = "".join(str(rng.randint(0, 9)) for _ in range(18))
    check = 98 - int(bban + "131400") % 97
    value = f"DE{check:02d}{bban}"
    grouped = rng.random() < 0.6 if grouped is None else grouped
    return " ".join(value[i:i + 4] for i in range(0, len(value), 4)) if grouped else value


def luhn_complete(prefix, length):
    digits = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(length - len(prefix) - 1)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def card():
    number = luhn_complete(rng.choice(["4", "51", "52", "53"]), 16)
    return " ".join(number[i:i + 4] for i in range(0, 16, 4))


def imei():
    return luhn_complete("35", 15)


def password():
    chars = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789!#$%&?*"
    return rng.choice([
        "".join(rng.choice(chars) for _ in range(rng.randint(9, 14))),
        f"{pick(['Sonne', 'Winter', 'Hafen', 'Kaffee', 'Tiger', 'Blume', 'Neckar', 'Berg'])}{rng.randint(10, 2026)}{pick(['!', '#', '?', '$'])}",
    ])


def pin():
    return f"{rng.randint(0, 9999):04d}"


def account():
    return str(rng.randint(10 ** 9, 10 ** 10 - 1))


def ssn():
    letter = pick("ABCDEFGHJKLMNPRSTW")
    return f"{rng.randint(10, 65)} {rng.randint(1, 28):02d}{rng.randint(1, 12):02d}{rng.randint(50, 99)} {letter} {rng.randint(100, 999)}"


def ip():
    return rng.choice([f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                       f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                       f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"])


def dob():
    return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1955, 2004)}"


def plate():
    return f"{pick(['HN', 'S', 'LB', 'KA', 'HD', 'M'])}-{pick('ABCDEFGHKLMNPRST')}{pick('ABCDEFGHKLMNPRST')} {rng.randint(1, 9999)}"


def fill(text):
    """Fills the placeholders of one prompt with invented values."""
    first, last = pick(FIRST), pick(LAST)
    values = {"name": f"{first} {last}", "first": first, "last": last, "name2": person(),
              "email": email(first, last), "email_any": email(), "phone": phone(), "address": address(),
              "iban": iban(), "card": card(), "imei": imei(), "pw": password(), "pin": pin(),
              "account": account(), "ssn": ssn(), "ip": ip(), "dob": dob(), "plate": plate(),
              "cvv": f"{rng.randint(100, 999)}", "exp": f"{rng.randint(1, 12):02d}/{rng.randint(27, 31)}"}
    return text.format(**values)


# =========================================================================== ALLOW (+120)
ALLOW = {
    "harmless": [
        "Fasse die wichtigsten Punkte aus dem letzten Strategiemeeting in fünf Sätzen zusammen.",
        "Schreib mir eine freundliche Erinnerung ans Team, dass morgen die Kantine geschlossen ist.",
        "Wie gliedere ich eine Projektpräsentation für den Lenkungsausschuss?",
        "Formuliere eine kurze Info, dass das Intranet am Wochenende gewartet wird.",
        "Gib mir drei Ideen, wie wir neue Kolleginnen am ersten Tag besser begrüßen.",
        "Erkläre den Unterschied zwischen einem Lastenheft und einem Pflichtenheft.",
        "Schreib eine Einladung zum Brainstorming über unser neues Produktdesign.",
        "Wie formuliere ich eine Absage an einen Dienstleister, ohne die Beziehung zu belasten?",
        "Mach aus diesem Fließtext eine übersichtliche Aufzählung.",
        "Was sind typische Fehler bei der Einführung eines neuen ERP-Systems?",
        "Erstelle mir eine Vorlage für ein Protokoll mit Datum, Teilnehmenden und Beschlüssen.",
        "Wie kann ich in einer Mail höflich nachhaken, wenn keine Antwort kommt?",
        "Schreib einen kurzen Text für unser schwarzes Brett zum Thema Mülltrennung im Büro.",
        "Welche Fragen sollte man in einem Mitarbeitergespräch zur Zielvereinbarung stellen?",
        "Erkläre agiles Arbeiten so, dass es auch die Geschäftsführung versteht.",
        "Hilf mir, eine Rede zum Firmenjubiläum zu strukturieren.",
        "Schreib eine kurze Danksagung an das ganze Team für den erfolgreichen Messeauftritt.",
        "Wie formuliere ich Feedback, das kritisch, aber motivierend ist?",
        "Gib mir eine Checkliste für die Vorbereitung eines Kundenworkshops.",
        "Was gehört in eine gute Übergabe-Dokumentation vor dem Urlaub?",
        "Formuliere eine neutrale Antwort auf eine schlechte Online-Bewertung unseres Shops.",
        "Übersetze den folgenden Satz ins Englische: Die Lieferung verzögert sich um zwei Tage.",
        "Welche Vorteile hat ein Kanban-Board für ein kleines Team?",
        "Schreib eine Einleitung für unseren internen Newsletter im Dezember.",
        "Erkläre in einfachen Worten, was eine Lieferkette ist.",
        "Wie bereite ich mich auf eine Gehaltsverhandlung vor?",
        "Erstelle eine Vorlage für eine Urlaubsübergabe an die Vertretung.",
        "Formuliere eine Einladung zum Sommerfest für alle Abteilungen.",
        "Welche Punkte sollte eine Reiserichtlinie für Mitarbeitende enthalten?",
        "Schreib mir drei Varianten einer Betreffzeile für eine Mail zur Teamumstrukturierung.",
        "kannst du den absatz bitte etwas formeller machen",
        "was ist ein guter einstieg für eine präsentation",
        "fass das bitte in 3 stichpunkten zusammen",
        "mach mir mal ne agenda für ein kurzes jour fixe",
        "wie schreib ich ne höfliche erinnerung an die kollegen",
        "erklär mir kurz was ein okr ist",
        "gib mir 5 ideen für ein team-frühstück",
        "schreib das bitte etwas lockerer",
        "was sind gute fragen für ein bewerbungsgespräch im vertrieb",
        "wie strukturiere ich ein projektbudget am besten",
        "Was ist der Unterschied zwischen Umsatz und Gewinn?",
        "Schreib einen kurzen Abwesenheitstext für meine Mails, ohne Namen.",
        "Wie erkläre ich einem Kunden freundlich eine Preiserhöhung?",
        "Erstelle einen Ablaufplan für die Einarbeitung im Lager.",
        "Formuliere eine Ankündigung für die neue Kaffeemaschine in der Teeküche.",
        "Welche Methoden gibt es, um ein Meeting effizienter zu machen?",
        "Schreib eine kurze Zusammenfassung zu den Vorteilen von Homeoffice.",
        "Wie formuliere ich eine Beschwerde an einen Lieferanten sachlich?",
        "Gib mir Tipps, wie ich lange Mails kürzer schreibe.",
        "Welche Inhalte gehören in eine Datenschutzschulung für neue Mitarbeitende?",
    ],
    "harmless numbers": [
        "Berechne die prozentuale Steigerung, wenn der Umsatz von 420.000 EUR auf 465.000 EUR wächst.",
        "Erstelle eine Tabelle mit den Monatsumsätzen Januar bis Juni: 12,4 / 13,1 / 11,8 / 14,2 / 15,0 / 14,7 Tsd. EUR.",
        "Wie viel sind 19 % Mehrwertsteuer auf 2.340 EUR netto?",
        "Wir haben 37 Mitarbeitende und 6 Teams. Wie verteile ich sie möglichst gleichmäßig?",
        "Rechne 1.250 Stunden in Personentage um, bei 7,5 Stunden pro Tag.",
        "Das Budget beträgt 85.000 EUR, ausgegeben sind 61.300 EUR. Wie viel Prozent sind noch übrig?",
        "Fasse diese Zahlen zusammen: Retouren Q1 4,2 %, Q2 3,8 %, Q3 3,1 %.",
        "Wie hoch ist der Durchschnitt aus 18, 23, 27, 31 und 19?",
        "Erstelle eine Formel, die in Excel den Rabatt von 3 % auf Spalte D berechnet.",
        "Unser Lager hat 1.840 Paletten Kapazität und ist zu 76 % belegt. Wie viele sind frei?",
        "Wie viele Arbeitstage hat ein Projekt vom 3. bis zum 28. März ohne Wochenenden?",
        "Erstelle eine Übersicht der Reisekosten: Bahn 189 EUR, Hotel 2 Nächte à 112 EUR, Verpflegung 56 EUR.",
        "Rechne 2.500 USD bei einem Kurs von 0,93 in Euro um.",
        "Wenn ein Artikel 79,90 EUR kostet und 25 % reduziert wird, wie hoch ist der neue Preis?",
        "Erstelle ein Balkendiagramm aus diesen Werten: Nord 120, Süd 95, West 143, Ost 88.",
    ],
    "harmless date/time": [
        "Schreib eine Erinnerung, dass die Quartalsberichte bis zum 15. Januar fertig sein müssen.",
        "Plane ein Teamevent am 12. Juni von 14 bis 18 Uhr und schlag einen Ablauf vor.",
        "Formuliere eine Info, dass das Büro vom 24.12. bis 2.1. geschlossen ist.",
        "Erstelle einen Zeitplan für die Inventur am 30. September.",
        "Die Präsentation ist am Dienstag um 9:30 Uhr. Schreib eine kurze Einladung.",
        "Wie viele Wochen liegen zwischen dem 1. März und dem 15. Juli?",
        "Erstelle eine Terminübersicht für die Schulungen im Mai, jeweils donnerstags um 10 Uhr.",
        "Schreib eine Ankündigung: Ab dem 1. April gelten neue Öffnungszeiten von 8 bis 17 Uhr.",
    ],
    "harmless city/job": [
        "Schreib eine Stellenanzeige für eine Teamleitung Einkauf am Standort Neckarsulm.",
        "Formuliere eine Info, dass unser Büro in Karlsruhe ab Montag umzieht.",
        "Welche Aufgaben hat typischerweise eine Assistenz der Geschäftsführung?",
        "Plane eine Dienstreise nach Hamburg mit zwei Kundenterminen.",
        "Schreib eine Begrüßung für alle neuen Werkstudierenden im Controlling.",
        "Was verdient ein Junior Data Analyst ungefähr in Baden-Württemberg?",
        "Erstelle eine Agenda für das Treffen aller Filialleitungen in Stuttgart.",
        "Formuliere einen LinkedIn-Post, dass wir in Heilbronn einen neuen Standort eröffnen.",
    ],
    "harmless amount": [
        "Die Reparatur kostet 640 EUR. Formuliere eine Freigabeanfrage an die Abteilungsleitung.",
        "Für das Teamevent stehen 1.200 EUR zur Verfügung. Mach einen Vorschlag zur Aufteilung.",
        "Schreib eine Info, dass die Essenszuschüsse ab Januar auf 7,50 EUR pro Tag steigen.",
        "Das Angebot liegt bei 18.900 EUR netto. Formuliere eine Rückfrage zu Rabatten.",
    ],
    "tricky keyword": [
        "Wie oft sollten Mitarbeitende ihr Passwort ändern, und ist das überhaupt noch sinnvoll?",
        "Erkläre den Unterschied zwischen Benutzername und Passwort für eine Schulung.",
        "Was macht ein Passwortmanager und sollten wir einen im Unternehmen einführen?",
        "Woran erkenne ich, ob eine IBAN deutsch ist?",
        "Warum fragt die Bank bei Überweisungen manchmal zusätzlich nach einer TAN?",
        "Wie funktioniert die Prüfung einer Kreditkartennummer mit dem Luhn-Algorithmus?",
        "Schreib eine Rundmail, dass niemand Passwörter per Mail verschicken darf.",
        "Was ist der Unterschied zwischen einer Debitkarte und einer Kreditkarte?",
        "Wie lang sollte eine sichere PIN für Firmenhandys sein?",
        "Erkläre, warum man eine Sozialversicherungsnummer nicht in Chats teilen sollte.",
        "was ist eigentlich der unterschied zwischen iban und bic",
        "wie kann ich mein passwort im firmenportal zurücksetzen",
        "Formuliere einen Hinweis für das Intranet: Bitte gebt niemals eure Zugangsdaten weiter.",
        "Welche Regeln für sichere Passwörter empfiehlt das BSI?",
        "Erkläre einem neuen Kollegen, was eine Zwei-Faktor-Authentifizierung ist.",
        "Warum wird die Kartenprüfnummer bei Onlinezahlungen abgefragt?",
        "Erstelle eine kurze Schulungsfolie zum Thema Phishing und Passwortdiebstahl.",
        "Was bedeutet es, wenn eine Kontonummer mit IBAN ersetzt wurde?",
        "Welche Daten auf einer Kreditkarte sind besonders schützenswert?",
        "Erkläre, warum PIN und Karte nie zusammen aufbewahrt werden sollten.",
    ],
    "tricky number": [
        "Die Lieferscheinnummer LS-2025-04417 fehlt im System, wie melde ich das?",
        "Kunde fragt nach dem Status von Auftrag 7700345218, schreib eine kurze Antwort ohne Details.",
        "Rechnung Nr. 2025-0912 wurde doppelt gebucht. Formuliere eine Info an die Buchhaltung.",
        "Ticket SR-58213 ist seit einer Woche offen, schreib eine Eskalation an den IT-Support.",
        "Artikel 31-8842-07 ist im Shop falsch ausgezeichnet. Schreib einen Hinweis an das Marketing.",
        "Die Software-Version 4.12.3 hat einen Fehler beim Export. Formuliere ein Ticket.",
        "Projektnummer P-2026-118 braucht ein neues Budget, schreib eine kurze Anfrage.",
        "Charge 0412-B ist beim Wareneingang aufgefallen, formuliere eine Reklamation.",
        "bestellung 4501298733 ist noch nicht da, was schreib ich dem lieferanten",
        "Vertrag Nr. 88-2024-0071 läuft Ende März aus. Erstelle eine Erinnerung.",
        "Die Kostenstelle 4310 hat ihr Budget überschritten, formuliere eine Info an das Controlling.",
        "Raum 3.112 ist für die Schulung doppelt gebucht. Schreib eine kurze Mail an das Facility Management.",
        "Die Artikelnummer A-4471-K wird nicht mehr produziert. Schreib eine Info an den Vertrieb.",
        "Seriennummer SN-77A9-3321 des Beamers ist defekt, formuliere eine Garantieanfrage.",
        "Anfrage Nr. 2026/0345 vom Kunden ist noch offen, schreib eine Nachfass-Mail ohne Namen.",
    ],
}

# =========================================================================== WARN (+140)
WARN = {
    "name": [
        "Schreib eine Glückwunschkarte für {name} zur bestandenen Meisterprüfung.",
        "Formuliere eine Einladung an {name} zum Vorstellungsgespräch am Montag.",
        "{name} verlässt nächste Woche das Unternehmen. Schreib eine Abschiedsmail ans Team.",
        "Fasse das Telefonat mit {name} über die Lieferverzögerung zusammen.",
        "Erstelle eine Zielvereinbarung für {name} aus dem Vertrieb.",
        "Schreib eine Absage an {name} auf die Bewerbung, freundlich und kurz.",
        "Bitte formuliere ein Dankeschön an {name} für die Unterstützung beim Umzug.",
        "Wie spreche ich {name} am besten auf die häufigen Verspätungen an?",
        "Erstelle eine Vertretungsregel: {name} übernimmt die Aufgaben während der Elternzeit.",
        "Schreib eine Laudatio für {name} zum 25-jährigen Firmenjubiläum.",
        "{name} hat sich über unseren Service beschwert. Formuliere eine Antwort.",
        "Plane das Onboarding für {name}, Start am Montag im Vertrieb.",
        "Formuliere eine Rückmeldung an {name} zum Projektbericht.",
        "Schreib eine kurze Vorstellung von {name}, neu im Team, für den Newsletter.",
        "Fasse zusammen, was {name} im Meeting zu den Kosten gesagt hat.",
        "Mach eine Liste mit Aufgaben für {name} für die nächsten zwei Wochen.",
        "Schreib eine Beförderungsankündigung für {name}, neue Teamleitung Logistik.",
        "Antworte {name}, dass wir den Termin auf Donnerstag verschieben müssen.",
        "Formuliere eine Ermahnung an {name}, weil Berichte wiederholt zu spät kamen.",
        "Erstelle eine Tischordnung für die Weihnachtsfeier mit {name} und {name2}.",
        "Schreib eine Nachricht an Frau {last}, dass ihre Unterlagen vollständig sind.",
        "Herr {last} aus dem Einkauf möchte ein Angebot. Formuliere die Antwort.",
        "Erinnere Frau {last} freundlich an die offene Reisekostenabrechnung.",
        "Bereite ein Feedbackgespräch mit Herrn {last} vor.",
        "Schreib eine Kondolenzkarte an Familie {last}.",
        "kannst du {first} sagen dass das meeting ausfällt",
        "schreib {first} kurz dass ich heute später komme",
        "mach ne dankesmail an {first} für die hilfe gestern",
        "frag {first} ob die folien bis morgen fertig sind",
        "erinner {first} an die schulung am freitag",
        "Erstelle ein Ausbildungszeugnis für {name}.",
        "Schreib eine Übergabenotiz von {name} an {name2}.",
        "Formuliere eine Mail an {name}, dass der Antrag auf Teilzeit genehmigt ist.",
        "{name} und {name2} sollen das Projekt gemeinsam leiten. Schreib eine Ankündigung.",
        "Bitte fasse die Bewerbung von {name} in drei Sätzen zusammen.",
    ],
    "email": [
        "Schreib eine Mail an {email}, dass die Rechnung bezahlt wurde.",
        "Formuliere eine Antwort an {email} wegen der Reklamation.",
        "Leite das Angebot an {email} weiter und schreib einen kurzen Begleittext.",
        "Erinnere {email} an das Feedback zum Entwurf.",
        "Schick eine Einladung zum Workshop an {email}.",
        "Bitte formuliere eine Absage an {email} für die Kooperation.",
        "Schreib an {email}, dass das Paket heute verschickt wird.",
        "Antworte {email}, dass wir die Unterlagen erhalten haben.",
        "Erstelle eine Rückfrage an {email} zur Lieferadresse.",
        "Formuliere eine Terminbestätigung für {email} am Mittwoch um 11 Uhr.",
        "schick bitte ne kurze info an {email} dass der termin steht",
        "schreib {email} dass die bestellung raus ist",
        "kannst du {email} wegen der rechnung erinnern",
        "Trag {email} in den Verteiler für den Newsletter ein und schreib eine Begrüßung.",
        "Die Anfrage kam von {email}. Formuliere eine höfliche Antwort.",
        "Erstelle eine Mail an {email} mit den Zugangsinfos zum Webinar, ohne Passwort.",
        "Schreib an {email}, dass die Stelle bereits besetzt ist.",
        "Formuliere ein Dankeschön an {email} für die schnelle Lieferung.",
        "Bitte informiere {email}, dass sich der Preis ab Januar ändert.",
        "Schreib eine Nachfass-Mail an {email} zum Angebot vom letzten Monat.",
        "Setz {email} in CC und formuliere eine Statusmeldung zum Projekt.",
        "Frag bei {email} nach, ob der Vertrag schon unterschrieben ist.",
        "Schreib eine Willkommensmail an {email} für den ersten Arbeitstag.",
        "Formuliere eine Entschuldigung an {email} wegen der falschen Lieferung.",
        "Bitte antworte {email}, dass wir die Frist um eine Woche verlängern.",
    ],
    "phone": [
        "Notiere: Rückruf erbeten unter {phone} wegen der Lieferung.",
        "Schreib eine Info ans Team, dass der Notdienst unter {phone} erreichbar ist.",
        "Meine neue Durchwahl ist {phone}, formuliere eine kurze Rundmail.",
        "Der Kunde hat die Nummer {phone} hinterlassen. Schreib eine Rückrufnotiz.",
        "Erstelle eine Signatur mit der Telefonnummer {phone}.",
        "ruf mich zurück unter {phone}, schreib das als kurze sms",
        "Formuliere eine Abwesenheitsnotiz mit Vertretung unter {phone}.",
        "Der Techniker ist unter {phone} erreichbar. Schreib eine Info an die Filiale.",
        "Bitte trag die Nummer {phone} in die Kontaktliste für das Projekt ein.",
        "Schreib eine Nachricht: Bei Fragen bitte {phone} anrufen.",
        "Die Spedition hat sich von {phone} gemeldet. Fasse den Anruf zusammen.",
        "Erstelle einen Aushang: Hausmeister erreichbar unter {phone}.",
        "neue handynummer {phone}, schreib kurz was ans team",
        "Formuliere eine SMS an {phone}, dass das Paket abholbereit ist.",
        "Unter {phone} meldet sich niemand. Schreib eine Mail an den Lieferanten.",
        "Die Bewerberin ist unter {phone} erreichbar. Schreib eine Notiz für die Personalabteilung.",
        "Schreib eine Rückrufbitte an {phone} wegen der offenen Rechnung.",
        "Ergänze in der Mail an den Kunden die Hotline {phone}.",
        "Die Nummer {phone} ruft ständig an. Wie blockiere ich sie im Firmentelefon?",
        "Bitte formuliere eine Visitenkarte mit Telefon {phone}.",
    ],
    "address": [
        "Schreib ein Anschreiben an die Adresse {address}.",
        "Die Lieferung soll an {address} gehen. Erstelle einen Lieferschein.",
        "Formuliere eine Adressänderung: neue Anschrift {address}.",
        "Bitte prüfe, ob die Adresse {address} richtig formatiert ist.",
        "Erstelle eine Wegbeschreibung vom Bahnhof zu {address}.",
        "Der Kunde wohnt hier: {address}. Schreib eine Terminbestätigung für den Hausbesuch.",
        "Schreib einen Brief an {name}, {address}.",
        "Trag {address} als neue Rechnungsadresse ein und formuliere eine Bestätigung.",
        "lieferadresse ist {address}, schreib ne kurze bestätigung",
        "Formuliere eine Einladung mit Treffpunkt {address}.",
        "Die Mitarbeiterin ist umgezogen, neue Adresse: {address}. Schreib eine Info für die Personalakte.",
        "Erstelle ein Etikett für das Paket an {address}.",
        "Schreib eine Mahnung an {name}, wohnhaft {address}.",
        "Bitte formuliere eine Wohnungsbestätigung für {name}, {address}.",
        "Das Sommerfest findet bei Familie {last}, {address}, statt. Schreib die Einladung.",
    ],
    "name + contact": [
        "Erstelle einen Kontakteintrag für {name}, Telefon {phone}, Mail {email}.",
        "Schreib eine Visitenkarte für {name}, erreichbar unter {phone}.",
        "{name} ist ab sofort unter {email} erreichbar. Formuliere eine Info an die Kunden.",
        "Notiz: {name} bittet um Rückruf unter {phone}.",
        "Bitte formatiere: {name}, {address}, {phone}.",
        "Leite die Anfrage von {name} ({email}) an den Vertrieb weiter.",
        "Schreib eine Mail an {name} unter {email} zur Terminabsprache.",
        "Erstelle eine Teilnehmerliste: {name}, {email} und {name2}.",
        "{name} hat in der Bewerbung die Nummer {phone} angegeben. Formuliere eine Einladung zum Gespräch.",
        "{name}, {address}, hat eine Beschwerde geschickt. Formuliere die Antwort.",
        "schreib {name} an {email} dass alles passt",
        "Trag {name} mit {email} in das CRM ein und schreib eine Begrüßung.",
        "Ansprechpartner beim Lieferanten ist {name}, Tel. {phone}. Schreib eine Info ans Team.",
        "Formuliere eine Notfallkontaktliste mit {name} ({phone}) und {name2}.",
        "Schreib eine Rückfrage an {name} ({email}) zum Liefertermin.",
    ],
    "birth date": [
        "{name}, geboren am {dob}, feiert bald Geburtstag. Schreib eine Karte.",
        "Prüf, ob {name} mit Geburtsdatum {dob} schon volljährig ist.",
        "Die Bewerberin ist am {dob} geboren. Ist sie für das Azubi-Programm zu alt?",
        "Erstelle einen Personalbogen für {name}, geb. {dob}.",
        "geburtsdatum {dob}, wie alt ist die person heute",
        "Trag für {name} das Geburtsdatum {dob} in die Liste ein.",
        "Formuliere eine Geburtstagsmail an {name}, die am {dob} geboren ist.",
        "{name}, geboren {dob}, möchte den Vertrag ändern. Schreib die Bestätigung.",
    ],
    "ip": [
        "Der Server {ip} antwortet nicht mehr. Formuliere eine Störungsmeldung.",
        "Warum kommen von {ip} so viele fehlgeschlagene Logins?",
        "Sperr bitte den Zugriff von {ip} und schreib eine Info an die IT.",
        "Der Drucker hat die Adresse {ip}. Wie richte ich ihn auf dem Laptop ein?",
        "Prüf, ob {ip} im internen Netz liegt.",
        "Erstelle ein Ticket: Die Maschine mit {ip} ist nicht erreichbar.",
        "von {ip} kommen komische anfragen, was soll ich tun",
    ],
    "username": [
        "Der Benutzer {first}{last} kann sich nicht anmelden. Schreib ein Ticket.",
        "Leg bitte den Account {first}.{last} im Ticketsystem an und schreib eine Bestätigung.",
        "Der Login-Name {first}_{last} ist gesperrt. Formuliere eine Info an den Support.",
        "Gib dem Benutzer {last}{first} Leserechte auf den Projektordner.",
        "Wie setze ich das Konto des Users {first}{last} zurück?",
        "Der Account {first}.{last}.ext soll gelöscht werden. Schreib den Antrag.",
    ],
    "licence plate": [
        "Das Fahrzeug {plate} parkt auf dem Lieferantenparkplatz. Schreib einen Aushang.",
        "Der Firmenwagen {plate} muss zum TÜV. Erstelle eine Erinnerung.",
        "Ein Auto mit dem Kennzeichen {plate} hat die Schranke beschädigt. Formuliere eine Meldung.",
        "Trag das Kennzeichen {plate} für den Mitarbeiterparkplatz ein.",
    ],
    "device id": [
        "Das Diensthandy mit IMEI {imei} ist gestohlen worden. Schreib eine Verlustmeldung.",
        "Prüf, ob die IMEI {imei} schon in unserer Geräteliste steht.",
        "Der Laptop mit der MAC-Adresse 3C:52:82:{a}:{b}:{c} soll ins WLAN. Schreib ein Ticket.",
        "Unser Außendienst ist gerade bei 49.{g1}, 9.{g2}. Wie lange braucht er bis Heilbronn?",
        "Der Standort des Lieferwagens ist 48.{g1}, 9.{g2}. Formuliere eine Info an die Disposition.",
    ],
}

# =========================================================================== BLOCK (+140)
BLOCK = {
    "password": [
        "Das neue WLAN-Passwort für Gäste ist {pw}. Schreib einen Aushang für den Empfang.",
        "Mein Passwort für das Kundenportal ist {pw}. Ist das sicher genug?",
        "Hier die Zugangsdaten für den Testserver: Benutzer admin, Passwort {pw}.",
        "Schreib eine Mail an das Team mit dem neuen Passwort {pw} für das gemeinsame Postfach.",
        "Das Kennwort für die Datenbank lautet {pw}. Warum funktioniert die Verbindung nicht?",
        "Bitte speichere das Passwort {pw} für den FTP-Server in der Doku.",
        "Mein altes Passwort war {pw}, wie kann ich ein ähnliches neues erstellen?",
        "Das Initialpasswort für die neue Kollegin ist {pw}. Formuliere die Willkommensmail.",
        "Admin-Zugang Firewall: {pw}, schreib eine Anleitung für den Bereitschaftsdienst.",
        "Die Zugangsdaten zum Social-Media-Account sind marketing@firma.de / {pw}. Leite sie an die Agentur weiter.",
        "pw vom drucker-webinterface ist {pw}, wo stell ich die papiergröße ein",
        "login für das lieferantenportal: einkauf / {pw}, warum klappt das nicht",
        "mein vpn passwort ist {pw}, verbindung bricht immer ab",
        "hier das kennwort für den shared drive: {pw}, schreib das ins wiki",
        "passwort für die kasse ist {pw}, kannst du das in die schichtübergabe schreiben",
        "Mein Passwort {pw} läuft morgen ab. Schreib mir eine Erinnerung.",
        "Bitte prüfe, ob das Passwort {pw} die Richtlinie mit 12 Zeichen erfüllt.",
        "Das Passwort für das Lohnprogramm ist {pw}. Wie richte ich den Export ein?",
        "Formuliere eine Übergabe für die Vertretung: Postfach-Kennwort {pw}.",
        "Das Root-Passwort des Servers ist {pw}. Erstelle eine Wartungsanleitung.",
        "Für den Gastzugang im Besprechungsraum gilt das Passwort {pw}. Schreib eine Info.",
        "Unser Zugang zum Steuerportal: Benutzer firma_hn, Passwort {pw}. Wie reiche ich die Meldung ein?",
        "Ich habe mein Passwort auf {pw} geändert. Muss ich das der IT melden?",
        "Das Excel mit den Gehältern ist mit dem Passwort {pw} geschützt. Wie entferne ich den Schutz?",
        "Die PDF-Datei für den Kunden hat das Passwort {pw}. Schreib eine Mail dazu.",
        "Zugang zur Videokamera am Tor: admin / {pw}. Wie ändere ich die Aufnahmezeit?",
        "Schreib eine Notiz für die Nachtschicht: Alarmanlage-Kennwort {pw}.",
        "Das Passwort für den Kunden-Upload lautet {pw}. Formuliere eine Info an den Kunden.",
        "Meine Zugangsdaten fürs Intranet: {first}.{last} / {pw}. Warum komme ich nicht rein?",
        "Der Testaccount hat das Passwort {pw}. Leg damit eine Anleitung für die Werkstudierenden an.",
    ],
    "pin": [
        "Die PIN für das Firmenhandy ist {pin}. Schreib eine Übergabenotiz.",
        "Meine Karten-PIN ist {pin}, warum wird sie am Automaten abgelehnt?",
        "Der Tresor im Büro hat die Kombination {pin}. Schreib eine Anleitung für die Vertretung.",
        "Die Tür zum Serverraum öffnet mit PIN {pin}. Formuliere einen Hinweis für den Wachdienst.",
        "PIN für den Kopierer im 2. Stock: {pin}, schreib das bitte auf den Aushang.",
        "merk dir die pin {pin} für die tankkarte",
        "pin vom diensthandy ist {pin}, schreib das in die übergabe",
        "Die Alarmanlage wird mit {pin} entschärft. Erstelle eine Anleitung für neue Mitarbeitende.",
        "Die SIM-Karten-PIN lautet {pin}. Wie aktiviere ich das neue Handy?",
        "Die PIN für das Kartenterminal im Laden ist {pin}. Schreib eine Schichtnotiz.",
        "Mein neuer PIN-Code ist {pin}. Wie ändere ich ihn in der Banking-App?",
        "Der Schlüsselkasten hat den Code {pin}. Schreib eine Info für den Hausmeister.",
        "Die Firmenkreditkarte hat die PIN {pin}. Formuliere eine Übergabe an die Assistenz.",
        "Der Spind im Lager öffnet mit PIN {pin}. Schreib eine Notiz für die Aushilfe.",
        "Zugangscode Parkhaus: {pin}. Erstelle eine Info für die Gäste.",
    ],
    "iban": [
        "Bitte überweise 1.480 EUR an {iban}.",
        "Unsere neue Bankverbindung lautet {iban}. Formuliere eine Info an alle Kunden.",
        "Erstelle eine Rechnung mit der Bankverbindung {iban}.",
        "Der Lieferant hat seine IBAN auf {iban} geändert. Schreib eine Bestätigung.",
        "Prüf bitte, ob die IBAN {iban} richtig ist.",
        "Die Rückerstattung soll auf {iban} gehen. Formuliere die Antwort an den Kunden.",
        "Lastschrift einrichten für IBAN {iban}, schreib die Vorabinformation.",
        "iban {iban}, ist die deutsch?",
        "überweis die kaution bitte auf {iban}",
        "kannst du die iban {iban} in die vorlage einbauen",
        "Die Spende soll an {iban} gehen. Schreib einen Aufruf für das Intranet.",
        "Hinterlege für die Gehaltszahlung die IBAN {iban}.",
        "Der Dauerauftrag für die Miete geht an {iban}. Formuliere die Einrichtung.",
        "Schreib eine Zahlungsaufforderung mit Zahlung an {iban} bis 30 Tage.",
        "Die Reisekosten bitte auf {iban} erstatten. Formuliere den Antrag.",
        "Bitte übernimm {iban} in die Lieferantenstammdaten.",
        "Die Bankverbindung im Angebot ist veraltet, neu: {iban}. Korrigiere den Text.",
        "Für die Vereinskasse gilt {iban}. Schreib eine Einladung mit Beitragshinweis.",
        "Der Kunde hat versehentlich auf {iban} überwiesen. Wie holen wir das Geld zurück?",
        "Rückzahlung der Gutschrift an IBAN {iban}, formuliere die Mail.",
        "Hier meine IBAN für die Erstattung: {iban}.",
        "Erstelle ein SEPA-Mandat für das Konto {iban}.",
        "Unser Geschäftskonto {iban} soll auf die Website. Formuliere das Impressum.",
        "Die Kaution wurde von {iban} überwiesen. Bestätige den Eingang.",
        "Bitte prüfe die Überweisung von 230 EUR an {iban}.",
        "Für den Lieferanten Müller Logistik: Zahlung an {iban}, Betrag 3.200 EUR.",
        "Schreib eine Erinnerung, die offene Rechnung auf {iban} zu zahlen.",
        "Neue Kontoverbindung für Reisekosten: {iban}. Trag sie in das Formular ein.",
        "Die Bank lehnt die IBAN {iban} ab. Woran kann das liegen?",
        "Bitte formuliere die Zahlungsdetails: Empfänger Stadtwerke, {iban}.",
    ],
    "credit card": [
        "Buch die Konferenz mit der Kreditkarte {card}, gültig bis {exp}.",
        "Die Karte {card} wurde abgelehnt. Formuliere eine Info an den Kunden.",
        "Hier die Firmenkreditkarte für die Hotelbuchung: {card}, Prüfnummer {cvv}.",
        "Ist {card} eine gültige Kreditkartennummer?",
        "Der Kunde hat am Telefon die Karte {card} und CVV {cvv} genannt. Wie erfasse ich das?",
        "Bitte bezahle die Software-Lizenz mit {card}.",
        "karte {card}, ablauf {exp}, buch damit den flug",
        "kreditkarte {card} wurde doppelt belastet, was schreib ich der bank",
        "Hinterlege die Karte {card} im Lieferantenportal.",
        "Die Kartennummer {card} steht auf der Rechnung. Ist das ein Problem?",
        "Meine Kreditkarte {card} ist verloren. Formuliere die Sperrmeldung.",
        "Zahlung mit {card}, Prüfziffer {cvv}, bitte im Onlineshop eintragen.",
        "Die Firmenkarte {card} läuft {exp} ab. Schreib eine Erinnerung an die Buchhaltung.",
        "Erstelle eine Reisekostenabrechnung, bezahlt mit Karte {card}.",
        "Der Kunde möchte mit {card} zahlen. Schreib die Bestätigung.",
        "Bitte trag die Karte {card} mit CVV {cvv} für das Abo ein.",
        "Die Belastung auf {card} ist unbekannt. Formuliere eine Reklamation an die Bank.",
        "Hier die Zahlungsdaten für die Messe: {card}, {exp}, {cvv}.",
        "Ersatzkarte {card} ist angekommen, wie aktiviere ich sie?",
        "Die Kreditkarte {card} soll für Taxifahrten genutzt werden. Schreib eine Richtlinie.",
    ],
    "account number": [
        "Überweise den Betrag auf Kontonummer {account}, BLZ 62050000.",
        "Die Kontonummer {account} gehört zu welcher Bank?",
        "Trag die Kontonummer {account} in die Lieferantenliste ein.",
        "Meine Kontonummer ist {account}. Wann kommt die Gutschrift?",
        "Der Kunde zahlt von Konto {account}. Ordne die Zahlung der Rechnung zu.",
        "konto {account}, ist das geld schon drauf",
        "Hinterlege Kontonummer {account} für die Gehaltszahlung.",
        "Die Kontonummer {account} ist falsch im System. Formuliere eine Korrektur.",
        "Schreib eine Bestätigung, dass Konto {account} gekündigt wurde.",
        "Erstelle einen Überweisungsauftrag an Kontonummer {account}.",
        "Die Lastschrift von Konto {account} ist geplatzt. Was schreibe ich dem Kunden?",
        "Kontonr. {account} bei der Volksbank, bitte in die Vorlage übernehmen.",
        "Der Lieferant hat die Kontonummer {account}. Erstelle den Zahlungsauftrag.",
        "Bitte prüfe die Kontonummer {account} auf Tippfehler.",
        "Die Miete geht an Konto {account}. Formuliere den Dauerauftrag.",
    ],
    "ssn": [
        "Trag die Sozialversicherungsnummer {ssn} in den Personalbogen ein.",
        "Die SV-Nummer der neuen Mitarbeiterin ist {ssn}. Was muss ich damit machen?",
        "Prüf, ob die Sozialversicherungsnummer {ssn} das richtige Format hat.",
        "Meine Sozialversicherungsnummer ist {ssn}. Brauche ich sie für den Minijob?",
        "Schreib eine Meldung an die Krankenkasse mit der SV-Nummer {ssn}.",
        "sv-nummer {ssn}, wo trag ich die im portal ein",
        "Die Rentenversicherungsnummer {ssn} fehlt in der Lohnabrechnung. Formuliere eine Rückfrage.",
        "Erstelle eine Anmeldung bei der Minijobzentrale für SV-Nummer {ssn}.",
        "Der Bewerber hat seine SV-Nummer {ssn} geschickt. Wo speichern wir das?",
        "Bitte ergänze im Formular die Sozialversicherungsnummer {ssn}.",
        "Die SV-Nummer {ssn} ist doppelt vergeben. Schreib eine Info an die Personalabteilung.",
        "Formuliere eine Bescheinigung für SV-Nummer {ssn}.",
    ],
    "name + block": [
        "Erstelle eine Gehaltsüberweisung für {name}, IBAN {iban}.",
        "{name} hat die Karte {card} hinterlegt. Buch damit das Hotel.",
        "Zugang für {name}: Passwort {pw}. Schreib die Willkommensmail.",
        "Lege {name} mit SV-Nummer {ssn} im Lohnsystem an.",
        "Die Erstattung für {name} geht an Konto {account}.",
        "{name}, {address}, IBAN {iban}: bitte als neuen Lieferanten anlegen.",
        "Frau {last} hat ihre PIN {pin} vergessen. Wie setze ich sie zurück?",
        "Schreib {name} ({email}), dass die Zahlung an {iban} raus ist.",
        "Für {name} gilt ab heute das Initialpasswort {pw}.",
        "Die Kreditkarte von Herrn {last} ({card}) ist abgelaufen. Schreib eine Info.",
        "Bitte überweise {name} die Reisekosten auf {iban}.",
        "{name} hat die SV-Nummer {ssn} und fängt am Montag an. Erstelle die Anmeldung.",
        "schick {first} das wlan passwort {pw}",
        "überweis {first} die 50 euro auf {iban}",
        "Kontaktdaten {name}, Tel. {phone}, IBAN {iban} bitte ins CRM übernehmen.",
        "{name} hat am Telefon die Karte {card} genannt. Formuliere die Bestätigung.",
        "{name} braucht Zugriff auf den Server, Passwort {pw}. Schreib die Freigabe.",
        "Bitte trag für {name} das Konto {account} für die Gehaltszahlung ein.",
    ],
}


def device_fill(text):
    return text.format(imei=imei(), a=f"{rng.randint(0, 255):02X}", b=f"{rng.randint(0, 255):02X}",
                       c=f"{rng.randint(0, 255):02X}", g1=rng.randint(1000, 9999), g2=rng.randint(1000, 9999))


def v1_category(category, template):
    """Maps the v2 groups to the finer category names used in v1, so both versions can be analysed together."""
    t = template
    has = lambda *keys: any("{" + k + "}" in t for k in keys)
    named = has("name", "first", "last")
    if category == "name":
        return "names" if has("name2") or t.count("{name}") > 1 else "name"
    if category == "address":
        return "name + address" if named else "address"
    if category == "birth date":
        return "name + birth date" if named else "birth date"
    if category == "name + contact":
        parts = [k for k in ("address", "phone", "email") if has(k)]
        return "name + " + " + ".join(parts)
    if category == "device id":
        return "imei" if "IMEI" in t else "mac" if "MAC" in t else "gps"
    if category == "credit card":
        return "credit card + cvv" if has("cvv") else "credit card"
    if category == "name + block":
        for key, label in [("iban", "iban"), ("card", "credit card"), ("pw", "password"), ("ssn", "ssn"),
                           ("account", "account number"), ("pin", "pin")]:
            if has(key):
                return "name + " + label
    return category


CASUAL_MARKERS = (" ne ", " pls", " u ", "hey ", "kannst du", "schreib ne", "creds", "pw ")


def style(prompt):
    p = " " + prompt.lower() + " "
    return "casual" if prompt[0].islower() or any(m in p for m in CASUAL_MARKERS) else "formal"


def build(v1_path, harmless_path, out_path):
    v1 = pd.read_csv(v1_path)
    if "version" in v1.columns:  # file was already extended: rebuild v2 from the v1 rows only
        v1 = v1[v1["version"] == "v1"].drop(columns="version")
    rows = []
    for decision, groups in [("allow", ALLOW), ("warn", WARN), ("block", BLOCK)]:
        for category, prompts in groups.items():
            for text in prompts:
                prompt = device_fill(text) if category == "device id" else fill(text)
                rows.append({"prompt": prompt, "expected_decision": decision, "language": "de",
                             "category": v1_category(category, text), "style": style(prompt)})
    new = pd.DataFrame(rows)

    # checks: counts, no duplicates, no overlap with v1 or with the harmless training prompts
    counts = new["expected_decision"].value_counts().to_dict()
    assert counts == {"allow": 120, "warn": 140, "block": 140}, counts
    norm = lambda s: " ".join(str(s).lower().split())
    assert new["prompt"].map(norm).is_unique, "duplicate prompt"
    assert not new["prompt"].map(norm).isin(set(v1["prompt"].map(norm))).any(), "overlap with v1"
    if harmless_path:
        harmless = set(pd.read_csv(harmless_path)["source_text"].map(norm))
        overlap = new["prompt"].map(norm).isin(harmless)
        assert not overlap.any(), new.loc[overlap, "prompt"].tolist()

    v1 = v1.assign(version="v1")
    new = new.assign(version="v2")
    new.insert(0, "id", range(int(v1["id"].max()) + 1, int(v1["id"].max()) + 1 + len(new)))
    out = pd.concat([v1, new], ignore_index=True)[["id", "prompt", "expected_decision", "language",
                                                     "category", "style", "version"]]
    out.to_csv(out_path, index=False)
    print(out.groupby(["version", "expected_decision"]).size().unstack())
    print(out.groupby(["expected_decision", "style"]).size().unstack())
    print("total:", len(out))


if __name__ == "__main__":
    # run from the repo root: python src/own_prompts.py
    build("data/own_prompts.csv", "data/harmless_prompts.csv", "data/own_prompts.csv")
