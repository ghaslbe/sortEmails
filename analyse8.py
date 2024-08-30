import imaplib
import email
import os
import json
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
from transformers import DetrImageProcessor, DetrForObjectDetection, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import io
import requests
import pytesseract
from pytesseract import TesseractError

# guentherhaslbeck@gmail.com
# https://youtu.be/dCP2TYrzMOc

# Unterdrücke nicht kritische Warnungen
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def log(message):
    print(f"[INFO] {message}")

# Verbindung zum IMAP-Server herstellen
imap_server = 'mail.xxxx.com'
imap_port = 993
username = 'acxxxxxxxxxxga.com'
password = '.xxxxxxxxx'

log("Stelle Verbindung zum IMAP-Server her")
mail = imaplib.IMAP4_SSL(imap_server, imap_port)
mail.login(username, password)

# Posteingang auswählen
log("Wähle Posteingang aus")
mail.select('inbox')

# Alle ungelesenen Emails auswählen
log("Rufe alle ungelesenen E-Mails ab")
status, messages = mail.search(None, 'ALL')

# Liste der Nachrichten-IDs erhalten
message_ids = messages[0].split()

# Ordner "Gelesen" und "Analyse" erstellen, falls nicht vorhanden
def create_folder_if_not_exists(mail, folder_name):
    log(f"Überprüfe, ob der Ordner '{folder_name}' existiert")
    status, folders = mail.list()
    if folder_name not in [folder.decode().split(' "/" ')[-1] for folder in folders]:
        log(f"Erstelle den Ordner '{folder_name}'")
        mail.create(folder_name)

create_folder_if_not_exists(mail, "Gelesen")
create_folder_if_not_exists(mail, "Analyse")

# Funktion zur Objekterkennung in Bildern
def detect_objects_in_image(image):
    try:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detected_objects.append({
                "label": model.config.id2label[label.item()],
                "score": round(score.item(), 3)
            })
        
        log(f"Objekterkennung abgeschlossen: {len(detected_objects)} Objekte erkannt")
        return detected_objects
    except Exception as e:
        log(f"Fehler bei der Objekterkennung: {str(e)}")
        return {"error": f"Object detection failed: {str(e)}"}

# Funktion zur Bildbeschreibung
def describe_image(image):
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        log("Bildbeschreibung erfolgreich generiert")
        return description
    except Exception as e:
        log(f"Fehler bei der Bildbeschreibung: {str(e)}")
        return f"Image description failed: {str(e)}"

# Funktion zur Textextraktion (OCR)
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        log("Textextraktion aus Bild erfolgreich")
        return text
    except TesseractError as e:
        log(f"Fehler bei der Textextraktion: {str(e)}")
        return f"OCR failed: {str(e)}"
    except Exception as e:
        log(f"Allgemeiner Fehler bei der Textextraktion: {str(e)}")
        return f"OCR failed: {str(e)}"

# Funktion zur Kommunikation mit der Ollama-API
def call_ollama(prompt):
    log("rufe ollama mit gemma2 auf")
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "gemma2:latest",
        "prompt": prompt,
        "temperature": 0.1,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # raises exception when not a 2xx response

        json_response = response.json()
        log("Ollama API erfolgreich aufgerufen:")
        # log(json_response)
        return json_response.get('response', 'No response key found')
    except requests.exceptions.RequestException as e:
        log(f"Fehler beim Aufruf der Ollama API: {str(e)}")
        return f"Ollama API call failed: {str(e)}"


def extract_json_from_response(response):
    try:
        # Suche den Beginn des JSON-Blocks
        start_idx = response.find("```json")
        if start_idx != -1:
            # Extrahiere den JSON-Text ab "```json"
            json_str = response[start_idx + len("```json"):].strip()

            # Finde das Ende des JSON-Blocks (``` am Ende)
            end_idx = json_str.rfind("```")
            if end_idx != -1:
                json_str = json_str[:end_idx].strip()

            # Versuche das JSON zu parsen
            json_data = json.loads(json_str)
            log("---- response --")
            log(json_data)
            log("---- response --")
            return json_data
        else:
            log("Fehler: Kein JSON-Block in der Antwort gefunden")
            return {"error": "No JSON block found in response"}
    except json.JSONDecodeError as e:
        log(f"Fehler beim Parsen von JSON: {str(e)}")
        return {"error": f"JSON decoding failed: {str(e)}"}
    except Exception as e:
        log(f"Allgemeiner Fehler: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


# Textanalysefunktionen
def analyze_text(extracted_text):
    log("Analysiere E-Mail-Text auf Unstimmigkeiten")
    prompt = f"""
    was fällt dir an folgendem text auf? sind die daten in sich stimmig. Prüfe die einzelnen angaben. Stimmen zum beispiel land des Empfängers und Iban überein? was fällt dir auf ? Was ist nicht stimmig, was könnte eine drohung sein?
    Ignoriere Dinge wie fehlende Umlaute, Zahlen oder Abkürzungen ausser sie deuten auf Missbrauch oder drohungen hin.
    Erstelle deine ausgaben Nur als json mit 2 Bereichen. Gib keinerlei andere Daten aus, ausser JSON.
    Seltsam: Hier ist eine Liste von auffälligkeiten hinterlegt
    Scorewert: 1: unauffällig, 100: auffällig
    Starte mit ```JSON und ende mit ```
    Gib immer ein JSON Objekt aus.
    Hier ist der Text: 
    {extracted_text}
    """
    # Aufruf der Ollama-API und Ergebnis holen
    response = call_ollama(prompt)
    
    # Verwende die neue Methode, um den JSON-Teil aus der Antwort zu extrahieren
    return extract_json_from_response(response)

def analyze_address(extracted_text):
    log("Analysiere E-Mail-Text auf Adressen, Emails, Telefonnummern und Kundennummern")
    prompt = f"""
    lese aus dem folgenden Text alle postadressen, alle emailadressen, alle Kundennummern, Versicherungsnummern und alle Handynummern.
    Versuche nur die herauszulesen die vom absender sind.  Gib die Daten  als json aus als Absenderadresse. 
    Versuche weitere Daten herauszulesen wie Kundennummern use.  Gib die Daten  als json aus als WeitereDaten. 
    Gib keinerlei andere Daten aus, ausser JSON.
    Starte mit ```JSON und ende mit ```
    Gib immer ein JSON Objekt aus.
    Hier ist der Text: 
    {extracted_text}
    """
    # Aufruf der Ollama-API und Ergebnis holen
    response = call_ollama(prompt)
    
    # Verwende die neue Methode, um den JSON-Teil aus der Antwort zu extrahieren
    return extract_json_from_response(response)

def identify_document_type(extracted_text):
    log("Identifiziere Dokumententyp basierend auf dem E-Mail-Text")
    prompt = f"""
    Erkenne, ob es sich um eines der folgenden Dokumente handelt.
    Gib den Typ als Json zurück. Gib keinerlei andere Daten aus, ausser die JSON Daten. 
    Starte mit ```JSON und ende mit ```
    Gib immer ein JSON Objekt aus.
    Hier ist der Text: 

    Hier die Kategorien dazu:
    1. Versicherungsnummer
    2. Versicherungssparte
    3. Kontaktanlass
    4. Name
    5. Adresse
    6. E-Mail
    7. Geschäftsvorfall, aus nachfolgenden Kategorien und Unterkategorien:
    Versicherungssparte
    * Kfz
    * Hausrat
    * Wohngebäude
    * Haftpflicht
    * Lebensversicherung
    * Krankenversicherung
    Schadenmeldung
    * Kfz-Schäden
    * Hausrat- und Wohngebäudeschäden
    * Haftpflichtschäden
    * Personenschäden
    Vertragsangelegenheiten
    * Vertragsabschluss
    * Vertragsänderung
    * Vertragskündigung
    * Vertragsverlängerung
    Leistungsanfragen
    * Leistungsanträge
    * Leistungsabrechnungen
    * Rückfragen zu Leistungen
    Rechnungen und Zahlungen
    * Beitragszahlungen
    * Erstattungsanfragen
    * Mahnungen
    * Zahlungseingänge
    Beschwerden und Reklamationen
    * Leistungsbeschwerden
    * Servicebeschwerden
    * Schadenregulierungsbeschwerden
    Informationen und Anfragen
    * Produktinformationen
    * Beratungsgesuche
    * allgemeine Anfragen
    Dokumentenanforderungen
    * Kopien von Verträgen
    * Nachweise und Bescheinigungen
    * Gutachtenanforderungen
    Risikomeldungen und Anderungsanzeigen
    * Risikoänderungen (z.B. Wohnortwechsel, Fahrzeugwechsel)
    * Berufliche Änderungen
    * Persönliche Änderungen (z.B. Eheschließung, Geburt)
    Marketing und Angebote
    * Werbemitteilungen
    * Sonderangebote
    * Kundenumfragen
    Sonstiges
    * Unkategorisierte Anliegen
    * Allgemeine Korrespondenz
    
    {extracted_text}
    """
    # Aufruf der Ollama-API und Ergebnis holen
    response = call_ollama(prompt)
    
    # Verwende die neue Methode, um den JSON-Teil aus der Antwort zu extrahieren
    return extract_json_from_response(response)

# Nachrichten durchlaufen
for msg_id in message_ids:
    # Nachricht abrufen
    log(f"Verarbeite Nachricht ID {msg_id.decode()}")
    status, msg_data = mail.fetch(msg_id, '(RFC822)')
    
    # Email-Objekt aus den Bytes erstellen
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)
    
    # Titel und Body extrahieren
    subject = msg['subject']
    from_name = email.utils.parseaddr(msg['From'])[0]
    from_email = email.utils.parseaddr(msg['From'])[1]
    send_date = msg['date']
    
    body_text = ""
    body_html = ""
    attachments = []  # Initialisierung der Liste
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body_text = part.get_payload(decode=True).decode()
            elif content_type == "text/html":
                body_html = part.get_payload(decode=True).decode()
            elif part.get_content_maintype() == 'image':
                # Bildanhang verarbeiten
                image_data = part.get_payload(decode=True)
                try:
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    log(f"Fehler beim Öffnen des Bildes: {str(e)}")
                    attachments.append({
                        "file_name": part.get_filename(),
                        "content_type": content_type,
                        "size": len(image_data),
                        "error": f"Failed to open image: {str(e)}"
                    })
                    continue
                
                # Objekterkennung, Bildbeschreibung und OCR auf das Bild anwenden
                detected_objects = detect_objects_in_image(image)
                image_description = describe_image(image)
                extracted_text = extract_text_from_image(image)
                
                # Erstellung des Prompts für die Ollama-Schnittstelle
                prompt = f"Erstelle eine Beschreibung eines Bilds. Anbei die Json-Daten:\n" + json.dumps({
                    "detected_objects": detected_objects,
                    "image_description": image_description,
                    "extracted_text": extracted_text
                }, indent=4)
                
                ollama_response = call_ollama(prompt)
                
                # Erkennungsergebnisse zu den Anhängen hinzufügen
                attachment = {
                    "file_name": part.get_filename(),
                    "content_type": content_type,
                    "size": len(image_data),
                    "detected_objects": detected_objects,
                    "image_description": image_description,
                    "extracted_text": extracted_text,
                    "ollama_description": ollama_response
                }
                attachments.append(attachment)
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            body_text = msg.get_payload(decode=True).decode()
        elif content_type == "text/html":
            body_html = msg.get_payload(decode=True).decode()

    # Konvertiere HTML zu Plain Text, falls kein Plain Text vorhanden ist
    if not body_text and body_html:
        soup = BeautifulSoup(body_html, "html.parser")
        body_text = soup.get_text()
    
    # Textanalyse durchführen
    analysis_result = analyze_text(body_text)
    address_analysis = analyze_address(body_text)
    document_type = identify_document_type(body_text)

    # JSON-Datenstruktur erstellen und Fehlerbehandlung beim JSON-Parsing hinzufügen
    try:
        email_data = {
            "Email": {
                "Sendername": from_name,
                "Senderemail": from_email,
                "Senddate": send_date,
                "Mailtitle": subject,
                "Mailbody_HTML": body_html,
                "Mailbody_TXT": body_text,
                "Analysis_Result": analysis_result ,
                "Address_Analysis": address_analysis ,
                "Document_Type": document_type
            },
            "Attachments": attachments
        }
    except json.JSONDecodeError as e:
        log(f"Fehler beim Parsen von JSON: {str(e)}")
        continue
    
    # Neue Nachricht erstellen mit Email- und Anhangsdaten im JSON-Format
    new_msg = MIMEMultipart()
    new_msg['Subject'] = f"Analyse: {subject}"
    new_msg['From'] = username
    new_msg['To'] = username  # Kann beliebig angepasst werden
    
    body_content = json.dumps(email_data, indent=4)
    new_msg.attach(MIMEText(body_content, 'plain'))
    
    # Nachricht als Email-String
    new_email_str = new_msg.as_string()
    
    # Neue Nachricht in den "Analyse" Ordner ablegen
    log(f"Lege Analyse-E-Mail in den 'Analyse'-Ordner ab")
    mail.append('Analyse', '', imaplib.Time2Internaldate(email.utils.localtime()), new_email_str.encode('utf-8'))
    
    # Verschiebe die Original-Nachricht in den "Gelesen"-Ordner
    log(f"Verschiebe Originalnachricht in den 'Gelesen'-Ordner")
    mail.copy(msg_id, 'Gelesen')
    mail.store(msg_id, '+FLAGS', '\\Deleted')

# Gelöschte E-Mails endgültig entfernen
log("Entferne gelöschte E-Mails endgültig")
mail.expunge()

# Verbindung trennen
log("Trenne Verbindung zum IMAP-Server")
mail.logout()

