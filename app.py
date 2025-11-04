import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import gender_guesser.detector as gender_detector
from flask import Flask, render_template, request, send_file, session
# import pandas as pd
import spacy
# import os
import re
# import webbrowser
import nltk
import unicodedata
from consts import determinant_de_l, determinant_du, determinant_de_la, determinant_des
# Télécharger les ressources nécessaires pour NLTK
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("wordnet")
# base_path = os.path.abspath(os.path.dirname(__file__))
# model_directory = os.path.join(base_path, 'fr_core_news_sm')

model_directory = 'fr_core_news_md'

nlp = spacy.load(model_directory)

# Load the SpaCy model
# nlp = spacy.load(model_directory)
gender_detector_instance = gender_detector.Detector(case_sensitive=False)


# Charger le modèle de langue spaCy
model_directory = 'fr_core_news_md'

nlp = spacy.load(model_directory)

# Load the SpaCy model
# nlp = spacy.load(model_directory)




# company_name = ["bureaux", "travaux", "grands", "groupes", "pavillons", "ministeres", "ministères", "réseaux", "reseaux", "groupement", "cheminées", "cheminees", 'mission', "maison", "companie", "cci", "fiduciaire", "compagnie", "caisse", "protection", "chambre", "commune", "place", "sncf", "banque", "fédération", "federation", "cheminée", "cheminee", "bureau", "travail", "groupe", "pavillon", "cabinet", "ministère", "ministere", "grand", "université", "universite", "réseau", "reseau", "club", "fc", "football", "groupement", "concret", "quai", "studio", "forum", "festival", "quai", "département", "departement", "grand", "institut", "agence", "atelier", "assurance", "association", "alliance", "etablissement", "établissement", "afnor", "essec", "appel", "orchestre", "académie", "academie", 'orchestre', "ensemble"]

company_name = determinant_des + determinant_de_la + determinant_de_l + determinant_du

# def segment_text(text, word_lists):

#     # Création du dictionnaire unique des mots connus (en minuscule pour normaliser)
#     word_set = set()
#     for word_list in word_lists:
#         word_set.update(w.lower() for w in word_list)

#     text = text.lower()
#     result = []
#     i = 0
#     while i < len(text):
#         match = None
#         # On teste tous les mots possibles en partant du plus long
#         for j in range(len(text), i, -1):
#             candidate = text[i:j]
#             if candidate in word_set:
#                 match = candidate
#                 break
#         if match:
#             result.append(match)
#             i += len(match)
#         else:
#             # Si aucun mot trouvé : ignorer un caractère et avancer
#             i += 1  # ou: result.append(text[i]); i += 1 pour le conserver

#     return result


def normalize_col(name):
    # Remove accents and special characters
    name = unicodedata.normalize('NFKD', str(name))
    name = name.encode('ASCII', 'ignore').decode('utf-8')
    # Standardize to lowercase with underscores
    return name.strip().lower().replace(' ', '_')

def decompose_string(input_string, word_list):
    if not word_list:
        return input_string
    
    word_set = set(word_list)
    max_len = max(len(word) for word in word_set)
    result = []
    unknown_buf = ""
    i = 0
    n = len(input_string)
    
    while i < n:
        found_word = None
        start = min(n - i, max_len)
        
        for length in range(start, 0, -1):
            candidate = input_string[i:i + length]
            if candidate in word_set:
                found_word = candidate
                break
        
        if found_word:
            if unknown_buf:
                result.append(unknown_buf)
                unknown_buf = ""
            result.append(found_word)
            i += len(found_word)
        else:
            unknown_buf += input_string[i]
            i += 1
    
    if unknown_buf:
        result.append(unknown_buf)
    
    return " ".join(result)

# Fonction pour vérifier si un mot appartient à une des listes de déterminants
def is_in_determinant_lists(word, determinant_de_l, determinant_du, determinant_de_la):
    base_word = word[:-1]  # Enlever le dernier caractère (s ou x)
    return (base_word in determinant_de_l or 
            base_word in determinant_du or 
            base_word in determinant_de_la)


# Fonction pour supprimer la virgule et tout ce qui suit dans un texte
def remove_comma_and_following(text):
    if isinstance(text, str):
        return text.split(',')[0]
    else:
        return text
    
# Fonction pour supprimer les deux-points et tout ce qui suit dans un texte
def remove_twopoints_and_following(text):
    if isinstance(text, str):
        return text.split(':')[0]
    else:
        return text
    
# Fonction pour supprimer les termes de dénomination sociale dans un texte
def remove_enterprise_term(text):
    if isinstance(text, str):
        # Variantes de formes sociales, avec ou sans points/espaces
        denominations = [
            r'\bInc\.?\b',
            r'\bSAS\b', r'\bS\.?\s*A\.?\s*S\.?\b',
            r'\bSARL\b', r'\bS\.?\s*A\.?\s*R\.?\s*L\.?\b',
            r'\bSA\b', r'\bS\.?\s*A\.?\b',
            r'\bLtd\.?\b',
            r'\bLLC\b',
            r'\bCorp\.?\b',
            r'\bGmbH\b',
            r'\bCo\.?\b',
            r'\bPty\.?\b',
            r'\bAG\b',
            r'\bFreelance\b'
        ]
        
        pattern = '|'.join(denominations)
        
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    else:
        return text


def remove_hyphen_and_following(text):
    if isinstance(text, str):
        return text.split('- ')[0]
    else:
        return text
    
def remove_pipe_and_following(text):
    if isinstance(text, str):
        return text.split('|')[0]
    else:
        return text

def remove_dot(text):
    if isinstance(text, str):
        return text.split('.')[0]
    else:
        return text

# Fonction pour insérer un espace dans un texte après un nom d'entreprise reconnu
def create_space_in_societe(text, company_names):
    if isinstance(text, str) and isinstance(company_names, list):
        # Trouver le mot le plus long correspondant au début du texte
        matching_word = ""
        for word in company_names:
            if text.startswith(word) and len(word) > len(matching_word):
                matching_word = word

        # Ajouter un espace après le mot correspondant s'il n'est pas déjà présent
        if matching_word:
            index = text.find(matching_word)
            if index + len(matching_word) < len(text) and text[index + len(matching_word)] != " ":
                text = text[:index + len(matching_word)] + " " + text[index + len(matching_word):]
                
    return text

def detect_first_word_type(sentence):
    doc = nlp(sentence)
    first_token = doc[0]
    if first_token.pos_ == "DET":
        return "determinant"
    elif first_token.pos_ == "NOUN":
        if first_token.ent_type_ == "PROPN":
            return "nom_propre"
        else:
            return "nom_commun"
    else:
        return "autre"

def remove_non_latin_characters(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0000-\u00FF]', '', text)
    else:
        return text

# Fonction pour supprimer les parenthèses et leur contenu dans un texte
def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', text)

# Fonction pour supprimer les noms de famille avec une seule lettre suivie d'un point
def remove_single_letter_names(text):
    if isinstance(text, str):
        # Supprimer les noms qui sont une seule lettre suivie d'un point
        text = re.sub(r'\b\w\.\b', '', text)
        # Nettoyer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return text

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Route principale pour l'index
@app.route('/')
def index():
    return render_template('index.html')

# Route pour traiter le fichier CSV
@app.route('/process_csv', methods=['POST'])
def process_csv():
    file = request.files['file']
    filename = file.filename
    session['filename'] = filename

    try:
        df = pd.read_csv(file)
    except:
        file.seek(0)
        try:
            df = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            return f"Error reading file: {str(e)}"
    def detect_gender(name):
        # If name is missing (NaN), empty, or not a string → just return an empty civilité
        if name is None or (not isinstance(name, str)) or not name.strip():
            return ""
        return gender_detector_instance.get_gender(name)


    # Normalize column names with accent removal
    def normalize_col(name):
        name = unicodedata.normalize('NFKD', str(name))
        name = name.encode('ASCII', 'ignore').decode('utf-8')
        return name.strip().lower().replace(' ', '_')
    
    df.columns = [normalize_col(col) for col in df.columns]

    # Expanded column mapping
    column_map = {
        'firstname': ['firstname', 'prenom', 'givenname', 'prenom', 'first_name'],
        'suggestion_de_prenom': ['suggestionprenom', 'prenomsuggestion', 'suggestion_de_prenom'],
        'societe': ['societe', 'company', 'entreprise'],
        'civilite': ['civilite', 'title', 'gender']
    }

    for standard_name, variants in column_map.items():
        for variant in variants:
            if variant in df.columns:
                df.rename(columns={variant: standard_name}, inplace=True)

    # Handle missing firstnames using suggestion column
    # if 'suggestion_de_prenom' in df.columns:
    df['firstname'] = df['firstname'].fillna('undefined')
    app.logger.info("firstname null counts: %s", df['firstname'].isnull().value_counts())
    missing_first = df[df['firstname'].isnull()]
    if not missing_first.empty:
        return "error: Missing firstnames in the file. Please check the file and try again."


    # Validate required columns
    required = ['societe', 'civilite', 'email']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        return render_template('error.html',
                            message=f"Colonnes obligatoires manquantes: {', '.join(missing)}")
    print("Mapped columns:", df.columns.tolist())
    app.logger.info("Firstname column values / null‐counts:\n%s", df['firstname'].isnull().value_counts())
    app.logger.info("Sample of firstname column:\n%s", df['firstname'].head(10))

    # Process rows with missing firstnames instead of failing
    df['civilite'] = df.apply(lambda row: (
        row['civilite']
        if pd.notnull(row['civilite'])
        else (
            # only call detect_gender if we actually have a string
            detect_gender(row['firstname'])
            if isinstance(row['firstname'], str) and row['firstname'].strip()
            else ""
        )
    ), axis=1)


    
    if missing:
        return render_template('error.html',
                            message=f"File missing required columns: {', '.join(missing)}")

    # Rest of your processing code using .get() for column access
    for index, row in df.iterrows():
        current_firstname = row['firstname']
        current_email     = row['email']
        current_societe   = row['societe']
        current_civilite  = row['civilite']
        # Check if the first name is empty or NaN
            # Handle missing first name

    # Fonction pour détecter le genre d'un prénom

    # Fonction pour vérifier si un nom commun commence par une voyelle
    def is_commom_noun_starting_with_vowel(word) -> bool:

        word = word.lower()

        pos_tags = pos_tag([word])

        if pos_tags[0][1] in ['NN', 'NNS', 'NNP', 'NNPS'] and word[0] in ['a', 'e', 'i', 'o', 'u', "é", "è", "î", "ô", "à"]:
            return True

        synsets = wn.synsets(word, pos=wn.NOUN)
        if synsets:
            if any(word[0] in ['a', 'e', 'i', 'o', 'u', "é", "è", "î", "ô", "à"] for synset in synsets for word in synset.lemma_names()):
                return True
        return False

    # Fonction pour déterminer le préfixe et le déterminant à utiliser avec le nom d'une entreprise
    def determiner_prefixe_pronom(nom_entreprise):
    # Vérifier si la colonne "societe" est vide ou NaN
        if not nom_entreprise or nom_entreprise.strip() == "" or nom_entreprise == "au sein de votre entreprise" or pd.isna(nom_entreprise):
            return "au sein", "de votre entreprise"
        doc = nlp(nom_entreprise.strip())
        if not doc:
            return "chez", ""
        
        premier_mot = doc[0].text.lower()

        tokens = word_tokenize(nom_entreprise.lower())
        prefixe = 'au sein'
        determinant = ""
        
        # print(determinant_des)
        
        try:
            if tokens[0][0].lower() == "l" and tokens[0][1] == "'":
                prefixe = "au sein"
                determinant = "de"
            elif premier_mot in determinant_de_l:
                prefixe = "au sein"
                determinant = "de l'"
            elif premier_mot in determinant_du: #"mannequin", "chancelier", "chineur", "cireur"
                prefixe = "au sein"
                determinant = "du"
            elif premier_mot in determinant_des or (premier_mot[:-1] in determinant_des and premier_mot[-1] in ['s', 'x']):
                prefixe = "au sein"
                determinant = "des"
            elif premier_mot in determinant_de_la:
                prefixe = "au sein"
                determinant = "de la"        # Règles spécifiques pour certaines entreprises
            elif detect_first_word_type(nom_entreprise) == "nom_commun":
                if is_commom_noun_starting_with_vowel(nom_entreprise):
                    print("Nom commun commençant par une voyelle détecté :", nom_entreprise)
                    determinant = "d'"
                    if (nom_entreprise.strip() == "Total"):
                        print("Total détecté, utilisation de 'chez'")
                        prefixe = "chez"
                        determinant = ""
                else:
                    determinant = "de"
            elif detect_first_word_type(nom_entreprise) == "determinant":
                if tokens[0].lower() == "les":
                    prefixe = "chez"
                elif tokens[0].lower() == "le":
                    prefixe = "chez"
                elif tokens[0].lower() == "la":
                    determinant = "de"
        except IndexError:
            prefixe = "error"
            determinant = ""
        try:
            if (prefixe == "au sein" and determinant == "") or (tokens[0].lower() in ['vertical', "shine", "illuminerie", "iconoproduction", "ctv", "unlimitail", "datadome", "iqo", "smart", "vinci"]):
                prefixe = "chez"
        except IndexError:
            prefixe = "error"
            determinant = ""
        try:
            if prefixe == "au sein" and tokens[0].lower() in ["umake", "owkin", "isocel", "isocel.", "isocel.leclerc", "leclerc", "carrefour", "géant", "geant", "imerys", "afept", "acteon", "valorem", "voxelis", "vatel", "yzar", "accenture", "aemsofts", "sii", "sll", "metapolis", "memoandco", "maincare", "fayat", "eove", "cybertek", "cultura", ""] and prefixe == "au sein":
                prefixe = "chez"
                determinant = ""
            if prefixe == "chez" and tokens[0].lower in ["agriculteur", "agricultrice", "agriculture", "amateur", "analyste", "arbitre", "artiste"]:
                determinant = "l'"
            elif prefixe == "chez" and tokens[0].lower in ["capitaine", ""]:
                determinant == "le"
            elif prefixe == "chez" and tokens[0].lower in [""]:
                determinant = "la"
            
        except:
            prefixe = "chez"
            determinant = ""
        return prefixe, determinant


    # Créer un nouveau DataFrame pour les lignes avec des valeurs d'e-mail nulles
    
    df_null_email = pd.DataFrame()  
    if 'email' in df.columns:
        df_null_email = df[df['email'].isnull()]

    if 'Suggestion de Prénom' not in df.columns:
        df['Suggestion de Prénom'] = ""
    
    if 'chez' not in df.columns:
        df['chez'] = ""
    all_current_columns = df.columns.tolist()
    expected_leading_cols = ['civilite', 'firstname', 'suggestion_de_prenom', 'lastname']
    col_to_insert_1 = 'chez'
    col_to_insert_2 = 'societe'

    new_order = []
    processed_cols = set()

    for col_name in expected_leading_cols:
        if col_name in all_current_columns:
            new_order.append(col_name)
            processed_cols.add(col_name)

    # 2. Add 'chez' (it should exist at this point due to the check above)
    if col_to_insert_1 in all_current_columns:
        new_order.append(col_to_insert_1)
        processed_cols.add(col_to_insert_1)
    
    # 3. Add 'societe' (it should exist as it's a required column from earlier check)
    if col_to_insert_2 in all_current_columns:
        new_order.append(col_to_insert_2)
        processed_cols.add(col_to_insert_2)
    
    # 4. Add all other remaining columns from the DataFrame
    for col_name in all_current_columns:
        if col_name not in processed_cols:
            new_order.append(col_name)
            # processed_cols.add(col_name) # No need to add to set here

    columns_order = new_order


    df = df[columns_order]

    total_rows = len(df)
    civility_columns = ['Civilité', 'civilite', 'Civilite', 'civilité']
    existing_civility_col = next((col for col in civility_columns if col in df.columns), None)
    if existing_civility_col:
        # Standardize the column name to 'civilité'
        df.rename(columns={existing_civility_col: 'civilité'}, inplace=True)
    else:
        # Create the column if it doesn't exist
        df['civilité'] = None

    # Renommer la colonne 'nom' en 'lastname'
    if 'nom' in df.columns:
        df.rename(columns={'nom': 'lastname'}, inplace=True)

    # Compter le nombre total de lignes où la civilité est "Monsieur"
    if 'civilité' in df.columns:
        # Compte le nombre de lignes où la civilité est "Monsieur"
        count_monsieur = max((df['civilité'] == 'Monsieur').sum(),1)
        print("Nombre total de lignes avec civilité 'Monsieur':", count_monsieur, "nombre total de lignes", total_rows)

        # Utilise tqdm pour afficher une barre de progression lors du chargement du fichier
        with tqdm(total=total_rows, desc="Chargement du fichier") as pbar_load:
            # Insère une colonne 'lastname' vide au début du DataFrame si elle n'existe pas
            if 'lastname' not in df.columns:
                df.insert(loc=0, column='lastname', value=None)

            # Parcourt chaque ligne du DataFrame
            for index, row in df.iterrows():
                # Si la colonne 'lastname' n'est pas vide et ne contient pas de point, ajoute la valeur à 'Suggestion de Prénom'
                lastname_val = row.get('lastname')
                print("La valeur du nom est")
                print(lastname_val)
                if isinstance(lastname_val, str) and (re.match(r'^\w\.$', lastname_val.strip()) or re.match(r'^\w\;$', lastname_val.strip())):
                    print(lastname_val)
                    df.at[index, 'lastname'] = ''
                elif isinstance(lastname_val, str) and lastname_val:
                    # Set the first letter to a capital letter
                    df.at[index, 'lastname'] = lastname_val[0].upper() + lastname_val[1:] if len(lastname_val) > 1 else lastname_val.upper()
                
                # Si la colonne 'societe' est une chaîne de caractères
                societe_val = row.get('societe', '')
                if pd.notna(societe_val):
                    cleaned_societe = remove_parentheses(societe_val)
                    cleaned_societe = remove_hyphen_and_following(cleaned_societe)
                    cleaned_societe = remove_pipe_and_following(cleaned_societe)
                    cleaned_societe = remove_twopoints_and_following(cleaned_societe)
                    cleaned_societe = remove_comma_and_following(cleaned_societe)
                    cleaned_societe = remove_enterprise_term(cleaned_societe)
                    cleaned_societe = remove_non_latin_characters(cleaned_societe)
                    cleaned_societe = remove_dot(cleaned_societe)

                    # cleaned_societe = decompose_string(cleaned_societe, company_name)
                    df.at[index, 'societe'] = cleaned_societe.strip()
                else:
                    cleaned_societe = ''

                prefix, determinant = determiner_prefixe_pronom(cleaned_societe)

                if determinant and not determinant.endswith("'"):
                    chez_string = f"{prefix} {determinant} "
                else:
                    chez_string = f"{prefix} {determinant}"
                df.at[index, 'chez'] = chez_string

                # Si la colonne 'firstName' est vide
                if pd.isnull(row['firstname']):
                    if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'civilité'] = "Monsieur"
                    else:
                        df.at[index, 'civilité'] = 'Madame'
                # Si la colonne 'civilité' est vide
                elif pd.isnull(row['civilité']):
                    first_names = row['firstname'].split()
                    first_name = first_names[0]
                    print(first_name);
                    gender = detect_gender(first_name)
                    print(gender)
                    if gender in ["andy", "unknown", "error"]:
                        if len(first_names) > 1:
                            second_name = first_names[1]
                            gender = detect_gender(second_name)
                    if gender == "female" or gender == "mostly_female":
                        df.at[index, 'civilité'] = "Madame"
                    elif gender == "male" or gender=="mostly_male":
                        print("Setting male to Monsieur")
                        df.at[index, 'civilité'] = "Monsieur"
                    elif gender == "andy":
                        if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'civilité'] = "Monsieur"
                        else:
                            df.at[index, 'civilité'] = 'Madame'
                    elif gender == "unknown":
                        if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'civilité'] = "Monsieur"
                        else:
                            df.at[index, 'civilité'] = 'Madame'
                    else:
                        df.at[index, 'civilité'] = "Erreur"

                # Si la colonne 'email' est vide
                if pd.isnull(row['email']):
                    if pd.isnull(row['firstname']):
                        df_null_email.at[index, 'civilité'] = "Prénom non attribué"
                    else:
                        first_names = row['firstname'].split()
                        first_name = first_names[0]
                        gender = detect_gender(first_name)
                        print(first_name)
                        print(gender)
                        if gender in ["andy", "unknown", "error"]:
                            if len(first_names) > 1:
                                second_name = first_names[1]
                                gender = detect_gender(second_name)
                        if gender == "female" or gender == "mostly_female":
                            df_null_email.at[index, 'civilité'] = "Madame"
                        elif gender == "male" or gender=="mostly_male":
                            df_null_email.at[index, 'civilité'] = "Monsieur"
                        elif gender == "andy":
                            if total_rows / count_monsieur >= 0.5:
                                df.at[index, 'civilité'] = "Monsieur"
                            else:
                                df.at[index, 'civilité'] = 'Madame'
                        elif gender == "unknown":
                            if count_monsieur / total_rows < 50:
                                df_null_email.at[index, 'civilité'] = "Madame"
                            else:
                                df_null_email.at[index, 'civilité'] = "Monsieur"
                        else:
                            df_null_email.at[index, 'civilité'] = "Erreur"
                
                # Vérifie si une partie du nom de famille correspond au nom de la société
                if isinstance(row['lastname'], str) and isinstance(row['societe'], str):
                    last_name_parts = row['lastname'].split()
                    for part in last_name_parts:
                        if part.lower() in row['societe'].lower():
                            df.at[index, 'Match Entreprise'] = 'Oui'
                            break
                    else:
                        df.at[index, 'Match Entreprise'] = 'Non'
                # Met à jour la barre de progression
                pbar_load.update(1)
        print("\nLancement du nettoyage final de la colonne 'civilité'...")

    # 1. Standardiser les remplacements directs
    # Convertir la colonne en chaîne de caractères et en minuscules pour une comparaison fiable
    civilite_lower = df['civilité'].astype(str).str.strip().str.lower()

    # Remplacer les variantes masculines
    df.loc[civilite_lower.isin(['male', 'mostly_male', 'mr', 'mister', 'm']), 'civilité'] = 'Monsieur'

    # Remplacer les variantes féminines
    df.loc[civilite_lower.isin(['female', 'mostly_female', 'mme', 'ms', 'mrs', 'miss', 'f']), 'civilité'] = 'Madame'


    # 2. Gérer les valeurs inconnues ou ambiguës restantes
    # Compter le nombre de 'Monsieur' et 'Madame'
    monsieur_count = (df['civilité'] == 'Monsieur').sum()
    madame_count = (df['civilité'] == 'Madame').sum()

    # Déterminer le genre majoritaire (par défaut 'Monsieur' en cas d'égalité)
    majority_gender = 'Monsieur' if monsieur_count >= madame_count else 'Madame'
    print(f"Genre majoritaire détecté : {majority_gender} ({monsieur_count} H / {madame_count} F)")

    # Identifier toutes les lignes qui ne sont ni 'Monsieur' ni 'Madame'
    # Celles-ci incluent 'unknown', 'andy', 'nan', les chaînes vides, etc.
    rows_to_update = ~df['civilité'].isin(['Monsieur', 'Madame'])
    df.loc[rows_to_update, 'civilité'] = majority_gender

    print("Nettoyage final de la colonne 'civilité' terminé.")

    # Appliquer les nouvelles transformations demandées
    
    # 1. Supprimer les noms de famille avec une seule lettre suivie d'un point
    if 'lastname' in df.columns:
        df['lastname'] = df['lastname'].apply(remove_single_letter_names)
    
    # 2. Supprimer les emails répétés (garder une seule occurrence)
    if 'email' in df.columns:
        df = df.drop_duplicates(subset=['email'], keep='first')
    
    # 3. Ajouter la colonne nbcar avec le nombre de caractères du nom
    if 'lastname' in df.columns:
        df['nbcar'] = df['lastname'].astype(str).apply(len)

    # Appliquer les mêmes transformations à df_null_email
    if not df_null_email.empty:
        if 'lastname' in df_null_email.columns:
            df_null_email['lastname'] = df_null_email['lastname'].apply(remove_single_letter_names)
        if 'lastname' in df_null_email.columns:
            df_null_email['nbcar'] = df_null_email['lastname'].astype(str).apply(len)

    # df_combined = pd.concat([df, df_null_email], ignore_index=True)

    # df_combined = df_combined.dropna(subset=['email'])

    # Obtient le nom de fichier de la session
    filename = session.get('filename')
    print(filename)

    # Enlève l'extension du fichier
    filename_without_extension = filename.rsplit('.', 1)[0]

    # Remplace les espaces par des underscores
    filename_without_extension = filename_without_extension.replace(' ', '_')

    # Ajoute le suffixe '_updated.xlsx'
    output_file = filename_without_extension + '_updated.xlsx'
    session['output_file'] = output_file

    with pd.ExcelWriter(output_file) as writer:
        # Écrit le DataFrame original dans la première feuille
        df.dropna(subset=['email']).to_excel(writer, sheet_name='Data', index=False)

        # Écrit le DataFrame avec les emails manquants dans la deuxième feuille, s'il n'est pas vide
        if not df_null_email.empty:
            df_null_email.to_excel(writer, sheet_name='Null-email', index=False)
    
    # Supprime les lignes avec des emails manquants du DataFrame
    if 'email' in df.columns:
        df.dropna(subset=['email'], inplace=True)
    # Affiche le résultat dans le template HTML
    return render_template('result.html', data=df.to_html(), df_email_data=df_null_email.to_html(), filename=output_file)

# Route pour télécharger le fichier Excel
@app.route('/download_excel')
def download_excel():
    excel_file_path = session.get('output_file')
    return send_file(excel_file_path, as_attachment=True)

# Point d'entrée de l'application
if __name__ == '__main__':
    # webbrowser.open('http://localhost:5000')
    app.run(host="0.0.0.0", port=5000)