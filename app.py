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


# Listes de déterminants pour différentes prépositions
determinant_de_l = ['agence', "union", "éducation", "education", "institut", "agence", "atelier", "assurance", "association", "alliance", "etablissement", "établissement", "afnor", "essec", "appel", "orchestre", "académie", "academie", 'orchestre', "ensemble", "abaissement", "abajoue", "abandon", "abaque", "abat", "abattant", "abattement", "abattis", "abattoir", "abaya", "abbaye", "aberration", "abime", "abîme", "abject", "ablatif", "ablation", "ablette", "abnégation", "abnegation", "aboiement", "abois", "abolition", "abolitionnisme", "abolissioniste", "abominable", "abondance", "abondant", "abonné", "abonnement", "abord", "abordage", "aborigène", "aboulie", "aboutissant", "aboutissement", "abrasif", "abreuvoir", "abri", "abricot", "abricotier", "abrogation", "absolu", "absolution", "abysse", "académie", "academie", "acajou", "acanthe", "accalmie", "accastillage", "accélération", "acceleration", "accent", "accentuation", "acceptation", "acception", "accès", "acces", "accessit", "accessoire", "accessoiriste", "acclamation", "acclimatement", "accointance", "accolade", "accomandation", "accompagnat", "accompagnement", "accord", "accordéoniste", "accordeur", "accotement", "accotoir", "accoutrement", "accréditation", "accreditation", "accrobranche", "accroche", "accueil", "acculturation", "acerola", "acérola", "achat", "acheteur", "acheteuse", "acide", "aciérage", "acierage", "acierie", "aciérie", "acolyte", "acropole", "acrostiche", "acte", "action", "acting", "actionnaire", "actionnariat", "activisme", "activiste", "activité", "activite", "actualité", "actualite", "acupuncteur", "acupunctrice", "accunpuncture", "adage", "adaptation", "adaptateur", "adaptatrice", "adhérence", "adhésif", "adherence", "adhesif","adjoint", "adjointe", "adjonction", "administrateur", "admission", "adoption", "adoucissant", "adsl", "aero", "aéro", "aérodrome", "aerodrome", "aérogare", "aerogare", "adresse", "aéronef", "aeronef", "aéroport", "aeroport", "aérospatial", "aerospatial", "affaire", "affectation", "affectif", "affiche", "affichage", "affiliation", "affluence", "agencement", "agile", "agissement", "agrafeuse", "agrégation", "agregation", "agriculture", "allié", "allie", "alchimiste", "algorithme", "algothérapie", "algotherapie", "alignement", "alinéa", "alinea", "allée", "allee", "allégorie", "allegorie", "allergologue", "allocation", "almanach", "alpaga", "alphabet", "alternant", "amandier", "amandine", "aménagement", "amenagement", "amphithéâtre", "amphitheatre", "anaconda", "analyse", "ancre", "annonciation", "aqueduc", "arbre", "arcade", "archipel", "architecte", "archive", "archiviste", "arène", "arene", "arlequin", "armée", "armee", "armorier", "arrondissement", "artefact", "article", "assemblement", "assistance", "association", "associe", "associé", "assurance", "assureur", "astuce", "atelier", "atrium", "attelage", "atypique", "audience", "auto-école", "autoécole", "auto-ecole", "autoecole", "avancée", "avenir", "eau", "écart", "ecart", "e-book", "ebook", "écran", "ecran", "écrit", "ecrit", "écurie", "ecurie", "éducation", "education", "effet", "église", "eglise", "ehpad", "empire", "emploi", "enclos", "encodage", "engagement", "engin", "énigme", "enigme", "ennui", "enregistrement", "enseignement", "entente", "entraide", "entrainement", "entrée", "entree", "entretien", "enveloppe", "envers", "épreuve", "epreuve", "esg", "essence", "étale", "etale", "état", "etat", "étude", "etude", "éveil", "eveil", "excès", "exces", "excursion", "exercice", "exode", "expérience", "experience", "expert", "exposition", "expression", "habitat", "hall", "harmonie", "hauteur", "herbe", "héritier", "heritier", "heure", "histoire", "homme", "honneur", "hôpial", "hopital", "hôtel", "hotel", "humain", "hypothèse", "hypothese", "icône", "icone", "idéal", "ideal", "identite", "identité", "idole", "image", "illusion", "imitaton", "immobilier", "inclusion", "incubateur", "indice", "induction", "information", "innocence", "inscription", "insolite", "instance", "instrument", "interface", "intérieur", "interieur", "interprète", "interprete", "iut", "iufm", "observatoire", "objet", "océan", "ocean", "oeil", "oiseau", "onu", "opération", "operation", "opinion", "option", "orage", "ordinateur", "organe", "organisation", "oreille", "orientation", "orsec", "ours", "umts", "union", "unité", "unite", "urgence", "urgent", 'usine', "utilitaire", "utopie"]
determinant_du = ["bureau", "travail", "groupe", "pavillon", "cabinet", "ministère", "ministere", "grand", "université", "universite", "réseau", "reseau", "club", "fc", "football", "groupement", "concret", "quai", "studio", "forum", "festival", "quai", "département", "departement", "grand", "baladeur", "bailleur", "balai", "balcon", "balisage", "banquet", "barbecue", "barrage", "barreau", "basket", "bâteau", "bateau", "bataillon", "bénévole", "bénévolat", "bénédiction", "benevole", "benediction", "benevolat", "berceau", "berlingot", "bermuda", "bétisier", "betisier", "bijou", "bouquin", "bricolage", "bulletin", "business", "bivouac","cabanon", "cabaret", "cac", "cacaoyer", "cacaotier", "cachalot", "cachemire", "cachet", "cachot", "cadavre", "cadeau", "cadenas", "cadran", "cadre", "café", "cafe", "caféier", "cafeier", "cageot", "cagibi", "cahier", "cahiou", "caissier", "calcium", "calcul", "calendrier", "calepin", "calibre", "calibrage", "calice", "califat", "câlin", "calin", "calisson", "calligramme", "calme", "calque", "calvaire", "cambiste", "camelot", "camion", "camionnage", "camp", "camouflage", "campement", "camping", "campus", "canal", "candidat", "canif", "canon", "cantal", "canton", "canular", "cap", "capes", "capet","capital", "capteur", "capuchon", "car", "caractère", "caractere", "caramèle", "caravane", "carburateur", "cardinal", "carillon", "cariste", "carnettiste", "carrelage", "carrosse", "carrousel", "cartel", "cartographe", "carton", "casting", "catalogue", "cauchemard", "caveau", "cedex", "centre", "centuple", "cercle", "cerf", "challenge", "champagne", "champignon", "changement", "capeau", "chaperon", "chariot", "chargeur", "charismatique", "château", "chateau", "chauffagiste", "chauffage", "chauffeur", "chemin", "chéquier", "chequier", "cheval", "chevalier", "chevet", "chewing-gum", "chewinggum", "chic", "chien", "chiffon", "chimiste", "chocolat", "chu", "cirque", "circuit", "cercle", "classeur", "clerc", "climat", "clocher", "clown", "club", "cocher", "coffre", "cognac", "coin", "colibri", "collaborateur", "collège", "college", "colonel", "coloriage", "comble", "commandant", "commando", "commandement", "commencement", "commerce", "commercant", "commerçant", "compas", "compte", "concept", "concert", "concierge", "concours", "congé", "conge", "corps", "correspondant", "corridor", "cours", "cousin", "cp", "cpa", "cpf", "crédit", "credit", "crosse", "cross", "crs", "cse", "dao", "dat", "dea", "débat", "debat", "décret", "decret", "désert", "desert", "design", "dess", "dessous", "dessus", "détail", "detail", "deug", "deust", "devis", "devoir", "dg", "dîner", "diner", "dj", "doigt", "dom", "domino", "don", "dossier", "doux", "dragon", "droit", "duo", "feu", "fichier", "flan", "fleuriste", "flocon", "fort", "franc", "frère", "frere", "gang", "garagiste", "garde", "génie", "genie", "genre", "golf", "golfe", "groupe", "jardin", "jeu","jeune", "jour", "journal", "juge", "jury", "jus", "karma", "kiosque", "label", "lac", "laboratoire", "lacet", "laitage", "langage", "lapin", "lecteur", "lingot", "littoral", "livre", "local", "logement", "logo", "lp", "lustre", "lundi", "magasin", "magnolia", "mais", "maïs", "maitre", "mandat", "manga", "manteau", "manuel", "marais", "marathon", "marchand", "marché", "membre", "mensuel", "hebdomadaire", "métro", "metro", "meuble", "ministère", "ministere", "modèle", "modele", "module", "mof", "monde", "mooc", "mot", "muscle", "nerf", "neveu", "nez", "nid", "niveau", "nomade", "nord", "paf", "panier", "papier", "papillon", "paquet", "paramètre", "parametre", "parc", "parcours", "parlement", "parquet", "parti", "passage", "pc", "pdf", "pcv", "père", "pere", "permis", "peuple", "phare", "pilote", "plan", "pôle", "port", "pont", "portfolio", "post", "principe", "principal", "profil", "programme", "projet", "propos", "quotidien", "rappel", "rapport", "rayon", "rep", "repos", "résidence", "residence", "résolution", "resolution", "resto", "restaurant", "rice", "roman", "rse", "rsa", "sac", "saint", "samu", "saut", "sav", "script", "scribe", "secret", "secours", "sel", "service", "seuil", "signe", "soin", "sort", "sourire", "soutien", "spam", "spectre", "stade", "stick", "suv", "tableau", "talon", "temps", "tissu", "tir", "tombeau", "trait", "traité", "traite", "traiteur", "trader", "transfert", "travail", "tribunal", "triomphe", "troc", "trou", "valet", "vase", "vélo", "velo", "vent", "vin", "wagon", "web"]
determinant_de_la = ['mission', "maif", "galerie", "matmut", "région", "region", "monnaie","maison", "companie", "cci", "fiduciaire", "compagnie", "caisse", "protection", "chambre", "commune", "place", "sncf", "banque", "fédération", "federation", "cheminée", "cheminee", "balance", "balise", "balise", "banane", "banderolle", "bande", "bannière", "banniere", "banque", "barre", "bibliothèque", "bijouterie", "borne", "buanderie", "cabine", "cacahouète", "cacahouete", "cachette", "cachotterie", "cadence", "caf", "caféine", "cafeine", "cafétériat", "cafeteria", "cafetière", "cafetiere", "cage", "cagnotte", "cagoule", "caille", "caissette", "caisse", "caissière", "caissiere", "calculette", "calculatrice", "cale", "caline", "câline", "câlinerie", "calinerie", "calligraphie", "calotte", "calvitie", "camaraderie", "cambuse", "campagne", "candidate", "cantine", "CAO", "capacité", "capacite", "cape", "capitale", "capsule", "carafe", "carapace", "carcasse", "caricature", "carotte", "cartoucherie", "cascade", "caserne", "casquette", "casserole", "caste", "catégorie", "categorie", "cavalerie", "caverne", "cave", "ceinture", "ceinturon", "célébration", "celebration", "célèbre", "celebre", "cellule", "centaine", "centrale", "centrifuge", "certification", "cfao", "chaîne", "chaine", "chaise", "chambre", "chanson", "chapelle", "charge", "chasse", "chaudière", "chaudiere", "chaussée", "chaussee", "chemise", "chevelure", "chope", "ville", "mairie", "citadine", "citadelle", "cité", "cite", "classe", "clairière", "clairiere", "classe", "classification", "clause", "clinique", "clique", "cloche", "clôture", "cloture", "collection", "coiffure", "colline", "colonie", "colonne", "coloration", "com", "combinaison", "commande", "commedia", "communauté", "communaute", "commune", "communication", "compétition", "competition", "compote", "concurrence", "concentration", "convention", "corde", "correspondante", "correspondance", "cote", "couche", "couleur", "coupe", "cousine", "coutellerie", "cravate", "crème", "creme", "csg", "csp", "cuisine", "culture", "dame", "décharge", "decharge", "déclaration", "declaration", "dérive", "derive", "descente", "dgse", "dgsi", "diaspora", "dictée", "dictee", "diction", "dictionnaire", "diction", "discussion", "dissertation", "distribution", "famille", "fée", "fee", "femme", "ferme", "feuille", "fiche", "figure", "flamme", "fleur", "flotte", "fondue", "forge", "foule", "fourrure", "fraternté", "fraternite", "fruitière", "fruitiere", "galerie", "garantie", "garde-robe", "garderobe", "greffe", "grenouille", "jetée", "jetee", "justice", "lecture", "légende", "legende", "lettre", "libération", "liberation", "ligne", "limite", "lingerie", "loge", "loi", "lumière", "lumiere", "lutte", "machine", "magazine", "main", "marche", "marine", "marque", "mégapole", "mégalopole", "mémoire", "memoire", "mine", "mise", "mode", "monnaie", "nageoire", "nature", "neige", "niche", "noblesse", "nomenclature", "note", "page", "paire", "pancarte", "panoplie", "pao", "papeterie", "parabole", "parade", "parole", "part", "partie", "patte", "peche", "pêche", "perspective", "pièce", "piece", "piste", "place", "plateforme", "plate-forme", "porte", "pratique", "pratique", "présentation", "presentation", "presse", "preuve", "primaire", "principale", "prise", "profession", "province", "question", "queue", "recherche", "région", "region", "reprise", "réussite", "rfid", "rgpd", "roue", "route", "rue", "salade", "salle", "santé", "série", "sirène", "sirene", "société", "societe", "station", "structure", "suite", "supérette", "superette", "surface", "table", "tablée", "tablee", "tapisserie", "tête", "tete", "thèse", "these", "tirelire", "toile", "tournée", "tournee", "tranche", "transaction", "troupe", "tutoriel", "valeur", "valise", "vérité", "verite", "vérification", "verification", "vertu", "voile", "voie", "voix", "vpc", "zif", "zone"]
determinant_des = [mot + 's' for mot in determinant_de_l + determinant_du + determinant_de_la] + \
                  [mot + 'x' for mot in determinant_de_l + determinant_du + determinant_de_la] + ["travaux", ""]

# company_name = ["bureaux", "travaux", "grands", "groupes", "pavillons", "ministeres", "ministères", "réseaux", "reseaux", "groupement", "cheminées", "cheminees", 'mission', "maison", "companie", "cci", "fiduciaire", "compagnie", "caisse", "protection", "chambre", "commune", "place", "sncf", "banque", "fédération", "federation", "cheminée", "cheminee", "bureau", "travail", "groupe", "pavillon", "cabinet", "ministère", "ministere", "grand", "université", "universite", "réseau", "reseau", "club", "fc", "football", "groupement", "concret", "quai", "studio", "forum", "festival", "quai", "département", "departement", "grand", "institut", "agence", "atelier", "assurance", "association", "alliance", "etablissement", "établissement", "afnor", "essec", "appel", "orchestre", "académie", "academie", 'orchestre', "ensemble"]

company_name = determinant_des + determinant_de_la + determinant_de_l + determinant_du

def normalize_col(name):
    # Remove accents and special characters
    name = unicodedata.normalize('NFKD', str(name))
    name = name.encode('ASCII', 'ignore').decode('utf-8')
    # Standardize to lowercase with underscores
    return name.strip().lower().replace(' ', '_')

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
        denominations = [
            r'\bInc\b', r'\bSAS\b', r'\bSARL\b', r'\bLtd\b', r'\bLLC\b', r'\bCorp\b', r'\bGmbH\b', r'\bCo\b', r'\bPty\b', r'\bAG\b', r'\bFreelance\b'
        ]
        pattern = '|'.join(denominations)
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
        return cleaned_text
    else:
        return text

# Fonction pour supprimer le tiret et tout ce qui suit dans un texte
def remove_hyphen_and_following(text):
    if isinstance(text, str):
        return text.split('- ')[0]
    else:
        return text

# Fonction pour supprimer le pipe et tout ce qui suit dans un texte
def remove_pipe_and_following(text):
    if isinstance(text, str):
        return text.split('|')[0]
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

# Fonction pour détecter le type du premier mot dans une phrase (déterminant, nom propre, nom commun, autre)
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

# Fonction pour supprimer les parenthèses et leur contenu dans un texte
def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', text)

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
                    determinant = "d'"
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
    expected_leading_cols = ['civilite', 'firstname', 'suggestion_de_prenom', 'nom']
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
        # Standardize the column name to 'civilite'
        df.rename(columns={existing_civility_col: 'civilite'}, inplace=True)
    else:
        # Create the column if it doesn't exist
        df['civilite'] = None

    # Compter le nombre total de lignes où la civilité est "Monsieur"
    if 'civilite' in df.columns:
        # Compte le nombre de lignes où la civilité est "Monsieur"
        count_monsieur = max((df['civilite'] == 'Monsieur').sum(),1)
        print("Nombre total de lignes avec civilité 'Monsieur':", count_monsieur, "nombre total de lignes", total_rows)

        # Utilise tqdm pour afficher une barre de progression lors du chargement du fichier
        with tqdm(total=total_rows, desc="Chargement du fichier") as pbar_load:
            # Insère une colonne 'nom' vide au début du DataFrame si elle n'existe pas
            if 'nom' not in df.columns:
                df.insert(loc=0, column='nom', value=None)

            # Parcourt chaque ligne du DataFrame
            for index, row in df.iterrows():
                # Si la colonne 'nom' n'est pas vide et ne contient pas de point, ajoute la valeur à 'Suggestion de Prénom'
                if (not pd.isnull(row['nom'])) and '.' not in str(row['nom']):
                    df.at[index, 'Suggestion de Prénom'] = row['nom']
                
                # Si la colonne 'societe' est une chaîne de caractères
                if isinstance(row['societe'], str):
                        # Supprime les parenthèses et leur contenu de 'societe'
                        row_societe_cleaned = remove_parentheses(row['societe'])
                        # Détermine le préfixe et le déterminant
                        prefix, determinant = determiner_prefixe_pronom(row_societe_cleaned)
                        
                        # Met à jour le DataFrame avec la valeur nettoyée et traitée de 'societe'
                        df.at[index, 'societe'] = f"{row_societe_cleaned}"
                        if isinstance(row['chez'], str):
                            prefix, determinant = determiner_prefixe_pronom(row_societe_cleaned)
                            df.at[index, 'chez'] = f"{prefix} {determinant} "
                
                # Si la colonne 'firstName' est vide
                if pd.isnull(row['firstname']):
                    if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'civilite'] = "Monsieur"
                    else:
                        df.at[index, 'civilite'] = 'Madame'
                # Si la colonne 'civilite' est vide
                elif pd.isnull(row['civilite']):
                    first_names = row['firstname'].split()
                    first_name = first_names[0]
                    gender = detect_gender(first_name)
                    if gender in ["andy", "unknown", "error"]:
                        if len(first_names) > 1:
                            second_name = first_names[1]
                            gender = detect_gender(second_name)
                    if gender == "female" or gender == "mostly_female":
                        df.at[index, 'civilite'] = "Madame"
                    elif gender == "male" or gender=="mostly_male":
                        df.at[index, 'civilite'] = "Monsieur"
                    elif gender == "andy":
                        if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'civilite'] = "Monsieur"
                        else:
                            df.at[index, 'civilite'] = 'Madame'
                    elif gender == "unknown":
                        if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'civilite'] = "Monsieur"
                        else:
                            df.at[index, 'civilite'] = 'Madame'
                    else:
                        df.at[index, 'civilite'] = "Erreur"

                # Si la colonne 'email' est vide
                if pd.isnull(row['email']):
                    if pd.isnull(row['firstname']):
                        df_null_email.at[index, 'civilite'] = "Prénom non attribué"
                    else:
                        first_names = row['firstname'].split()
                        first_name = first_names[0]
                        gender = detect_gender(first_name)
                        if gender in ["andy", "unknown", "error"]:
                            if len(first_names) > 1:
                                second_name = first_names[1]
                                gender = detect_gender(second_name)
                        if gender == "female" or gender == "mostly_female":
                            df_null_email.at[index, 'civilite'] = "Madame"
                        elif gender == "male" or gender=="mostly_male":
                            df_null_email.at[index, 'civilite'] = "Monsieur"
                        elif gender == "andy":
                            if total_rows / count_monsieur >= 0.5:
                                df.at[index, 'civilite'] = "Monsieur"
                            else:
                                df.at[index, 'civilite'] = 'Madame'
                        elif gender == "unknown":
                            if count_monsieur / total_rows < 50:
                                df_null_email.at[index, 'civilite'] = "Madame"
                            else:
                                df_null_email.at[index, 'civilite'] = "Monsieur"
                        else:
                            df_null_email.at[index, 'civilite'] = "Erreur"
                
                # Vérifie si une partie du nom de famille correspond au nom de la société
                if isinstance(row['nom'], str) and isinstance(row['societe'], str):
                    last_name_parts = row['nom'].split()
                    for part in last_name_parts:
                        if part.lower() in row['societe'].lower():
                            df.at[index, 'Match Entreprise'] = 'Oui'
                            break
                    else:
                        df.at[index, 'Match Entreprise'] = 'Non'
                # Met à jour la barre de progression
                pbar_load.update(1)

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
