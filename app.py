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
import openpyxl

# Télécharger les ressources nécessaires pour NLTK
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
# base_path = os.path.abspath(os.path.dirname(__file__))
# model_directory = os.path.join(base_path, 'fr_core_news_sm')

model_directory = 'fr_core_news_md'

nlp = spacy.load(model_directory)

# Load the SpaCy model
# nlp = spacy.load(model_directory)


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
        # Si la lecture en tant que CSV échoue, essayer de lire en tant que fichier Excel
        file.seek(0)
        try:
            df = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            # Si les deux méthodes échouent, renvoyer une erreur
            return "Erreur de lecture du fichier : le format n'est ni CSV ni Excel: {str(e)}"
    else:
        file.seek(0)
    # Appliquer les différentes fonctions de nettoyage sur la colonne "Société"
    df['Société'] = df['Société'].apply(lambda x: create_space_in_societe(x, company_name))
    df['Société'] = df['Société'].replace(r'[\u4e00-\u9fff]+', 'Société', regex=True)
    df['Société'] = df['Société'].apply(remove_enterprise_term)
    df['Société'] = df['Société'].apply(remove_comma_and_following)
    df['Société'] = df['Société'].apply(remove_hyphen_and_following)
    df['Société'] = df['Société'].apply(remove_pipe_and_following)
    df['Société'] = df['Société'].apply(remove_twopoints_and_following)

    # Fonction pour détecter le genre d'un prénom
    def detect_gender(name):
        detector = gender_detector.Detector(case_sensitive=False)
        return detector.get_gender(name)

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
    # Vérifier si la colonne "Société" est vide ou NaN
        if not nom_entreprise or nom_entreprise.strip() == "" or nom_entreprise == "au sein de votre entreprise" or pd.isna(nom_entreprise):
            return "au sein", "de votre entreprise"

        tokens = word_tokenize(nom_entreprise.lower())
        prefixe = 'au sein'
        determinant = ""
        
        
        # print(determinant_des)
        
        try:
            if tokens[0][0].lower() == "l" and tokens[0][1] == "'":
                prefixe = "au sein"
                determinant = "de"
            elif tokens[0].lower() in determinant_de_l:
                prefixe = "au sein"
                determinant = "de l'"
            elif tokens[0].lower() in determinant_du: #"mannequin", "chancelier", "chineur", "cireur"
                prefixe = "au sein"
                determinant = "du"
            elif tokens[0].lower() in determinant_des or (tokens[0].lower()[:-1] in determinant_des and tokens[0].lower()[-1] in ['s', 'x']):
                prefixe = "au sein"
                determinant = "des"
            elif tokens[0].lower() in determinant_de_la:
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
            if tokens[0].lower() in ["umake", "owkin", "isocel", "isocel.", "isocel.leclerc", "leclerc", "carrefour", "géant", "geant", "imerys", "afept", "acteon", "valorem", "voxelis", "vatel", "yzar", "accenture", "aemsofts", "sii", "sll", "metapolis", "memoandco", "maincare", "fayat", "eove", "cybertek", "cultura", ""] and prefixe == "au sein":
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
    if 'Email' in df.columns:
        df_null_email = df[df['Email'].isnull()]

    if 'Suggestion de Prénom' not in df.columns:
        df['Suggestion de Prénom'] = ""
    
    if 'chez' not in df.columns:
        df['chez'] = ""
        columns_order = [col for col in df.columns if col not in ['chez', 'Société']]

    columns_order.insert(4, 'chez')
    columns_order.insert(5, 'Société')

    df = df[columns_order]

    total_rows = len(df)

    # Compter le nombre total de lignes où la civilité est "Monsieur"
    if 'Civilité' in df.columns:
        # Compte le nombre de lignes où la civilité est "Monsieur"
        count_monsieur = (df['Civilité'] == 'Monsieur').sum()
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
                
                # Si la colonne 'Société' est une chaîne de caractères
                if isinstance(row['Société'], str):
                        # Supprime les parenthèses et leur contenu de 'Société'
                        row_societe_cleaned = remove_parentheses(row['Société'])
                        
                        # Détermine le préfixe et le déterminant
                        prefix, determinant = determiner_prefixe_pronom(row_societe_cleaned)
                        
                        # Met à jour le DataFrame avec la valeur nettoyée et traitée de 'Société'
                        df.at[index, 'Société'] = f"{row_societe_cleaned}"
                        if isinstance(row['chez'], str):
                            prefix, determinant = determiner_prefixe_pronom(row_societe_cleaned)
                            df.at[index, 'chez'] = f"{prefix} {determinant} "
                
                # Si la colonne 'firstName' est vide
                if pd.isnull(row['firstName']):
                    if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'Civilité'] = "Monsieur"
                    else:
                        df.at[index, 'Civilité'] = 'Madame'
                # Si la colonne 'Civilité' est vide
                elif pd.isnull(row['Civilité']):
                    first_names = row['firstName'].split()
                    first_name = first_names[0]
                    gender = detect_gender(first_name)
                    if gender in ["andy", "unknown", "error"]:
                        if len(first_names) > 1:
                            second_name = first_names[1]
                            gender = detect_gender(second_name)
                    if gender == "female" or gender == "mostly_female":
                        df.at[index, 'Civilité'] = "Madame"
                    elif gender == "male" or gender=="mostly_male":
                        df.at[index, 'Civilité'] = "Monsieur"
                    elif gender == "andy":
                        if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'Civilité'] = "Monsieur"
                        else:
                            df.at[index, 'Civilité'] = 'Madame'
                    elif gender == "unknown":
                        if total_rows / count_monsieur >= 0.5:
                            df.at[index, 'Civilité'] = "Monsieur"
                        else:
                            df.at[index, 'Civilité'] = 'Madame'
                    else:
                        df.at[index, 'Civilité'] = "Erreur"

                # Si la colonne 'Email' est vide
                if pd.isnull(row['Email']):
                    if pd.isnull(row['firstName']):
                        df_null_email.at[index, 'Civilité'] = "Prénom non attribué"
                    else:
                        first_names = row['firstName'].split()
                        first_name = first_names[0]
                        gender = detect_gender(first_name)
                        if gender in ["andy", "unknown", "error"]:
                            if len(first_names) > 1:
                                second_name = first_names[1]
                                gender = detect_gender(second_name)
                        if gender == "female" or gender == "mostly_female":
                            df_null_email.at[index, 'Civilité'] = "Madame"
                        elif gender == "male" or gender=="mostly_male":
                            df_null_email.at[index, 'Civilité'] = "Monsieur"
                        elif gender == "andy":
                            if total_rows / count_monsieur >= 0.5:
                                df.at[index, 'Civilité'] = "Monsieur"
                            else:
                                df.at[index, 'Civilité'] = 'Madame'
                        elif gender == "unknown":
                            if count_monsieur / total_rows < 50:
                                df_null_email.at[index, 'Civilité'] = "Madame"
                            else:
                                df_null_email.at[index, 'Civilité'] = "Monsieur"
                        else:
                            df_null_email.at[index, 'Civilité'] = "Erreur"
                
                # Vérifie si une partie du nom de famille correspond au nom de la société
                if isinstance(row['nom'], str) and isinstance(row['Société'], str):
                    last_name_parts = row['nom'].split()
                    for part in last_name_parts:
                        if part.lower() in row['Société'].lower():
                            df.at[index, 'Match Entreprise'] = 'Oui'
                            break
                    else:
                        df.at[index, 'Match Entreprise'] = 'Non'
                # Met à jour la barre de progression
                pbar_load.update(1)

    # df_combined = pd.concat([df, df_null_email], ignore_index=True)

    # df_combined = df_combined.dropna(subset=['Email'])

    print(df)

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
        df.dropna(subset=['Email']).to_excel(writer, sheet_name='Data', index=False)

        # Écrit le DataFrame avec les emails manquants dans la deuxième feuille, s'il n'est pas vide
        if not df_null_email.empty:
            df_null_email.to_excel(writer, sheet_name='Null-Email', index=False)
    
    # Supprime les lignes avec des emails manquants du DataFrame
    if 'Email' in df.columns:
        df.dropna(subset=['Email'], inplace=True)
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
