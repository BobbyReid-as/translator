"""
BLEU Evaluation Script
Scores MarianMT translations against reference translations using SacreBLEU.
Outputs sacrebleu_scores.csv ready for submission.

Usage:
    uv add sacrebleu
    uv run python evaluate.py

After running:
    1. Open sacrebleu_scores.csv
    2. Fill in the google_translate column manually
    3. Run again with --with-google to score Google Translate too
"""

import csv
import sys
from main import translate

try:
    from sacrebleu.metrics import BLEU
except ImportError:
    print("Please run: uv add sacrebleu")
    sys.exit(1)

# ─────────────────────────────────────────────
# Test sentences: 20 per language pair → English
# Each entry: (source_text, reference_english_translation, language_pair)
# ─────────────────────────────────────────────

TEST_DATA = {

    "FR→EN": [
        ("Le soleil se lève à l'est.",                          "The sun rises in the east."),
        ("J'aime beaucoup la musique classique.",                "I really like classical music."),
        ("Elle travaille dans un hôpital depuis dix ans.",       "She has worked in a hospital for ten years."),
        ("Nous partirons en vacances la semaine prochaine.",     "We will go on holiday next week."),
        ("Les enfants jouent dans le parc.",                     "The children are playing in the park."),
        ("Il fait très froid en hiver ici.",                     "It is very cold in winter here."),
        ("Pouvez-vous m'indiquer le chemin pour la gare ?",      "Can you show me the way to the train station?"),
        ("Cette ville est connue pour ses musées.",              "This city is known for its museums."),
        ("Je n'ai pas encore mangé ce matin.",                   "I have not eaten yet this morning."),
        ("Le livre que tu m'as prêté était excellent.",          "The book you lent me was excellent."),
        ("Ils ont construit une nouvelle école dans le quartier.","They built a new school in the neighbourhood."),
        ("Ma mère prépare toujours de bons repas.",              "My mother always prepares good meals."),
        ("La réunion a été reportée à demain.",                  "The meeting has been postponed to tomorrow."),
        ("Il apprend le piano depuis l'âge de cinq ans.",        "He has been learning the piano since the age of five."),
        ("Les fleurs s'épanouissent au printemps.",              "The flowers bloom in spring."),
        ("Nous avons regardé un film hier soir.",                "We watched a film last night."),
        ("Le train est arrivé avec vingt minutes de retard.",    "The train arrived twenty minutes late."),
        ("Elle parle couramment trois langues.",                 "She speaks three languages fluently."),
        ("Ce restaurant sert d'excellents plats de fruits de mer.","This restaurant serves excellent seafood dishes."),
        ("Les nouvelles technologies changent notre façon de vivre.","New technologies are changing the way we live."),
    ],

    "DE→EN": [
        ("Die Sonne scheint heute sehr hell.",                   "The sun is shining very brightly today."),
        ("Ich lerne seit drei Jahren Deutsch.",                  "I have been learning German for three years."),
        ("Das Wetter wird morgen besser sein.",                  "The weather will be better tomorrow."),
        ("Er arbeitet in einer großen Firma in Berlin.",         "He works in a large company in Berlin."),
        ("Wir haben gestern Abend ein Konzert besucht.",         "We attended a concert yesterday evening."),
        ("Die Kinder spielen gerne im Garten.",                  "The children like to play in the garden."),
        ("Kannst du mir bitte helfen?",                          "Can you please help me?"),
        ("Das Buch liegt auf dem Tisch.",                        "The book is on the table."),
        ("Sie hat ihr Studium mit Auszeichnung abgeschlossen.",  "She completed her studies with distinction."),
        ("Im Winter schneit es oft in den Bergen.",              "In winter it often snows in the mountains."),
        ("Der Zug fährt um acht Uhr ab.",                        "The train departs at eight o'clock."),
        ("Ich esse jeden Morgen Brot zum Frühstück.",            "I eat bread for breakfast every morning."),
        ("Die Stadt hat viele schöne alte Gebäude.",             "The city has many beautiful old buildings."),
        ("Er hat gestern seinen Geburtstag gefeiert.",           "He celebrated his birthday yesterday."),
        ("Wir müssen früh aufstehen, um den Bus zu erwischen.",  "We have to get up early to catch the bus."),
        ("Das Restaurant ist für seine Küche bekannt.",          "The restaurant is known for its cuisine."),
        ("Sie kauft immer frisches Gemüse auf dem Markt.",       "She always buys fresh vegetables at the market."),
        ("Der Film war sehr spannend und gut gemacht.",          "The film was very exciting and well made."),
        ("Meine Schwester wohnt in einer anderen Stadt.",        "My sister lives in a different city."),
        ("Technologie verändert unsere Gesellschaft grundlegend.","Technology is fundamentally changing our society."),
    ],

    "ES→EN": [
        ("El cielo está despejado esta mañana.",                 "The sky is clear this morning."),
        ("Me gusta mucho leer libros de historia.",              "I really like reading history books."),
        ("Ella trabaja como médica en un hospital público.",     "She works as a doctor in a public hospital."),
        ("Vamos a la playa este fin de semana.",                 "We are going to the beach this weekend."),
        ("Los niños están haciendo los deberes.",                "The children are doing their homework."),
        ("¿Puedes ayudarme a encontrar la calle principal?",     "Can you help me find the main street?"),
        ("Este año el invierno ha sido muy frío.",               "This year winter has been very cold."),
        ("El tren llega a las nueve de la mañana.",              "The train arrives at nine in the morning."),
        ("Mi hermano aprendió a cocinar durante la pandemia.",   "My brother learned to cook during the pandemic."),
        ("La reunión empezará a las tres de la tarde.",          "The meeting will start at three in the afternoon."),
        ("Ella habla español, inglés y francés con fluidez.",    "She speaks Spanish, English and French fluently."),
        ("El museo cierra los lunes.",                           "The museum is closed on Mondays."),
        ("Hemos vivido en esta ciudad durante cinco años.",      "We have lived in this city for five years."),
        ("El partido de fútbol fue muy emocionante.",            "The football match was very exciting."),
        ("Los precios han subido mucho este año.",               "Prices have risen a lot this year."),
        ("Me despierto siempre a las siete de la mañana.",       "I always wake up at seven in the morning."),
        ("Esta película ganó varios premios internacionales.",   "This film won several international awards."),
        ("El jardín está lleno de flores en primavera.",         "The garden is full of flowers in spring."),
        ("Compré un nuevo ordenador para trabajar desde casa.",  "I bought a new computer to work from home."),
        ("La tecnología avanza más rápido que nunca.",           "Technology is advancing faster than ever."),
    ],

    "IT→EN": [
        ("Il sole tramonta ad ovest ogni sera.",                 "The sun sets in the west every evening."),
        ("Mi piace molto la cucina italiana.",                   "I really like Italian cuisine."),
        ("Lei lavora come insegnante da vent'anni.",             "She has worked as a teacher for twenty years."),
        ("Andiamo al cinema stasera?",                          "Shall we go to the cinema tonight?"),
        ("I bambini giocano a calcio nel cortile.",              "The children are playing football in the yard."),
        ("Fa molto caldo d'estate in questa città.",             "It is very hot in summer in this city."),
        ("Puoi dirmi dov'è la stazione dei treni?",             "Can you tell me where the train station is?"),
        ("Questo museo è famoso in tutto il mondo.",             "This museum is famous all over the world."),
        ("Non ho ancora fatto colazione stamattina.",            "I have not had breakfast yet this morning."),
        ("Il libro che mi hai consigliato era bellissimo.",      "The book you recommended to me was wonderful."),
        ("Hanno aperto un nuovo ristorante in centro.",          "They have opened a new restaurant in the city centre."),
        ("Mia nonna cucina sempre ottimi piatti.",               "My grandmother always cooks excellent dishes."),
        ("La riunione è stata spostata a domani pomeriggio.",    "The meeting has been moved to tomorrow afternoon."),
        ("Studia violino dall'età di sei anni.",                 "He has been studying the violin since the age of six."),
        ("I fiori sbocciano in primavera.",                      "The flowers bloom in spring."),
        ("Abbiamo guardato una serie televisiva ieri sera.",     "We watched a television series last night."),
        ("Il treno era in ritardo di mezz'ora.",                 "The train was half an hour late."),
        ("Parla tre lingue in modo fluente.",                    "She speaks three languages fluently."),
        ("Questo locale serve ottimi piatti di pesce.",          "This place serves excellent fish dishes."),
        ("Le nuove tecnologie stanno cambiando il mondo del lavoro.","New technologies are changing the world of work."),
    ],

    "PT→EN": [
        ("O sol nasce a leste todos os dias.",                   "The sun rises in the east every day."),
        ("Eu gosto muito de ouvir música.",                      "I really like listening to music."),
        ("Ela trabalha num hospital há quinze anos.",             "She has worked in a hospital for fifteen years."),
        ("Vamos viajar para o Brasil nas próximas férias.",      "We are going to travel to Brazil on the next holiday."),
        ("As crianças estão brincando no jardim.",               "The children are playing in the garden."),
        ("Faz muito frio aqui durante o inverno.",               "It is very cold here during winter."),
        ("Pode me indicar onde fica a estação de trem?",         "Can you tell me where the train station is?"),
        ("Esta cidade é famosa pelos seus monumentos históricos.","This city is famous for its historical monuments."),
        ("Ainda não tomei café da manhã hoje.",                  "I have not had breakfast yet today."),
        ("O livro que você me emprestou foi incrível.",          "The book you lent me was incredible."),
        ("Eles construíram uma nova escola no bairro.",          "They built a new school in the neighbourhood."),
        ("Minha mãe sempre prepara refeições deliciosas.",       "My mother always prepares delicious meals."),
        ("A reunião foi adiada para amanhã de manhã.",           "The meeting was postponed to tomorrow morning."),
        ("Ele aprende guitarra desde os oito anos de idade.",    "He has been learning guitar since the age of eight."),
        ("As flores desabrocham na primavera.",                  "The flowers bloom in spring."),
        ("Assistimos a um documentário ontem à noite.",          "We watched a documentary last night."),
        ("O trem chegou com trinta minutos de atraso.",          "The train arrived thirty minutes late."),
        ("Ela fala quatro idiomas com fluência.",                "She speaks four languages fluently."),
        ("Este restaurante é conhecido pelos seus pratos típicos.","This restaurant is known for its traditional dishes."),
        ("A tecnologia está transformando a maneira como vivemos.","Technology is transforming the way we live."),
    ],
}

# ─────────────────────────────────────────────
# Source language codes for each pair
# ─────────────────────────────────────────────

PAIR_SRC = {
    "FR→EN": "fr",
    "DE→EN": "de",
    "ES→EN": "es",
    "IT→EN": "it",
    "PT→EN": "pt",
}

# ─────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────

def bleu_score(hypotheses: list[str], references: list[str]) -> float:
    """Return corpus-level BLEU score (0–100) for a list of sentence pairs."""
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(hypotheses, [references])
    return round(result.score, 2)


def sentence_bleu(hypothesis: str, reference: str) -> float:
    """Return sentence-level BLEU score (0–100)."""
    bleu = BLEU(effective_order=True)
    result = bleu.sentence_score(hypothesis, [reference])
    return round(result.score, 2)


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────

def run_evaluation(with_google: bool = False, google_csv: str = "sacrebleu_scores.csv"):
    rows = []

    # If re-scoring with Google translations, load existing CSV
    google_lookup = {}
    if with_google:
        try:
            with open(google_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["language_pair"], row["sentence_id"])
                    google_lookup[key] = row.get("google_translate", "")
        except FileNotFoundError:
            print(f"Could not find {google_csv} — run without --with-google first.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("SACREBLEU EVALUATION")
    print("=" * 65)

    corpus_summary = []

    for pair, sentences in TEST_DATA.items():
        src_lang = PAIR_SRC[pair]
        hypotheses = []
        references  = []
        google_hyps = []

        print(f"\n[{pair}] Translating {len(sentences)} sentences...")

        for i, (source, reference) in enumerate(sentences, start=1):
            hypothesis = translate(source, "en", auto_detect=False, source_lang=src_lang)

            # Clean error outputs
            if hypothesis.startswith("❌") or hypothesis.startswith("⚠️"):
                print(f"  ✗ Sentence {i} failed: {hypothesis}")
                hypothesis = ""

            sent_bleu = sentence_bleu(hypothesis, reference) if hypothesis else 0.0

            google_translation = ""
            google_sent_bleu   = ""
            if with_google:
                google_translation = google_lookup.get((pair, str(i)), "")
                if google_translation:
                    google_sent_bleu = sentence_bleu(google_translation, reference)

            rows.append({
                "language_pair":        pair,
                "sentence_id":          i,
                "source":               source,
                "reference":            reference,
                "marianmt_translation": hypothesis,
                "marianmt_bleu":        sent_bleu,
                "google_translate":     google_translation,
                "google_bleu":          google_sent_bleu,
            })

            hypotheses.append(hypothesis if hypothesis else " ")
            references.append(reference)
            if google_translation:
                google_hyps.append(google_translation)

            print(f"  [{i:02d}] BLEU {sent_bleu:5.1f}  |  {source[:45]}")

        corpus_bleu = bleu_score(hypotheses, references)
        google_corpus_bleu = bleu_score(google_hyps, references[:len(google_hyps)]) if google_hyps else ""

        corpus_summary.append({
            "language_pair":      pair,
            "marianmt_corpus_bleu": corpus_bleu,
            "google_corpus_bleu":   google_corpus_bleu,
        })

        print(f"  → Corpus BLEU ({pair}): {corpus_bleu}")
        if google_corpus_bleu:
            print(f"  → Google BLEU  ({pair}): {google_corpus_bleu}")

    # ── Write sentence-level CSV ──
    fieldnames = [
        "language_pair", "sentence_id", "source", "reference",
        "marianmt_translation", "marianmt_bleu",
        "google_translate", "google_bleu",
    ]
    with open("sacrebleu_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Write corpus summary CSV ──
    with open("sacrebleu_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language_pair", "marianmt_corpus_bleu", "google_corpus_bleu"])
        writer.writeheader()
        writer.writerows(corpus_summary)

    print("\n" + "=" * 65)
    print("CORPUS BLEU SUMMARY")
    print("=" * 65)
    for row in corpus_summary:
        g = f"  |  Google: {row['google_corpus_bleu']}" if row["google_corpus_bleu"] else ""
        print(f"  {row['language_pair']:8s}  MarianMT: {row['marianmt_corpus_bleu']:5.1f}{g}")

    print("\nFiles written:")
    print("  sacrebleu_scores.csv   ← fill in google_translate column, then re-run with --with-google")
    print("  sacrebleu_summary.csv  ← corpus-level scores per language pair")
    print("=" * 65)


if __name__ == "__main__":
    with_google = "--with-google" in sys.argv
    run_evaluation(with_google=with_google)
