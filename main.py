import pandas as pd
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt')

path = "Sample.xlsx"
df = pd.read_excel(path)

def tokenize(text):
    return nltk.word_tokenize(text)

def calculate_bleu(reference, candidate):
    bleu_score = sacrebleu.corpus_bleu([candidate], [[reference]]).score
    return bleu_score / 100  # We normalize the BLEU score

def calculate_meteor(reference, candidate):
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    return meteor_score([reference_tokens], candidate_tokens)

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure

df['BLEU_Google'] = df.apply(lambda row: calculate_bleu(row['French'], row['Google']), axis=1)
df['METEOR_Google'] = df.apply(lambda row: calculate_meteor(row['French'], row['Google']), axis=1)
df['ROUGE_Google'] = df.apply(lambda row: calculate_rouge(row['French'], row['Google']), axis=1)

df['BLEU_Pons'] = df.apply(lambda row: calculate_bleu(row['French'], row['Pons']), axis=1)
df['METEOR_Pons'] = df.apply(lambda row: calculate_meteor(row['French'], row['Pons']), axis=1)
df['ROUGE_Pons'] = df.apply(lambda row: calculate_rouge(row['French'], row['Pons']), axis=1)

df['BLEU_DeepL'] = df.apply(lambda row: calculate_bleu(row['French'], row['DeepL']), axis=1)
df['METEOR_DeepL'] = df.apply(lambda row: calculate_meteor(row['French'], row['DeepL']), axis=1)
df['ROUGE_DeepL'] = df.apply(lambda row: calculate_rouge(row['French'], row['DeepL']), axis=1)

mean_bleu_google = df['BLEU_Google'].mean()
mean_meteor_google = df['METEOR_Google'].mean()
mean_rouge_google = df['ROUGE_Google'].mean()

mean_bleu_pons = df['BLEU_Pons'].mean()
mean_meteor_pons = df['METEOR_Pons'].mean()
mean_rouge_pons = df['ROUGE_Pons'].mean()

mean_bleu_deepl = df['BLEU_DeepL'].mean()
mean_meteor_deepl = df['METEOR_DeepL'].mean()
mean_rouge_deepl = df['ROUGE_DeepL'].mean()

print(f"Moyennes des scores pour Google: BLEU={mean_bleu_google}, METEOR={mean_meteor_google}, ROUGE={mean_rouge_google}")
print(f"Moyennes des scores pour Pons: BLEU={mean_bleu_pons}, METEOR={mean_meteor_pons}, ROUGE={mean_rouge_pons}")
print(f"Moyennes des scores pour DeepL: BLEU={mean_bleu_deepl}, METEOR={mean_meteor_deepl}, ROUGE={mean_rouge_deepl}")

# Display results
# print(df[['Sample', 'BLEU_Google', 'METEOR_Google', 'ROUGE_Google',
#           'BLEU_Pons', 'METEOR_Pons', 'ROUGE_Pons',
#           'BLEU_DeepL', 'METEOR_DeepL', 'ROUGE_DeepL']])
print(df[['BLEU_DeepL', 'METEOR_DeepL', 'ROUGE_DeepL']])

df.to_csv('translated_texts_all_metrics.csv', index=False, encoding='utf-8')

print("Les scores des différentes métriques ont été calculés et sauvegardés dans 'translated_texts_all_metrics.csv'.")
