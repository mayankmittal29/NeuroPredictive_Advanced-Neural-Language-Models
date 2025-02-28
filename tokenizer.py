import re
import string
import sys
import os
class Tokenizer:
    def __init__(self, corpus):
        self.corpus = corpus
        self.preprocess_corpus()
        self.sen_tokenize = re.compile(r'(?<=[.!?]")[\s]|(?<=[.!?])[\s]')
        self.word_and_punct_tokenize = re.compile(r'''(?:[A-Z]\.)+|\w+(?:-\w+)*|\w+(?:'\w+)?|\.\.\.|(?:Mr|Mrs|Dr|Ms)\.|\w+|[^\w\s]|'\w+''')
        self.num_tokenize = re.compile(r'\b\d+(\.\d+)?\b')
        self.mail_tokenize = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_tokenize = re.compile(r'(http[s]?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.hash_tokenize = re.compile(r'#[\w\-]+')
        self.mention_tokenize = re.compile(r'@[\w\-]+')
        self.money_tokenize = re.compile(r'\b\d+(\.\d+)?\s?\$|\$\s?\d+(\.\d+)?\b')
        self.percent_tokenize = re.compile(r'\b\d+(\.\d+)?\%')
        self.age_tokenize = re.compile(r'\b\d{1,3}(?:\s|-)?(?:year(?:s)?\s?-?\s?old)\b')
        self.time_tokenize = re.compile(r'\b\d{1,2}:\d{2}\s?(?:AM|PM)?\b|\b(?:morning|afternoon|evening|night)\b', flags=re.IGNORECASE)
        self.date_tokenize = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2,4}|\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{2,4})\b')

    def preprocess_corpus(self):
        self.corpus = self.corpus.lower()
        self.corpus = re.sub(r'\s+', ' ', self.corpus)
        alphanumeric_tokenize = re.compile(r'(\d+|\D+)')
        self.corpus = ' '.join([' '.join(alphanumeric_tokenize.findall(word)) if alphanumeric_tokenize.match(word) and '.' not in word else word for word in self.corpus.split()])
        self.corpus = self.corpus.replace("_", "")
        self.corpus = re.sub(r'\b(mr|mrs|ms|dr|prof|rev|rd|st|gen|rep|sen|sr|jr)\.', r'\1', self.corpus)

    def replace_entities(self, text):
        text = re.sub(self.url_tokenize, '<URL>', text)
        text = re.sub(self.mail_tokenize, '<MAILID>', text)
        text = re.sub(self.date_tokenize, '<DATE>', text)
        text = re.sub(self.time_tokenize, '<TIME>', text)
        text = re.sub(self.age_tokenize, '<AGE>', text)
        text = re.sub(self.percent_tokenize, '<PERCENTAGE>', text)
        text = re.sub(self.money_tokenize, '<CURRENCY>', text)
        text = re.sub(self.num_tokenize, '<NUM>', text)
        text = re.sub(self.mention_tokenize, '<MENTION>', text)
        text = re.sub(self.hash_tokenize, '<HASHTAG>', text)
        return text
    
    def tokenize_sentence(self, text):
        sentences = re.split(self.sen_tokenize, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

    def tokenize_word(self, sentence):
        words_and_punctuations = []
        
        # Keep special tokens and words, but remove standalone punctuations
        tokens = re.findall(r'<URL>|<MAILID>|<DATE>|<TIME>|<AGE>|<PERCENTAGE>|<CURRENCY>|<NUM>|<MENTION>|<HASHTAG>|' + self.word_and_punct_tokenize.pattern, sentence)
        
        for token in tokens:
            # Remove pure punctuation tokens (excluding predefined tags)
            if token.isalnum() or token.startswith("<") and token.endswith(">"):
                words_and_punctuations.append(token)
        
        return words_and_punctuations

    def tokenize(self):
        self.corpus = self.replace_entities(self.corpus)
        sentences = self.tokenize_sentence(self.corpus)
        tokenized_corpus = [self.tokenize_word(sentence) for sentence in sentences]
        return tokenized_corpus

def tokenize_sentences(text):
    # sentences = re.split(
    #     r'(?<=[.!?])"? | (?<=[.!?])\s+(?=[A-Z])', text.strip())

    patterns = {
        "percentage": r"\b\d+(\.\d+)?%",  # Match percentages like 45%
        "url": r"(https?://\S+|www\.\S+)",  # Match URLs
        # Match emails
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "hashtag": r"#\w+",  # Match hashtags
        "mention": r"@\w+",  # Match mentions
        "age": r"\b\d{1,3}\s?(years old|yo|yrs)\b",  # Match age patterns
        # Match times like 12:30 PM
        "time": r"\b\d{1,2}:\d{2} ?(AM|PM|am|pm)?\b",
        "number": r"\b\d+(\.\d+)?\b",  # Match any number
        "date": r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"  # Match dates like 15th October
    }

    # Placeholders for substitution
    placeholders = {
        "url": "<URL>",
        "email": "<EMAIL>",
        "hashtag": "<HASHTAG>",
        "mention": "<MENTION>",
        "percentage": "<PERCENTAGE>",
        "age": "<AGE>",
        "time": "<TIME>",
        "number": "<NUMBER>",
        "date": "<DATE>"
    }

    for key, pattern in patterns.items():
        text = re.sub(pattern, placeholders[key], text)
    
    
    # Split text into tokens, keeping punctuation separate
    tokens = re.findall(r'\b\w+\b|[.,!?;(){}\[\]":\'/-]', text)  # Match words and individual punctuation
    temp_placeholders=['URL', 'EMAIL', 'HASHTAG', 'MENTION','PERCENTAGE','AGE','TIME','NUMBER', 'DATE']
    
    # Clean up extra spaces around the tokens
    tokens = [token if token not in temp_placeholders else '<'+token+'>' for token in tokens]
    
    return tokens

def tokenize_text(corpus):
        # Remove quotation marks before splitting into sentences
        corpus = re.sub(r'[\'"]', '', corpus)
        corpus = re.sub(r'[\'_]', '', corpus)
        corpus = re.sub(r'[\'-]', '', corpus)
        

        # Split into sentences considering punctuation marks and the case of the next letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', corpus.strip())
        
        tokenized_sentences = [tokenize_sentences(sentence) for sentence in sentences]
        return tokenized_sentences
    
    
def read_corpus(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    with open("output.txt", 'w', encoding='utf-8') as file:
        file.write(text)

    return tokenize_text(text)


def main():
    # sentences = read_corpus("./pride_and_prejudice.txt")
    # for i in range(30):
    #     print (sentences[i])
    # print()
    print("Welcome to the Toknizer:")
    while True:
        text = input("\nEnter text to tokenize (or 'quit' to exit): ")
        if text.lower() == 'quit':
            print("Thank you for using tokenizer bye!")
            break

        tokens = tokenize_text(text)
        print("\nTokenized text:")
        print(tokens)

if __name__ == "__main__":
    main()
