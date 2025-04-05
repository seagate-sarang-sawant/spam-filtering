import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_html_tags(text):
    from bs4 import BeautifulSoup
    return BeautifulSoup(text, "html.parser").get_text()

def remove_urls(text):
    import re
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_punctuation(text):
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_special_characters(text):
    import re
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

def remove_numeric(text):
    return ''.join([i for i in text if not i.isdigit()])

def remove_non_alphanumeric(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def replace_chat_words(text):
    chat_words = {
        "y": "you",
        "r": "are",
        "b4": "before",
        "gr8": "great",
        "l8r": "later",
        "pls": "please",
        "thx": "thanks",
        "afaiK": "As Far As I Know",
        "afk": "Away From Keyboard",
        "asap": "As Soon As Possible",
        "atk": "At The Keyboard",
        "atm": "At The Moment",
        "a3": "Anytime, Anywhere, Anyplace",
        "bak": "Back At Keyboard",
        "bbl": "Be Back Later",
        "bbs": "Be Back Soon",
        "bfn": "Bye For Now",
        "b4n": "Bye For Now",
        "brb": "Be Right Back",
        "brt": "Be Right There",
        "btw": "By The Way",
        "b4": "Before",
        "b4n": "Bye For Now",
        "cu": "See You",
        "cul8r": "See You Later",
        "cya": "See You",
        "faq": "Frequently Asked Questions",
        "fc": "Fingers Crossed",
        "fwiw": "For What It's Worth",
        "fyi": "For Your Information",
        "gal": "Get A Life",
        "gg": "Good Game",
        "gn": "Good Night",
        "gmta": "Great Minds Think Alike",
        "gr8": "Great!",
        "g9": "Genius",
        "ic": "I See",
        "icq": "I Seek you (also a chat program)",
        "ilu": "ILU: I Love You",
        "imho": "In My Honest/Humble Opinion",
        "imo": "In My Opinion",
        "iow": "In Other Words",
        "irl": "In Real Life",
        "kiss": "Keep It Simple, Stupid",
        "ldr": "Long Distance Relationship",
        "lmao": "Laugh My A.. Off",
        "lol": "Laughing Out Loud",
        "ltns": "Long Time No See",
        "l8r": "Later",
        "mte": "My Thoughts Exactly",
        "m8": "Mate",
        "nrn": "No Reply Necessary",
        "oic": "Oh I See",
        "pita": "Pain In The A..",
        "prt": "Party",
        "prw": "Parents Are Watching",
        "qpsa?": "Que Pasa?",
        "rofl": "Rolling On The Floor Laughing",
        "roflol": "Rolling On The Floor Laughing Out Loud",
        "rotflmao": "Rolling On The Floor Laughing My A.. Off",
        "sk8": "Skate",
        "stats": "Your sex and age",
        "asl": "Age, Sex, Location",
        "thx": "Thank You",
        "ttfn": "Ta-Ta For Now!",
        "ttyl": "Talk To You Later",
        "u": "You",
        "u2": "You Too",
        "u4e": "Yours For Ever",
        "wb": "Welcome Back",
        "wtf": "What The F...",
        "wtg": "Way To Go!",
        "wuf": "Where Are You From?",
        "w8": "Wait...",
        "7k": "Sick:-D Laugher",
        "tfw": "That feeling when",
        "mfw": "My face when",
        "mrw": "My reaction when",
        "ifyp": "I feel your pain",
        "tntl": "Trying not to laugh",
        "jk": "Just kidding",
        "idc": "I don't care",
        "ily": "I love you",
        "imu": "I miss you",
        "adih": "Another day in hell",
        "zzz": "Sleeping, bored, tired",
        "wywh": "Wish you were here",
        "time": "Tears in my eyes",
        "bae": "Before anyone else",
        "fimH": "Forever in my heart",
        "bsaaW": "Big smile and a wink",
        "bwl": "Bursting with laughter",
        "bff": "Best friends forever",
        "csl": "Can't stop laughing"
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word.upper() in chat_words:
            words[i] = chat_words[word.upper()]
    return ' '.join(words)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def remove_emojis(text):
    import re
    return re.sub(r'[^\x00-\x7F]+', '', text)

def stem_words(text):
    from nltk.stem import PorterStemmer
    porter_stemmer = PorterStemmer()
    return ' '.join([porter_stemmer.stem(word) for word in text.split()])

def label_encode(df, column):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df, label_encoder

def lowercase_text(text):
    return text.lower()

def strip_whitespace(text):
    return text.strip()

def porter_stemmer():
    return PorterStemmer()
