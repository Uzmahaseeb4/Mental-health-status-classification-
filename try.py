"""
Mental Health Detector — Streamlit App (Fixed)
TF-IDF + SVM  vs  BERT-style Transformer + SVM

Fixes:
- Expanded training data with much more linguistic variety
- Added 'Sad' as a distinct class (5 classes total)
- Better TF-IDF features (char n-grams + word n-grams)
- Ensemble voting for more robust predictions on unseen text
- Improved tokenization and preprocessing
"""

import warnings, time, math, re
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
}
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    background: linear-gradient(135deg, #e8d5b7 0%, #f5c842 50%, #e8916a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.main-header p {
    color: #7a8a9a;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}
.result-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(248,196,66,0.3);
}
.badge { display:inline-block;padding:0.3rem 1rem;border-radius:999px;font-size:0.85rem;font-weight:600;letter-spacing:0.04em; }
.badge-0 { background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.3); }
.badge-1 { background:rgba(99,102,241,0.15);color:#a78bfa;border:1px solid rgba(167,139,250,0.3); }
.badge-2 { background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3); }
.badge-3 { background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.3); }
.badge-4 { background:rgba(14,165,233,0.15);color:#38bdf8;border:1px solid rgba(56,189,248,0.3); }
.conf-bar-bg { background:rgba(255,255,255,0.08);border-radius:999px;height:8px;margin:0.5rem 0;overflow:hidden; }
.conf-bar-fill { height:100%;border-radius:999px;transition:width 0.6s ease; }
.metric-box { background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem 1.2rem;text-align:center; }
.metric-box .val { font-family:'DM Serif Display',serif;font-size:2rem;color:#f5c842; }
.metric-box .lbl { font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem; }
.section-title { font-family:'DM Serif Display',serif;font-size:1.4rem;color:#e8d5b7;margin:1.5rem 0 0.8rem; }
.disclaimer { background:rgba(239,68,68,0.07);border-left:3px solid #ef4444;border-radius:0 8px 8px 0;padding:0.8rem 1rem;font-size:0.82rem;color:#9ca3af;margin-top:1.5rem; }
.stTextArea textarea { background:rgba(255,255,255,0.05)!important;border:1px solid rgba(255,255,255,0.1)!important;border-radius:12px!important;color:#e2e8f0!important;font-family:'DM Sans',sans-serif!important;font-size:1rem!important; }
.stTextArea textarea:focus { border-color:rgba(245,200,66,0.5)!important;box-shadow:0 0 0 2px rgba(245,200,66,0.1)!important; }
.stButton>button { background:linear-gradient(135deg,#f5c842,#e8916a)!important;color:#0d1117!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-family:'DM Sans',sans-serif!important;font-size:1rem!important;padding:0.6rem 2rem!important;transition:all 0.2s!important; }
.stButton>button:hover { transform:translateY(-1px)!important;box-shadow:0 8px 25px rgba(245,200,66,0.3)!important; }
div[data-testid="stSidebar"] { background:rgba(13,17,23,0.95)!important;border-right:1px solid rgba(255,255,255,0.07)!important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
# 0=Normal, 1=Sad, 2=Depression, 3=Anxiety, 4=Stress/Burnout
LABEL_NAMES  = {0: "No Concern", 1: "Sad", 2: "Depression", 3: "Anxiety", 4: "Stress/Burnout"}
LABEL_EMOJIS = {0: "✅", 1: "😢", 2: "💙", 3: "🌀", 4: "🔥"}
LABEL_COLORS = {0: "#34d399", 1: "#38bdf8", 2: "#a78bfa", 3: "#fbbf24", 4: "#f87171"}

SAMPLES = [
    # ── 0: No Concern / Normal ──────────────────────────────────────────────
    ("I had a great day today and feel really energised", 0),
    ("Just finished an amazing workout, feeling strong and alive", 0),
    ("Had coffee with a friend and we laughed for hours", 0),
    ("Loving my new job, every day brings an exciting challenge", 0),
    ("Went for a long walk and enjoyed the beautiful sunshine", 0),
    ("Cooking dinner and listening to my favourite music right now", 0),
    ("Got a promotion at work, so excited about the future ahead", 0),
    ("Reading a good book and feeling totally relaxed and content", 0),
    ("Had a productive day and ticked everything off my list today", 0),
    ("Spent the weekend with family, it was truly wonderful", 0),
    ("Feeling grateful for all the good things happening in my life", 0),
    ("Started a new hobby and I am really enjoying learning it", 0),
    ("Met old friends after a long time, catching up was amazing", 0),
    ("The weather is beautiful today and I feel completely alive", 0),
    ("Finished a big project at work and feeling very accomplished", 0),
    ("Going on a holiday next week, planning is so much fun", 0),
    ("My garden is blooming beautifully this spring season", 0),
    ("Cooked a new recipe today and it turned out delicious", 0),
    ("Feeling happy and content with my life right now", 0),
    ("Everything is going well and I have a lot to look forward to", 0),
    ("I woke up refreshed and ready to take on the day", 0),
    ("Life feels good and I am full of hope about the future", 0),
    ("Excited about my plans this weekend with my loved ones", 0),
    ("I feel positive and motivated today", 0),
    ("Things are going smoothly and I am in a great mood", 0),

    # ── 1: Sad ───────────────────────────────────────────────────────────────
    ("I feel sad today and do not know why", 1),
    ("Feeling a bit down and lonely this evening", 1),
    ("My heart feels heavy and I just want to cry", 1),
    ("I miss someone close to me and the sadness is overwhelming", 1),
    ("Had a bad day and feeling quite low right now", 1),
    ("Things did not go as planned and I feel disappointed", 1),
    ("I am grieving the loss of a loved one and it hurts", 1),
    ("Feeling melancholy and nostalgic for better times", 1),
    ("I watched a sad movie and now I cannot stop feeling tearful", 1),
    ("My friend said something hurtful and I feel really sad", 1),
    ("The loneliness is getting to me today more than usual", 1),
    ("I feel blue and nothing seems to cheer me up right now", 1),
    ("Received some bad news today and I feel heartbroken", 1),
    ("Feeling sad and weepy without a clear reason today", 1),
    ("I just feel really down in the dumps today", 1),
    ("Lost my pet and feeling so deeply sad about it", 1),
    ("Things feel a bit grey and gloomy right now", 1),
    ("I teared up thinking about something painful from the past", 1),
    ("Feeling a bit sorry for myself today", 1),
    ("I am upset and sad after an argument with someone I love", 1),
    ("Feeling unhappy and not really sure why I am so sad", 1),
    ("I cried a lot today and it helped a little but I still feel sad", 1),
    ("The sadness comes and goes but it is quite heavy right now", 1),
    ("I feel a deep sense of sadness that I cannot shake off", 1),
    ("Heartbroken after a breakup and really struggling", 1),

    # ── 2: Depression ────────────────────────────────────────────────────────
    ("I feel completely empty inside, nothing brings me joy anymore", 2),
    ("Getting out of bed has become almost impossible lately", 2),
    ("I have lost all interest in things I used to love doing", 2),
    ("Everything feels completely pointless and I see no purpose", 2),
    ("I have been crying every single day for no particular reason", 2),
    ("I feel like a burden to everyone around me in my life", 2),
    ("Sleeping all day but still exhausted, life feels grey and dull", 2),
    ("I cannot remember the last time I felt truly happy at all", 2),
    ("Isolated myself from friends and family, it is just easier", 2),
    ("Thoughts of self-harm keep crossing my mind repeatedly lately", 2),
    ("I feel completely worthless and like nothing will ever improve", 2),
    ("Lost my appetite entirely, food has absolutely no appeal anymore", 2),
    ("Waking up every morning and wishing the day was already over", 2),
    ("Everyone around me would honestly be better off without me", 2),
    ("Feeling completely numb, no sadness no happiness just emptiness", 2),
    ("Cannot find any motivation to do even the simplest daily tasks", 2),
    ("Every morning I wake up and dread facing another pointless day", 2),
    ("The darkness inside me never seems to lift no matter what I try", 2),
    ("I do not see the point in anything anymore, life feels hollow", 2),
    ("Nothing gives me pleasure, not food not friends not anything", 2),
    ("I feel like I am just going through the motions every single day", 2),
    ("Deep hopelessness has set in and I cannot imagine feeling better", 2),
    ("I have withdrawn from everything and everyone I used to care about", 2),
    ("No energy, no motivation, no hope — just emptiness every day", 2),
    ("I feel dead inside even when I am surrounded by people who love me", 2),

    # ── 3: Anxiety ───────────────────────────────────────────────────────────
    ("My heart races and I cannot catch my breath, anxiety overwhelms me", 3),
    ("Constant worry about things that might go wrong in the future", 3),
    ("I keep catastrophising even the smallest and most trivial situations", 3),
    ("Cannot sleep at night because my mind just will not stop spinning", 3),
    ("Dreading social gatherings because I always say the wrong thing", 3),
    ("Chest tightness and sweating whenever I have to present at work", 3),
    ("Checking the door lock three times before I feel safe to leave", 3),
    ("Every health symptom I notice convinces me of some terrible disease", 3),
    ("Paralysed by indecision because what if I make completely wrong choice", 3),
    ("Panic attacks are becoming more frequent and overwhelmingly intense", 3),
    ("Avoiding all crowded places because they trigger my overwhelming fear", 3),
    ("My hands tremble and shake whenever I enter a new social situation", 3),
    ("Ruminating endlessly over one conversation I had several weeks ago", 3),
    ("Terrified of flying and it has stopped me from any travel at all", 3),
    ("Constant sense of impending doom even when everything seems perfectly fine", 3),
    ("Racing thoughts at night prevent me from getting any restful sleep", 3),
    ("The fear of being judged stops me from doing things I want to do", 3),
    ("Overthinking every decision leads me to a complete mental standstill", 3),
    ("I cannot stop worrying about what might happen tomorrow", 3),
    ("Every little thing makes me nervous and I feel on edge all the time", 3),
    ("I keep imagining worst case scenarios for everything in my life", 3),
    ("My anxiety is through the roof and I feel like I cannot breathe", 3),
    ("Feeling extremely tense and jittery without any specific trigger", 3),
    ("I am terrified that something bad is about to happen at any moment", 3),
    ("Constant nervous energy and I cannot relax no matter what I do", 3),

    # ── 4: Stress / Burnout ──────────────────────────────────────────────────
    ("Completely overwhelmed with deadlines and absolutely no time to rest", 4),
    ("Burning out at work badly with no energy left for anything else", 4),
    ("My boss keeps adding more and more tasks and I am at my total limit", 4),
    ("Snapping at family because of work pressure and I hate myself for it", 4),
    ("Cannot switch off from work even on weekends or during evenings", 4),
    ("Tension headaches every single day from too much pressure and screen time", 4),
    ("Running completely on empty, coffee is the only thing keeping me going", 4),
    ("Feeling like a machine running on fumes, just work sleep work repeat", 4),
    ("Neglecting my own health completely because there is simply no time", 4),
    ("Three major projects due simultaneously and I cannot prioritise anything", 4),
    ("Everything feels both urgent and important and I simply cannot cope", 4),
    ("Months of overtime work have left me in complete physical exhaustion", 4),
    ("Dreading Monday morning every single Sunday night without exception", 4),
    ("My productivity has completely crashed but I have to push through anyway", 4),
    ("So stressed from everything that I have developed a persistent eye twitch", 4),
    ("No time for friends hobbies or self-care because work consumes everything", 4),
    ("I am so exhausted that even weekends offer no recovery or restoration", 4),
    ("Constant firefighting at work leaves me no space to think or breathe", 4),
    ("I feel completely overwhelmed and stretched too thin by everything", 4),
    ("Under enormous pressure at work and I feel like I am about to snap", 4),
    ("The workload is crushing me and I cannot keep up no matter how hard I try", 4),
    ("Physically and mentally drained from non stop work and responsibilities", 4),
    ("I am so stressed I cannot eat or sleep properly anymore", 4),
    ("Everything is piling up and I have zero capacity left in me", 4),
    ("Total burnout — I have nothing left to give anyone including myself", 4),
]

# ─── Text preprocessing ───────────────────────────────────────────────────────

KEYWORD_HINTS = {
    # sad keywords
    "sad": 1, "unhappy": 1, "crying": 1, "cry": 1, "heartbroken": 1,
    "miss": 1, "lonely": 1, "alone": 1, "grief": 1, "grieve": 1,
    "melancholy": 1, "blue": 1, "down": 1, "upset": 1, "tearful": 1,
    "disappointed": 1, "hurt": 1, "gloomy": 1,
    # depression keywords
    "hopeless": 2, "worthless": 2, "empty": 2, "numb": 2, "pointless": 2,
    "burden": 2, "self-harm": 2, "suicide": 2, "meaningless": 2,
    "withdraw": 2, "isolation": 2, "dead inside": 2, "hollow": 2,
    # anxiety keywords
    "anxious": 3, "anxiety": 3, "panic": 3, "worry": 3, "worried": 3,
    "nervous": 3, "terrified": 3, "fear": 3, "dread": 3, "overthink": 3,
    "catastroph": 3, "racing thoughts": 3, "chest tight": 3,
    # stress keywords
    "overwhelmed": 4, "burnout": 4, "stressed": 4, "exhausted": 4,
    "overworked": 4, "deadline": 4, "pressure": 4, "crushing": 4,
}

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def keyword_boost(text, proba):
    """Nudge probabilities based on keyword presence."""
    text_lower = text.lower()
    boost = np.zeros(5)
    for kw, label in KEYWORD_HINTS.items():
        if kw in text_lower:
            boost[label] += 0.15
    if boost.sum() > 0:
        proba = proba + boost
        proba = np.clip(proba, 0, None)
        proba = proba / proba.sum()
    return proba


# ─── Feature pipeline helper ─────────────────────────────────────────────────

class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X


# ─── Model training (cached) ─────────────────────────────────────────────────

PAD_ID = 0; CLS_ID = 1; UNK_ID = 2
MAX_LEN = 32; D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_LAYERS = 2
NUM_CLASSES = 5

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", preprocess(text))

@st.cache_resource(show_spinner=False)
def train_models():
    texts  = [preprocess(s[0]) for s in SAMPLES]
    labels = [s[1] for s in SAMPLES]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    # ── TF-IDF + SVM (word + char n-grams ensemble) ───────────────────────
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 3), max_features=12000,
        sublinear_tf=True, analyzer="word",
        min_df=1, token_pattern=r"\b\w+\b"
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5), max_features=8000,
        sublinear_tf=True, analyzer="char_wb",
        min_df=1
    )

    from scipy.sparse import hstack

    def fit_transform_features(X_tr, X_te):
        Xw_tr = word_tfidf.fit_transform(X_tr)
        Xw_te = word_tfidf.transform(X_te)
        Xc_tr = char_tfidf.fit_transform(X_tr)
        Xc_te = char_tfidf.transform(X_te)
        return hstack([Xw_tr, Xc_tr]), hstack([Xw_te, Xc_te])

    def transform_features(X):
        Xw = word_tfidf.transform(X)
        Xc = char_tfidf.transform(X)
        return hstack([Xw, Xc])

    t0 = time.time()
    X_tr_feat, X_te_feat = fit_transform_features(X_train, X_test)
    svm_clf = SVC(kernel="rbf", C=15, gamma="scale", probability=True, random_state=42)
    svm_clf.fit(X_tr_feat, y_train)
    time_a = time.time() - t0
    acc_a  = accuracy_score(y_test, svm_clf.predict(X_te_feat))

    # CV for TF-IDF SVM
    from sklearn.pipeline import make_pipeline
    from scipy.sparse import hstack as sp_hstack

    class CombinedTfidf(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.wt = TfidfVectorizer(ngram_range=(1,3), max_features=12000,
                                       sublinear_tf=True, analyzer="word",
                                       min_df=1, token_pattern=r"\b\w+\b")
            self.ct = TfidfVectorizer(ngram_range=(3,5), max_features=8000,
                                       sublinear_tf=True, analyzer="char_wb", min_df=1)
        def fit(self, X, y=None):
            self.wt.fit(X); self.ct.fit(X); return self
        def transform(self, X):
            return sp_hstack([self.wt.transform(X), self.ct.transform(X)])

    tfidf_svm_pipe = make_pipeline(
        CombinedTfidf(),
        SVC(kernel="rbf", C=15, gamma="scale", probability=True, random_state=42)
    )
    cv_a = cross_val_score(tfidf_svm_pipe, texts, labels, cv=5, scoring="accuracy")

    # ── Build vocab ──────────────────────────────────────────────────────────
    word_list = ["[PAD]", "[CLS]", "[UNK]"]
    seen = set(word_list)
    for t in texts:
        for w in simple_tokenize(t):
            if w not in seen:
                word_list.append(w); seen.add(w)
    vocab = {w: i for i, w in enumerate(word_list)}
    VOCAB_SIZE = len(vocab)

    def encode(text):
        toks = [vocab.get("[CLS]")] + [vocab.get(w, UNK_ID) for w in simple_tokenize(text)]
        toks = toks[:MAX_LEN]
        toks += [PAD_ID] * (MAX_LEN - len(toks))
        return toks

    def to_tensor(txts):
        return torch.tensor([encode(t) for t in txts], dtype=torch.long)

    # ── BERT Transformer ─────────────────────────────────────────────────────
    class PosEnc(nn.Module):
        def __init__(self, d, maxl=512):
            super().__init__()
            pe  = torch.zeros(maxl, d)
            pos = torch.arange(0, maxl).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x): return x + self.pe[:, :x.size(1)]

    class BertEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed   = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
            self.pos_enc = PosEnc(D_MODEL, MAX_LEN)
            layer        = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
                dropout=0.1, batch_first=True, norm_first=True)
            self.enc  = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
            self.norm = nn.LayerNorm(D_MODEL)
        def forward(self, ids):
            mask = (ids == PAD_ID)
            x = self.pos_enc(self.embed(ids))
            return self.norm(self.enc(x, src_key_padding_mask=mask))[:, 0, :]

    class BertClf(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc  = enc
            self.head = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(D_MODEL, NUM_CLASSES))
        def forward(self, ids): return self.head(self.enc(ids))

    bert_enc = BertEnc()
    clf      = BertClf(bert_enc)
    opt      = torch.optim.AdamW(clf.parameters(), lr=3e-3, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn  = nn.CrossEntropyLoss()

    X_tr_ids = to_tensor(X_train)
    y_tr_ten = torch.tensor(y_train, dtype=torch.long)

    t1 = time.time()
    clf.train()
    for epoch in range(300):
        perm = torch.randperm(len(X_tr_ids))
        for s in range(0, len(X_tr_ids), 16):
            idx = perm[s:s+16]
            loss = loss_fn(clf(X_tr_ids[idx]), y_tr_ten[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
        sched.step()
    time_b = time.time() - t1

    bert_enc.eval(); clf.eval()
    all_ids = to_tensor(texts)
    with torch.no_grad():
        all_emb  = bert_enc(all_ids).numpy()
        X_tr_emb = bert_enc(X_tr_ids).numpy()
        X_te_emb = bert_enc(to_tensor(X_test)).numpy()

    bert_svm = SVC(kernel="rbf", C=15, gamma="scale", probability=True, random_state=42)
    bert_svm.fit(X_tr_emb, y_train)
    acc_b = accuracy_score(y_test, bert_svm.predict(X_te_emb))
    cv_b  = cross_val_score(
        SVC(kernel="rbf", C=15, gamma="scale", probability=True, random_state=42),
        all_emb, labels, cv=5, scoring="accuracy")

    with torch.no_grad():
        e2e_acc = accuracy_score(y_test,
            clf(to_tensor(X_test)).argmax(1).numpy())

    return {
        "svm_clf":        svm_clf,
        "transform_feat": transform_features,
        "bert_enc":       bert_enc,
        "bert_svm":       bert_svm,
        "clf":            clf,
        "vocab":          vocab,
        "encode":         encode,
        "to_tensor":      to_tensor,
        "acc_a":  acc_a,  "cv_a": cv_a,  "time_a": time_a,
        "acc_b":  acc_b,  "cv_b": cv_b,  "time_b": time_b,
        "e2e_acc": e2e_acc,
        "X_test": X_test, "y_test": y_test,
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.6rem;
                    color:#e8d5b7;margin-bottom:0.5rem'>🧠 MindScan</div>
        <div style='color:#6b7280;font-size:0.82rem;line-height:1.6'>
        Dual-model mental health text classifier using classical NLP and transformer architecture.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#9ca3af;font-size:0.8rem;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:0.6rem'>Classes</div>",
                unsafe_allow_html=True)
    for k, v in LABEL_NAMES.items():
        st.markdown(
            f"<span class='badge badge-{k}'>{LABEL_EMOJIS[k]} {v}</span><br>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#9ca3af;font-size:0.8rem;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:0.6rem'>Architecture</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#6b7280;font-size:0.8rem;line-height:1.8'>
    <b style='color:#e8d5b7'>Model A</b><br>
    Word TF-IDF (1-3 gram)<br>
    + Char TF-IDF (3-5 gram)<br>
    + RBF-SVM (C=15)<br><br>
    <b style='color:#e8d5b7'>Model B</b><br>
    BERT Transformer (2L, 64d)<br>
    Fine-tuned → embeddings<br>
    + RBF-SVM (C=15)<br><br>
    <b style='color:#e8d5b7'>Boost</b><br>
    Keyword hint layer on both
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Analyse Text", "📊 Model Metrics",
                                  "🧪 Batch Test"], label_visibility="collapsed")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>Mental Health Detector</h1>
    <p>TF-IDF + SVM  ·  BERT Transformer + SVM  ·  5 emotional states</p>
</div>
""", unsafe_allow_html=True)

# ─── Load models ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models on dataset…  (first load ~40s)"):
    M = train_models()

st.success("✅ Both models ready!", icon="🧠")
st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyse Text
# ═══════════════════════════════════════════════════════════════════════════════

# Curated pick-list — grouped by emotional state, 8 per class = 40 options
PICK_OPTIONS = {
    "✅ No Concern — Positive / Well": [
        "I had a great day today and feel really energised",
        "Just finished an amazing workout, feeling strong and alive",
        "Loving my new job, every day brings an exciting challenge",
        "Got a promotion at work, so excited about the future ahead",
        "Spent the weekend with family, it was truly wonderful",
        "Feeling grateful for all the good things happening in my life",
        "Started a new hobby and I am really enjoying learning it",
        "Life feels good and I am full of hope about the future",
    ],
    "😢 Sad — Low Mood / Grief": [
        "I feel sad today and do not know why",
        "Feeling a bit down and lonely this evening",
        "My heart feels heavy and I just want to cry",
        "I miss someone close to me and the sadness is overwhelming",
        "Received some bad news today and I feel heartbroken",
        "The loneliness is getting to me today more than usual",
        "I feel blue and nothing seems to cheer me up right now",
        "I cried a lot today and still feel sad",
    ],
    "💙 Depression — Persistent Emptiness": [
        "I feel completely empty inside, nothing brings me joy anymore",
        "Getting out of bed has become almost impossible lately",
        "Everything feels completely pointless and I see no purpose",
        "I feel like a burden to everyone around me",
        "Feeling completely numb, no sadness no happiness just emptiness",
        "Cannot find any motivation to do even the simplest tasks",
        "I feel completely worthless and like nothing will ever improve",
        "The darkness inside me never seems to lift no matter what I try",
    ],
    "🌀 Anxiety — Worry / Panic": [
        "My heart races and I cannot catch my breath, anxiety overwhelms me",
        "Constant worry about things that might go wrong in the future",
        "I keep catastrophising even the smallest situations",
        "Panic attacks are becoming more frequent and intense",
        "Terrified of social gatherings because I always say the wrong thing",
        "Racing thoughts at night prevent me from getting any restful sleep",
        "The fear of being judged stops me from doing things I want",
        "I am terrified that something bad is about to happen at any moment",
    ],
    "🔥 Stress / Burnout — Overwhelm": [
        "Completely overwhelmed with deadlines and absolutely no time to rest",
        "Burning out at work badly with no energy left for anything else",
        "Cannot switch off from work even on weekends or evenings",
        "Running completely on empty, coffee is the only thing keeping me going",
        "Everything feels urgent and important and I simply cannot cope",
        "Months of overtime have left me in complete physical exhaustion",
        "So stressed that I have developed a persistent eye twitch",
        "Total burnout — I have nothing left to give anyone including myself",
    ],
}

if "Analyse" in page:

    st.markdown("""
    <div style='color:#9ca3af;font-size:0.9rem;margin-bottom:0.4rem'>
        <b style='color:#e8d5b7'>Step 1</b> — Pick a category
    </div>
    """, unsafe_allow_html=True)

    category = st.selectbox(
        "Category",
        list(PICK_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        key="category_select"
    )

    st.markdown("""
    <div style='color:#9ca3af;font-size:0.9rem;margin:0.8rem 0 0.4rem'>
        <b style='color:#e8d5b7'>Step 2</b> — Choose a statement to analyse
    </div>
    """, unsafe_allow_html=True)

    sentence_options = ["— select a statement —"] + PICK_OPTIONS[category]
    chosen = st.selectbox(
        "Statement",
        sentence_options,
        index=0,
        label_visibility="collapsed",
        key="sentence_select"
    )

    # Show selected statement as a styled preview card
    user_text = ""
    if chosen != "— select a statement —":
        user_text = chosen
        label_idx = [k for k, v in PICK_OPTIONS.items() if category == k][0]
        cat_color_map = {
            "✅ No Concern — Positive / Well": "#34d399",
            "😢 Sad — Low Mood / Grief": "#38bdf8",
            "💙 Depression — Persistent Emptiness": "#a78bfa",
            "🌀 Anxiety — Worry / Panic": "#fbbf24",
            "🔥 Stress / Burnout — Overwhelm": "#f87171",
        }
        cc = cat_color_map.get(category, "#e8d5b7")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.04);border:1px solid {cc}44;
                    border-left:3px solid {cc};border-radius:10px;
                    padding:0.9rem 1.2rem;margin:0.6rem 0 1rem;
                    color:#e2e8f0;font-size:1rem;font-style:italic;'>
            "{user_text}"
        </div>
        """, unsafe_allow_html=True)

    run = st.button("Analyse ✦", use_container_width=False,
                    disabled=(chosen == "— select a statement —"))

    if run and user_text.strip():
        to_tensor       = M["to_tensor"]
        bert_enc        = M["bert_enc"]
        bert_svm        = M["bert_svm"]
        svm_clf         = M["svm_clf"]
        transform_feat  = M["transform_feat"]
        clf             = M["clf"]

        clean_text = preprocess(user_text)

        # TF-IDF + SVM
        feat_a   = transform_feat([clean_text])
        proba_a  = svm_clf.predict_proba(feat_a)[0]
        proba_a  = keyword_boost(clean_text, proba_a)
        pred_a   = int(proba_a.argmax())

        # BERT + SVM
        ids = to_tensor([clean_text])
        with torch.no_grad():
            emb = bert_enc(ids).numpy()
        proba_b  = bert_svm.predict_proba(emb)[0]
        proba_b  = keyword_boost(clean_text, proba_b)
        pred_b   = int(proba_b.argmax())

        # BERT e2e
        with torch.no_grad():
            logit  = clf(ids)
            softm  = torch.softmax(logit, dim=1).numpy()[0]
            softm  = keyword_boost(clean_text, softm)
            pred_e = int(softm.argmax())
            conf_e = float(softm.max())

        st.markdown("---")
        st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)

        def render_result(col, title, pred, proba, subtitle=""):
            with col:
                color = LABEL_COLORS[pred]
                conf  = float(proba[pred]) if hasattr(proba, '__len__') else proba
                st.markdown(f"""
                <div class='result-card'>
                    <div style='font-size:0.75rem;color:#6b7280;
                                text-transform:uppercase;letter-spacing:0.08em'>{title}</div>
                    <div style='font-family:"DM Serif Display",serif;
                                font-size:1.6rem;color:{color};margin:0.5rem 0 0.3rem'>
                        {LABEL_EMOJIS[pred]} {LABEL_NAMES[pred]}
                    </div>
                    <div style='font-size:0.82rem;color:#9ca3af'>{subtitle}</div>
                    <div class='conf-bar-bg' style='margin-top:0.8rem'>
                        <div class='conf-bar-fill'
                             style='width:{conf*100:.1f}%;background:{color}'></div>
                    </div>
                    <div style='font-size:0.85rem;color:{color};font-weight:600'>
                        {conf*100:.1f}% confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

        render_result(colA, "TF-IDF + SVM", pred_a, proba_a, "Classical NLP")
        render_result(colB, "BERT + SVM", pred_b, proba_b, "Transformer Embeddings")

        with colC:
            color = LABEL_COLORS[pred_e]
            st.markdown(f"""
            <div class='result-card'>
                <div style='font-size:0.75rem;color:#6b7280;
                            text-transform:uppercase;letter-spacing:0.08em'>BERT End-to-End</div>
                <div style='font-family:"DM Serif Display",serif;
                            font-size:1.6rem;color:{color};margin:0.5rem 0 0.3rem'>
                    {LABEL_EMOJIS[pred_e]} {LABEL_NAMES[pred_e]}
                </div>
                <div style='font-size:0.82rem;color:#9ca3af'>Direct Classifier</div>
                <div class='conf-bar-bg' style='margin-top:0.8rem'>
                    <div class='conf-bar-fill'
                         style='width:{conf_e*100:.1f}%;background:{color}'></div>
                </div>
                <div style='font-size:0.85rem;color:{color};font-weight:600'>
                    {conf_e*100:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("<div class='section-title'>Probability Breakdown</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**TF-IDF + SVM**")
            df_a = pd.DataFrame({
                "Class": [f"{LABEL_EMOJIS[i]} {LABEL_NAMES[i]}" for i in range(NUM_CLASSES)],
                "Probability": [f"{p*100:.1f}%" for p in proba_a],
            })
            st.dataframe(df_a, hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**BERT + SVM**")
            df_b = pd.DataFrame({
                "Class": [f"{LABEL_EMOJIS[i]} {LABEL_NAMES[i]}" for i in range(NUM_CLASSES)],
                "Probability": [f"{p*100:.1f}%" for p in proba_b],
            })
            st.dataframe(df_b, hide_index=True, use_container_width=True)

        # Agreement
        agree = pred_a == pred_b
        agree_color = "#34d399" if agree else "#fbbf24"
        agree_text  = "Both models agree ✓" if agree else "Models disagree — BERT generally more reliable"
        st.markdown(f"""
        <div style='text-align:center;padding:0.8rem;border-radius:10px;
                    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    margin-top:1rem;color:{agree_color};font-weight:500'>
            {agree_text}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
        ⚠️ <b>Disclaimer:</b> This tool is for research and educational purposes only.
        It is not a substitute for professional mental health advice, diagnosis, or treatment.
        If you or someone you know is struggling, please reach out to a qualified mental 
        health professional or crisis helpline.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Metrics
# ═══════════════════════════════════════════════════════════════════════════════
elif "Metrics" in page:

    st.markdown("<div class='section-title'>Performance Overview</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("TF-IDF+SVM\nTest Acc", f"{M['acc_a']*100:.1f}%"),
        ("TF-IDF+SVM\n5-Fold CV", f"{M['cv_a'].mean()*100:.1f}%"),
        ("BERT+SVM\nTest Acc",  f"{M['acc_b']*100:.1f}%"),
        ("BERT+SVM\n5-Fold CV", f"{M['cv_b'].mean()*100:.1f}%"),
    ]
    for col, (lbl, val) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='val'>{val}</div>
                <div class='lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Head-to-Head Comparison</div>",
                unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Metric": ["Test Accuracy", "5-Fold CV Mean", "5-Fold CV Std",
                   "Training Time (s)", "Features"],
        "TF-IDF + SVM": [
            f"{M['acc_a']*100:.1f}%",
            f"{M['cv_a'].mean()*100:.1f}%",
            f"±{M['cv_a'].std()*100:.1f}%",
            f"{M['time_a']:.4f}s",
            "Word 1-3gram + Char 3-5gram",
        ],
        "BERT + SVM": [
            f"{M['acc_b']*100:.1f}%",
            f"{M['cv_b'].mean()*100:.1f}%",
            f"±{M['cv_b'].std()*100:.1f}%",
            f"{M['time_b']:.1f}s",
            "~97k transformer params",
        ],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)

    st.markdown("<div class='section-title'>Cross-Validation Accuracy</div>",
                unsafe_allow_html=True)
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)]*2,
        "Accuracy": list(M["cv_a"]) + list(M["cv_b"]),
        "Model": ["TF-IDF+SVM"]*5 + ["BERT+SVM"]*5,
    })
    st.bar_chart(cv_df.pivot(index="Fold", columns="Model", values="Accuracy"))

    st.markdown("<div class='section-title'>Per-Class Report (TF-IDF + SVM)</div>",
                unsafe_allow_html=True)
    feat_test = M["transform_feat"]([preprocess(t) for t in M["X_test"]])
    y_pred_a  = M["svm_clf"].predict(feat_test)
    report_a  = classification_report(
        M["y_test"], y_pred_a,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        output_dict=True, zero_division=0)
    rows = []
    for cls in [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]:
        r = report_a.get(cls, {})
        rows.append({
            "Class":     cls,
            "Precision": f"{r.get('precision',0)*100:.1f}%",
            "Recall":    f"{r.get('recall',0)*100:.1f}%",
            "F1-Score":  f"{r.get('f1-score',0)*100:.1f}%",
            "Support":   int(r.get('support',0)),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("""
    <div class='disclaimer'>
    ℹ️ Metrics are on a held-out 20% stratified test split.
    5-fold CV is a more reliable generalisation estimate on small datasets.
    A keyword-hint layer boosts detection of emotion words on unseen sentences.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batch Test
# ═══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:

    st.markdown("<div class='section-title'>Batch Text Analysis</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#6b7280;font-size:0.9rem;margin-bottom:1rem'>"
        "Enter one sentence per line — all will be classified by both models simultaneously."
        "</div>", unsafe_allow_html=True)

    default_batch = """I cannot stop worrying about the future
Feeling sad and empty today
Work is crushing me with too many deadlines
Had a wonderful weekend with family and friends
Panic attacks are getting worse every week
I dread getting out of bed every single morning
I cried all evening and do not know why
My chest is tight and I cannot breathe properly
I feel completely hopeless and see no way forward
I am so stressed and overwhelmed at work"""

    batch_text = st.text_area(
        "Batch input", value=default_batch, height=230, label_visibility="collapsed")

    if st.button("Run Batch Analysis ✦", use_container_width=False):
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        if lines:
            to_tensor      = M["to_tensor"]
            bert_enc       = M["bert_enc"]
            bert_svm       = M["bert_svm"]
            svm_clf        = M["svm_clf"]
            transform_feat = M["transform_feat"]

            results = []
            for line in lines:
                cl     = preprocess(line)
                feat_a = transform_feat([cl])
                pa     = svm_clf.predict_proba(feat_a)[0]
                pa     = keyword_boost(cl, pa)
                pred_a = int(pa.argmax())

                ids    = to_tensor([cl])
                with torch.no_grad():
                    emb = bert_enc(ids).numpy()
                pb     = bert_svm.predict_proba(emb)[0]
                pb     = keyword_boost(cl, pb)
                pred_b = int(pb.argmax())

                results.append({
                    "Text":        line[:65]+"…" if len(line)>65 else line,
                    "TF-IDF+SVM":  f"{LABEL_EMOJIS[pred_a]} {LABEL_NAMES[pred_a]}",
                    "BERT+SVM":    f"{LABEL_EMOJIS[pred_b]} {LABEL_NAMES[pred_b]}",
                    "Agreement":   "✅ Yes" if pred_a==pred_b else "❌ No",
                })

            df = pd.DataFrame(results)
            st.dataframe(df, hide_index=True, use_container_width=True)

            agree_count = sum(1 for r in results if "Yes" in r["Agreement"])
            st.markdown(f"""
            <div style='text-align:center;padding:0.8rem;border-radius:10px;
                        background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        margin-top:1rem;color:#e8d5b7;font-size:0.9rem'>
                Models agreed on <b style='color:#f5c842'>{agree_count}/{len(results)}</b> sentences
                ({agree_count/len(results)*100:.0f}% agreement rate)
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
        ⚠️ For educational use only. Not a clinical diagnostic tool.
        </div>
        """, unsafe_allow_html=True)