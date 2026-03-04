from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
COLLECTION_NAME = "pdf_rag"

# ----------------------------
# Settings (tune these)
# ----------------------------
CHUNK_SIZE = 500   # πόσους χαρακτήρες να έχει κάθε κομμάτι (chunk) κειμένου. 500-1000 είναι συνήθως καλό για LLM context. Μεγαλύτερα chunks = λιγότερα συνολικά, αλλά μπορεί να είναι λιγότερο ακριβή στην ανάκτηση.
CHUNK_OVERLAP = 150 # πόσοι χαρακτήρες να επικαλύπτονται μεταξύ διαδοχικών chunks. Αυτό βοηθά να μην χάνεται το νόημα που μπορεί να βρίσκεται στα όρια των chunks.

# Retrieval: grab many, then keep best
FETCH_K = 30     # πόσα σχετικά chunks να φέρνουμε από τη βάση για κάθε ερώτηση. Πάνω από 20-30 συνήθως δεν βοηθάει (και μπορεί να κάνει το LLM πιο αργό ή λιγότερο ακριβές). Μετά θα κρατήσουμε μόνο τα K_FINAL καλύτερα για το context.
K_FINAL = 6      # πόσα από τα φέρε-πολλά (fetch_k) chunks να κρατάμε τελικά για το context. 4-8 είναι συνήθως μια καλή ισορροπία μεταξύ πληροφορίας και θορύβου.

# Embeddings model (try mxbai-embed-large if you have it in Ollama)
EMBED_MODEL = "nomic-embed-text" # ένα ισχυρό μοντέλο embedding που είναι διαθέσιμο στο Ollama. Αν έχετε το mxbai-embed-large, δοκιμάστε και αυτό για πιθανώς καλύτερα αποτελέσματα (αν και μπορεί να είναι πιο αργό).

# LLM model
LLM_MODEL = "llama3.1" # ένα ισχυρό μοντέλο LLM που είναι διαθέσιμο στο Ollama.  
TEMPERATURE = 0 # χαμηλή θερμοκρασία σημαίνει πιο ακριβείς και λιγότερο "φανταστικές" απαντήσεις, που είναι επιθυμητό για ένα σύστημα βασισμένο σε πραγματικά δεδομένα.

# ----------------------------
# Optional reranker (Cross-Encoder)
# ----------------------------
def try_load_reranker(): # Κάνει προσπάθεια να φορτώσει ένα reranker μοντέλο από τη βιβλιοθήκη sentence-transformers. Αν δεν είναι εγκατεστημένη, επιστρέφει None και το pipeline θα λειτουργεί χωρίς reranking.
    """
    Tries to load a local reranker via sentence-transformers.
    If not installed, returns None and the pipeline will still work (just without rerank).
    """
    try:
        from sentence_transformers import CrossEncoder # Ένα ισχυρό μοντέλο reranker που μπορεί να αξιολογήσει την σχετικότητα κάθε chunk με την ερώτηση. Χρησιμοποιεί ένα cross-encoder που λαμβάνει ως είσοδο το ζεύγος (ερώτηση, chunk) και επιστρέφει μια βαθμολογία σχετικότητας. Αυτό μπορεί να βελτιώσει την ποιότητα των τελικών chunks που παρέχονται στο LLM, ειδικά αν το αρχικό retriever φέρνει πολλά σχετικά αλλά όχι αρκετά καλά chunks.
        # A strong, widely used reranker:
        # If you want smaller/faster: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" # ένα μικρότερο και ταχύτερο reranker μοντέλο που είναι επίσης διαθέσιμο. Μπορεί να είναι λιγότερο ακριβές από το μεγαλύτερο, αλλά θα είναι πιο γρήγορο.
        return CrossEncoder(model_name) # φορτώνει το μοντέλο reranker και το επιστρέφει. Αν η βιβλιοθήκη sentence-transformers δεν είναι εγκατεστημένη, θα πιαστεί η εξαίρεση και θα επιστρέψει None.
    except Exception: # 
        return None

def rerank_docs(reranker, question: str, docs, top_n: int) -> List: # Χρησιμοποιεί το reranker για να αξιολογήσει και να ταξινομήσει τα έγγραφα που επέστρεψε ο retriever με βάση τη σχετικότητά τους με την ερώτηση. Επιστρέφει τα top_n πιο σχετικά έγγραφα.
    """
    Reranks docs based on relevance to the question using a cross-encoder.
    """
    pairs = [(question, d.page_content) for d in docs] # δημιουργεί ζεύγη (ερώτηση, κείμενο chunk) για κάθε έγγραφο που επέστρεψε ο retriever. Αυτά τα ζεύγη θα αξιολογηθούν από το reranker για να δώσουν μια βαθμολογία σχετικότητας.
    scores = reranker.predict(pairs) # χρησιμοποιεί το reranker για να προβλέψει μια βαθμολογία σχετικότητας για κάθε ζεύγος (ερώτηση, chunk). Το αποτέλεσμα είναι ένας πίνακας βαθμολογιών που αντιστοιχεί σε κάθε έγγραφο.
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) # συνδυάζει τις βαθμολογίες με τα έγγραφα και τα ταξινομεί κατά φθίνουσα σειρά βαθμολογίας, έτσι ώστε τα πιο σχετικά έγγραφα να βρίσκονται στην κορυφή της λίστας.
    return [d for _, d in ranked[:top_n]] # επιστρέφει τα top_n έγγραφα με τις υψηλότερες βαθμολογίες σχετικότητας. Αυτά τα έγγραφα θα χρησιμοποιηθούν για να δημιουργήσουν το context που θα δοθεί στο LLM για να απαντήσει στην ερώτηση.

# ----------------------------
# 1) Load PDFs
# ----------------------------
documents = []
for pdf in DATA_DIR.glob("*.pdf"): # ψάχνει για όλα τα αρχεία PDF στον φάκελο δεδομένων και τα φορτώνει χρησιμοποιώντας το PyMuPDFLoader. Κάθε σελίδα του PDF μετατρέπεται σε ένα Document object που περιέχει το κείμενο της σελίδας και μεταδεδομένα όπως η πηγή (source) και ο αριθμός σελίδας (page). Όλα τα Document objects αποθηκεύονται στη λίστα documents. 
    loader = PyMuPDFLoader(str(pdf)) # δημιουργεί έναν loader για το συγκεκριμένο PDF αρχείο. Ο PyMuPDFLoader χρησιμοποιεί τη βιβλιοθήκη PyMuPDF για να διαβάσει το PDF και να εξάγει το κείμενο από κάθε σελίδα.
    docs = loader.load()  # φορτώνει το PDF και επιστρέφει μια λίστα από Document objects, ένα για κάθε σελίδα του PDF. Κάθε Document περιέχει το κείμενο της σελίδας και μεταδεδομένα όπως η πηγή (source) και ο αριθμός σελίδας (page).
    documents.extend(docs)

print("PDF pages loaded:", len(documents))

# ----------------------------
# 2) Split into chunks
# ----------------------------
splitter = RecursiveCharacterTextSplitter( # δημιουργεί έναν text splitter που θα χωρίσει το κείμενο των Document objects σε μικρότερα κομμάτια (chunks) με βάση τους χαρακτήρες. Ο RecursiveCharacterTextSplitter προσπαθεί να σπάσει το κείμενο σε λογικά σημεία (όπως παραγράφους ή προτάσεις) για να διατηρήσει το νόημα, ενώ σέβεται τα όρια του chunk_size και chunk_overlap που έχουμε ορίσει. Αυτό βοηθά να δημιουργήσουμε chunks που είναι αρκετά μικρά για να χωρέσουν στο context του LLM, αλλά αρκετά μεγάλα για να διατηρήσουν το νόημα.
    chunk_size=CHUNK_SIZE, # πόσους χαρακτήρες να έχει κάθε chunk. 500-1000 είναι συνήθως καλό για LLM context. Μεγαλύτερα chunks = λιγότερα συνολικά, αλλά μπορεί να είναι λιγότερο ακριβή στην ανάκτηση.  
    chunk_overlap=CHUNK_OVERLAP, # πόσοι χαρακτήρες να επικαλύπτονται μεταξύ διαδοχικών chunks. Αυτό βοηθά να μην χάνεται το νόημα που μπορεί να βρίσκεται στα όρια των chunks.
)
chunks = splitter.split_documents(documents)
print("Document chunks created:", len(chunks))

# ----------------------------
# 3) Embeddings
# ----------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL) # δημιουργεί ένα αντικείμενο embeddings που χρησιμοποιεί το μοντέλο που ορίσαμε στο Ollama για να μετατρέψει τα κείμενα των chunks σε διανύσματα (vectors). Αυτά τα διανύσματα θα χρησιμοποιηθούν από τη βάση δεδομένων vector για να βρουν τα πιο σχετικά chunks με βάση την ομοιότητα με την ερώτηση.

# ----------------------------
# 4) Vector DB (Chroma)
# ----------------------------
if DB_DIR.exists(): # ελέγχει αν ο φάκελος της βάσης δεδομένων υπάρχει ήδη. 
    print("Loading existing database...")# Αν ναι, φορτώνει την υπάρχουσα βάση δεδομένων Chroma από τον δίσκο.
    vectordb = Chroma(       # Αυτό επιτρέπει να μην χρειάζεται να επαναδημιουργούμε τη βάση δεδομένων κάθε φορά που τρέχουμε το πρόγραμμα, κάτι που μπορεί να είναι χρονοβόρο αν έχουμε πολλά έγγραφα.
        persist_directory=str(DB_DIR), # ο φάκελος όπου είναι αποθηκευμένη η βάση δεδομένων Chroma.
        embedding_function=embeddings, # το αντικείμενο embeddings που θα χρησιμοποιηθεί για να μετατρέψει τα chunks σε διανύσματα όταν χρειάζεται.
        collection_name=COLLECTION_NAME, # το όνομα της συλλογής μέσα στη βάση δεδομένων Chroma όπου είναι αποθηκευμένα τα chunks και τα embeddings τους. Αυτό επιτρέπει να έχουμε πολλαπλές συλλογές στην ίδια βάση δεδομένων αν θέλουμε.
    )

else:
    print("Creating database (first run only)...")# Αν ο φάκελος της βάσης δεδομένων δεν υπάρχει, δημιουργεί μια νέα βάση δεδομένων Chroma από τα chunks και τα embeddings τους. Αυτό θα είναι χρονοβόρο την πρώτη φορά που τρέχουμε το πρόγραμμα, αλλά μετά θα φορτώνει γρήγορα την υπάρχουσα βάση δεδομένων.
    vectordb = Chroma.from_documents( # δημιουργεί μια νέα βάση δεδομένων Chroma από τα Document objects (chunks) και τα embeddings τους. Η μέθοδος from_documents παίρνει τη λίστα των chunks, το αντικείμενο embeddings για να μετατρέψει τα chunks σε διανύσματα, και τις πληροφορίες για το πού να αποθηκεύσει τη βάση δεδομένων και με ποιο όνομα συλλογής.
        chunks,
        embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )
    vectordb.persist() # αποθηκεύει τη βάση δεδομένων στον δίσκο για μελλοντική χρήση. Αυτό δημιουργεί τα απαραίτητα αρχεία και φακέλους που θα χρησιμοποιηθούν την επόμενη φορά που θα τρέξουμε το πρόγραμμα για να φορτώσουμε την υπάρχουσα βάση δεδομένων αντί να τη δημιουργήσουμε από την αρχή.

print("Database ready!")

# ----------------------------
# 5) Retriever (MMR = more diverse, less duplicate chunks)
# ----------------------------
retriever = vectordb.as_retriever( # δημιουργεί έναν retriever από τη βάση δεδομένων Chroma. Ο retriever είναι το αντικείμενο που θα χρησιμοποιήσουμε για να φέρουμε τα πιο σχετικά chunks από τη βάση δεδομένων με βάση την ερώτηση. Εδώ χρησιμοποιούμε την τεχνική MMR (Maximal Marginal Relevance) για να φέρουμε πιο διαφορετικά και λιγότερο παρόμοια chunks, κάτι που μπορεί να βοηθήσει να έχουμε ένα πιο πλούσιο και λιγότερο επαναλαμβανόμενο context για το LLM.
    search_type="mmr",
    search_kwargs={"k": K_FINAL, "fetch_k": FETCH_K},
)

# ----------------------------
# 6) LLM
# ----------------------------
llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE) # δημιουργεί ένα αντικείμενο LLM που χρησιμοποιεί το μοντέλο που ορίσαμε στο Ollama για να απαντάει στις ερωτήσεις. Η παράμετρος temperature ελέγχει πόσο "δημιουργικές" ή "ακριβείς" θα είναι οι απαντήσεις του LLM, με χαμηλότερες τιμές να οδηγούν σε πιο ακριβείς και λιγότερο "φανταστικές" απαντήσεις, κάτι που είναι επιθυμητό για ένα σύστημα βασισμένο σε πραγματικά δεδομένα.

# ----------------------------
# 7) Reranker (optional)
# ----------------------------
reranker = try_load_reranker() # προσπαθεί να φορτώσει ένα reranker μοντέλο από τη βιβλιοθήκη sentence-transformers. Αν δεν είναι εγκατεστημένη, επιστρέφει None και το pipeline θα λειτουργεί χωρίς reranking. Το reranker μπορεί να βελτιώσει την ποιότητα των τελικών chunks που παρέχονται στο LLM, ειδικά αν το αρχικό retriever φέρνει πολλά σχετικά αλλά όχι αρκετά καλά chunks.
if reranker:
    print("Reranker enabled: sentence-transformers CrossEncoder")
else:
    print("Reranker not found (sentence-transformers not installed). Continuing without rerank.")

# ----------------------------
# Helpers
# ----------------------------
def build_context(docs: List) -> str: # δημιουργεί ένα κείμενο context από μια λίστα Document objects (chunks) που θα δοθεί στο LLM μαζί με την ερώτηση. Κάθε chunk συνοδεύεται από μια "παραπομπή" στη μορφή [1], [2], κλπ., που αντιστοιχεί στην πηγή του chunk (το PDF και τη σελίδα από όπου προέρχεται). Το κείμενο κάθε chunk περιορίζεται στα πρώτα 900 χαρακτήρες για να κρατήσει το context συμπαγές και γρήγορο, ενώ ταυτόχρονα παρέχει αρκετή πληροφορία για να απαντήσει στην ερώτηση.
    """
    Builds compact context blocks with citations [1], [2], ...
    """
    blocks = []
    for i, d in enumerate(docs, 1): # για κάθε Document object στη λίστα των τελικών εγγράφων (chunks) που θα χρησιμοποιηθούν για το context, δημιουργεί ένα μπλοκ κειμένου που περιλαμβάνει μια παραπομπή στη μορφή [i] όπου i είναι ο αριθμός του chunk, την πηγή του chunk (το PDF και τη σελίδα), και ένα snippet του κειμένου του chunk περιορισμένο στα πρώτα 900 χαρακτήρες. Αυτά τα μπλοκ κειμένου συνδυάζονται σε ένα ενιαίο string που θα δοθεί στο LLM ως context.
        source = d.metadata.get("source", "unknown.pdf")
        page = d.metadata.get("page", "unknown")
        text = d.page_content.strip()
        # keep context compact + fast
        snippet = text[:900]
        blocks.append(f"[{i}] {Path(source).name} (page {page})\n{snippet}") # δημιουργεί ένα μπλοκ κειμένου για κάθε chunk με την μορφή: [i] filename.pdf (page X) \n snippet of text. Αυτό παρέχει στο LLM την πληροφορία για την πηγή του chunk και ένα κομμάτι του κειμένου που μπορεί να χρησιμοποιήσει για να απαντήσει στην ερώτηση.
    return "\n\n".join(blocks) # συνδυάζει όλα τα μπλοκ κειμένου σε ένα ενιαίο string με διπλά νέα γραμμές ως διαχωριστικό μεταξύ των chunks.

def make_prompt(context: str, question: str) -> str: # δημιουργεί το prompt που θα δοθεί στο LLM για να απαντήσει στην ερώτηση με βάση το context. Το prompt περιλαμβάνει κανόνες για το πώς πρέπει να χρησιμοποιεί το LLM το context, τι να κάνει αν το context δεν είναι επαρκές, και πώς να παραθέτει τις πηγές των πληροφοριών που χρησιμοποιεί. Στόχος είναι να καθοδηγήσει το LLM να δώσει μια ακριβή, σχετική και καλά τεκμηριωμένη απάντηση βασισμένη μόνο στο παρεχόμενο context.
    return f"""You are a factual QA system.

Rules:
- Use ONLY the context.
- If the context is insufficient, say exactly: "I don't have enough info in the PDFs."
- Do NOT guess or add outside knowledge.
- Cite sources like [1], [2], etc.
- Prefer short, precise answers. If helpful, use bullet points.

CONTEXT: 
{context}

QUESTION:
{question}
"""

# ----------------------------
# 8) Interactive loop
# ----------------------------
def ask_question(question: str): # αυτή είναι η κύρια συνάρτηση που παίρνει μια ερώτηση ως είσοδο, φέρνει τα σχετικά chunks από τη βάση δεδομένων χρησιμοποιώντας τον retriever, προαιρετικά τα επαναταξινομεί με το reranker για να κρατήσει μόνο τα πιο σχετικά, δημιουργεί το context και το prompt, και στέλνει το prompt στο LLM για να πάρει την απάντηση. Τέλος, επιστρέφει την απάντηση του LLM.

    candidate_docs = retriever.invoke(question) # χρησιμοποιεί τον retriever για να φέρει τα σχετικά chunks από τη βάση δεδομένων με βάση την ερώτηση. Ο retriever θα επιστρέψει μέχρι fetch_k chunks, και μετά θα κρατήσει μόνο τα K_FINAL καλύτερα για το context. Αν έχουμε ένα reranker, θα χρησιμοποιηθεί για να αξιολογήσει και να ταξινομήσει αυτά τα chunks πριν κρατήσουμε τα τελικά K_FINAL.

    final_docs = candidate_docs # αρχικά, τα τελικά έγγραφα είναι αυτά που επέστρεψε ο retriever. Αν έχουμε ένα reranker, θα χρησιμοποιήσουμε το reranker για να αξιολογήσουμε και να ταξινομήσουμε αυτά τα έγγραφα με βάση τη σχετικότητά τους με την ερώτηση, και θα κρατήσουμε μόνο τα top_n (K_FINAL) πιο σχετικά έγγραφα για να δημιουργήσουμε το context που θα δοθεί στο LLM.
    if reranker: # αν έχουμε ένα reranker, χρησιμοποιούμε τη συνάρτηση rerank_docs για να αξιολογήσουμε και να ταξινομήσουμε τα έγγραφα που επέστρεψε ο retriever με βάση τη σχετικότητά τους με την ερώτηση, και κρατάμε μόνο τα top_n (K_FINAL) πιο σχετικά έγγραφα για να δημιουργήσουμε το context που θα δοθεί στο LLM. Αυτό μπορεί να βελτιώσει την ποιότητα των τελικών chunks που παρέχονται στο LLM, ειδικά αν το αρχικό retriever φέρνει πολλά σχετικά αλλά όχι αρκετά καλά chunks.
        final_docs = rerank_docs( # χρησιμοποιεί τη συνάρτηση rerank_docs για να αξιολογήσει και να ταξινομήσει τα έγγραφα που επέστρεψε ο retriever με βάση τη σχετικότητά τους με την ερώτηση, και κρατάει μόνο τα top_n (K_FINAL) πιο σχετικά έγγραφα για να δημιουργήσουμε το context που θα δοθεί στο LLM. Αυτό μπορεί να βελτιώσει την ποιότητα των τελικών chunks που παρέχονται στο LLM, ειδικά αν το αρχικό retriever φέρνει πολλά σχετικά αλλά όχι αρκετά καλά chunks.
            reranker,
            question,
            candidate_docs,
            top_n=K_FINAL
        )

    context = build_context(final_docs) # δημιουργεί το context που θα δοθεί στο LLM από τα τελικά έγγραφα (chunks) που επέλεξε ο retriever (και προαιρετικά επαναταξινόμησε ο reranker). Το context περιλαμβάνει τα κείμενα των chunks μαζί με παραπομπές στη μορφή [1], [2], κλπ., που αντιστοιχούν στην πηγή του chunk (το PDF και τη σελίδα από όπου προέρχεται). Αυτό το context θα χρησιμοποιηθεί από το LLM για να απαντήσει στην ερώτηση με βάση τις πληροφορίες που περιέχονται στα chunks.
    prompt = make_prompt(context, question) # δημιουργεί το prompt που θα δοθεί στο LLM για να απαντήσει στην ερώτηση με βάση το context. Το prompt περιλαμβάνει κανόνες για το πώς πρέπει να χρησιμοποιεί το LLM το context, τι να κάνει αν το context δεν είναι επαρκές, και πώς να παραθέτει τις πηγές των πληροφοριών που χρησιμοποιεί. Στόχος είναι να καθοδηγήσει το LLM να δώσει μια ακριβή, σχετική και καλά τεκμηριωμένη απάντηση βασισμένη μόνο στο παρεχόμενο context.

    response = llm.invoke(prompt)# στέλνει το prompt στο LLM για να πάρει την απάντηση. Το LLM θα επεξεργαστεί το prompt, θα χρησιμοποιήσει το context για να απαντήσει στην ερώτηση, και θα επιστρέψει την απάντηση. Η απάντηση θα πρέπει να ακολουθεί τους κανόνες που ορίσαμε στο prompt, όπως να χρησιμοποιεί μόνο το context, να παραθέτει τις πηγές, και να δίνει μια ακριβή και σχετική απάντηση.

    return response.content


# 👇 ΜΟΝΟ για CLI χρήση
if __name__ == "__main__": 

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\n===== ANSWER =====")
        print(answer)
