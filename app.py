from flask import Flask, request, jsonify, render_template # εισάγει τις απαραίτητες κλάσεις και συναρτήσεις από τη βιβλιοθήκη Flask για να δημιουργήσει μια web εφαρμογή. Το Flask είναι ένα μικρό και ευέλικτο web framework για Python που μας επιτρέπει να δημιουργήσουμε εύκολα web εφαρμογές και APIs.
from rag import ask_question
from flask import send_from_directory
import os
import markdown

DATA_FOLDER = "data"
app = Flask(__name__)

@app.route("/pdfs") # δημιουργεί ένα route "/pdfs" που επιστρέφει μια λίστα με τα ονόματα των PDF αρχείων που βρίσκονται στον φάκελο "data". Αυτή η λίστα θα χρησιμοποιηθεί στο frontend για να εμφανίσει τα διαθέσιμα PDF αρχεία που μπορεί να επιλέξει ο χρήστης για να δει το περιεχόμενό τους.
def list_pdfs():

    files = [
        f for f in os.listdir(DATA_FOLDER) # διαβάζει τα ονόματα των αρχείων που βρίσκονται στον φάκελο "data" και κρατάει μόνο αυτά που τελειώνουν με ".pdf". Αυτή η λίστα με τα ονόματα των PDF αρχείων θα επιστραφεί ως JSON στο frontend για να εμφανιστεί στον χρήστη.
        if f.endswith(".pdf")
    ]

    return jsonify(files)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]

    answer = ask_question(question)   
    formatted = markdown.markdown(answer) # μετατρέπει την απάντηση που επιστρέφει η συνάρτηση ask_question σε μορφή HTML χρησιμοποιώντας τη βιβλιοθήκη markdown. Αυτό επιτρέπει να εμφανιστεί η απάντηση με σωστή μορφοποίηση στο frontend, όπως έντονα γράμματα, λίστες, κλπ. 


    return jsonify({"answer": formatted}) # επιστρέφει την απάντηση σε μορφή JSON στο frontend, όπου το κλειδί "answer" περιέχει την απάντηση που έχει μετατραπεί σε HTML. Το frontend μπορεί να χρησιμοποιήσει αυτή την απάντηση για να εμφανίσει την απάντηση στον χρήστη με σωστή μορφοποίηση.   

@app.route("/pdf/<filename>") # δημιουργεί ένα route "/pdf/<filename>" που επιτρέπει στο frontend να ζητήσει την προβολή ενός συγκεκριμένου PDF αρχείου. Το <filename> είναι μια μεταβλητή που θα αντικατασταθεί με το όνομα του PDF αρχείου που θέλει να δει ο χρήστης. Όταν ο χρήστης επιλέξει ένα PDF αρχείο από τη λίστα, το frontend θα στείλει ένα αίτημα σε αυτό το route με το όνομα του αρχείου, και η συνάρτηση serve_pdf θα επιστρέψει το περιεχόμενο του PDF αρχείου για να εμφανιστεί στον χρήστη.
def serve_pdf(filename):
    return send_from_directory("data", filename)


if __name__ == "__main__":
    app.run(debug=True)
