# encoding: utf-8
import sys
import codecs
import nltk
from nltk import FreqDist
from nltk import bigrams
from collections import Counter
import math
import re

#Variabile usate per i controlli:
person = "PERSON"
luogo = "GPE"
nomi = ["NN", "NNS", "NNP", "NNPS"]
verbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
#Utilizzata per avere le date
regex_date = r"\b([0-1]?\d)[-/]([0-3]?\d)[-/][0-9]+|([0-3]?\d)[-/]([0-1]?\d)[-/][0-9]+|[0-9]+[-/]([0-1]?\d)[-/]([0-3]?\d)\b"
regex_giorni = r"\b(?:Sun|Mon|Tues|Wednes|Thurs|Fri|Satur)day\b"
regex_mesi = r"\b(?:Janu|Febru)ary|March|April|May|June|July|August|(?:Septem|Octo|Novem|Decem)ber\b"
#Lunghezza minimo token
MIN_LEN = 8
#Lunghezza massima token
MAX_LEN = 12

#Funzione utilizzata per indicare come eseguire il file
def usage():
    print("Eseguire il progetto nel seguente modo: ")
    print("$ python programma1.py alice.txt peter.txt")

#Funzione che inizializza le variabili utili nel progetto
def init(input1, input2):
    #Apro i file
    file_1 = codecs.open(input1, "r", "utf-8")
    file_2 = codecs.open(input2, "r", "utf-8")	

    #Leggo i file in una variabile
    raw_1 = file_1.read()
    raw_2 = file_2.read()

    #Carico il modello statistico
    tokenizer = nltk.data.load('tokenizers\punkt\english.pickle')

    return raw_1, raw_2, tokenizer

#Funzione che trova i 10 nomi più frequenti
def top10Nomi(list_tokens_annotati):
	#Lista dei nomi
	list_nomi = []
	for tokens in list_tokens_annotati:
		list_ne_chunk = nltk.ne_chunk(tokens)
		for node in list_ne_chunk:
			NE = ""
			#controllo se il nodo è un nodo intermedio
			if hasattr(node, 'label'):
				if node.label()==person:
					#Ciclo sulle foglie del nodo se era intermedio
					for part_NE in node.leaves():
						NE = NE + '' + part_NE[0]
					list_nomi.append(NE)
	top_nomi = nltk.FreqDist(list_nomi).most_common(10)
	return top_nomi

#Funzione che dai top 10 nomi estrapola le frasi
def frasi_from_nomi(frase, list_nomi):
    list_frasi_nomi = []
    list_ne_chunk = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(frase)))
    for node in list_ne_chunk:
        NE = ""
        #controllo se il nodo è un nodo intermedio
        if hasattr(node, 'label'):
            if node.label()==person:
                #Ciclo sulle foglie del nodo se era intermedio
                for part_NE in node.leaves():
                    NE = NE + '' + part_NE[0]
                    #Controllo se è nella lista dei nomi
                    if NE in list_nomi:
                        list_frasi_nomi.append((frase, NE))
    return list_frasi_nomi

#Trova i 10 luoghi, nomi, verbi, sostantivi più frequenti
def top_10_all(list_frasi_nomi):
    list_frasi = []
    for frase, _ in list_frasi_nomi:
        list_frasi.append(frase)
    #Insieme delle frasi distinte per i top 10 nomi
    set_frasi = set(list_frasi)
    
    list_nomi = []
    list_luoghi = []
    list_sostantivi = []
    list_verbi = []

    for frase in set_frasi:
        #tokenizzo la frase e trovo i pos tag per ogni token
        list_token_pos = nltk.pos_tag(nltk.word_tokenize(frase))
        list_ne_chunk = nltk.ne_chunk(list_token_pos)
        for node in list_ne_chunk:
            NE_nome = ''
            NE_luogo = ''
            #controllo se il nodo è un nodo intermedio
            if hasattr(node, 'label'):
                if node.label()==person:
                    #Ciclo sulle foglie del nodo se era intermedio
                    for part_NE in node.leaves():
                        NE_nome = NE_nome + '' + part_NE[0]
                    list_nomi.append(NE_nome)
                if node.label()==luogo:
                    #Ciclo sulle foglie del nodo se era intermedio
                    for part_NE in node.leaves():
                        NE_luogo = NE_luogo + '' + part_NE[0]
                    list_luoghi.append(NE_luogo)
        for token, pos in list_token_pos:
            if pos in nomi:
                list_sostantivi.append(token)
            if pos in verbi:
                list_verbi.append(token)
    top_nomi = nltk.FreqDist(list_nomi).most_common(10)
    top_luoghi = nltk.FreqDist(list_luoghi).most_common(10)
    top_sostantivi = nltk.FreqDist(list_sostantivi).most_common(10)
    top_verbi = nltk.FreqDist(list_verbi).most_common(10)

    return top_nomi, top_luoghi, top_sostantivi, top_verbi, set_frasi

def prob_markov(frequenze, len_corpus, set_frasi):
	proba_tmp = 1.0
	probab_massima = 0

	for frase in set_frasi:
		tokens = nltk.word_tokenize(frase)
		#Se la lunghezza è compresa tra 8 e 12
		if (len(tokens) >= MIN_LEN) and (len(tokens) <= MAX_LEN):
			for token in tokens:
				probabilita = float(frequenze[token])/float(len_corpus)
				proba_tmp  = proba_tmp * probabilita
			if proba_tmp > probab_massima:
				tmp_frase = frase
				probab_massima = proba_tmp
	return tmp_frase, probab_massima

def main():
    print("Progetto di Linguistica computazionale, a.a. 2019/2020\n Author: Clara Casandra, 566830\n PROGRAMMA 2\n")

    #Controllo che siano stati passati i due file di input
    if len(sys.argv) != 3:
        usage()
    raw_1, raw_2, sent_tokenizer = init(sys.argv[1], sys.argv[2])

    #Divido i due file letti in due frasi
    frasi_file_1 = sent_tokenizer.tokenize(raw_1)
    frasi_file_2 = sent_tokenizer.tokenize(raw_2)

    #Numero di frasi per ogni file
    n_frasi_1 = len(frasi_file_1)
    n_frasi_2 = len(frasi_file_2)

    #Lista di tokens divisa per frasi
    # e trovo i pos_tag per ogni token
    list_list_tokens_1 = []
    list_tokens_annotati_1 = []
    for frase in frasi_file_1:
    	tokens = nltk.word_tokenize(frase)
    	list_list_tokens_1 = list_list_tokens_1 + tokens
    	annotato = nltk.pos_tag(tokens)
    	list_tokens_annotati_1.append(annotato)
    #Lunghezza corpus
    len_corpus_1 = len(list_list_tokens_1)

    list_list_tokens_2 = []
    list_tokens_annotati_2 = []
    for frase in frasi_file_2:
        tokens = nltk.word_tokenize(frase)
        list_list_tokens_2 = list_list_tokens_2+ tokens
        annotato = nltk.pos_tag(tokens)
        list_tokens_annotati_2.append(annotato)
    #Lunghezza corpus
    len_corpus_2 = len(list_list_tokens_2)

    #Liste con i 10 nomi e frequenze
    top_10_nomi_freq_1 = top10Nomi(list_tokens_annotati_1)
    top_10_nomi_freq_2 = top10Nomi(list_tokens_annotati_2)

    #Estrapoliamo la lista dei 10 nomi più frequenti
    top_nomi_1 = []
    for nome, frequenza in top_10_nomi_freq_1:
        top_nomi_1.append(nome)
    top_nomi_2 = []
    for nome, frequenza in top_10_nomi_freq_2:
        top_nomi_2.append(nome)


    #Costruisco le liste di coppie frasi-nomi, con i 10-top nomi
    list_frasi_nomi_1 = []
    for frase in frasi_file_1:
        list_frasi_nomi_1 =  list_frasi_nomi_1 + frasi_from_nomi(frase, top_nomi_1)

    list_frasi_nomi_2 = []
    for frase in frasi_file_2:
        list_frasi_nomi_2 = list_frasi_nomi_2 + frasi_from_nomi(frase, top_nomi_2)


    print("*** Punto 1 ***")
    print("I 10 nomi propri più frequenti:")
    print("File 1: ")
    for nome,frequenza in top_10_nomi_freq_1:
        print("Nome: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("File 2: ")
    for nome,frequenza in top_10_nomi_freq_2:
        print("Nome: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("________________________________________")

    print("Frasi più lunghe e più brevi che contengono i top 10 nomi: ")
    print("File 1: ")
    for nome_top in top_nomi_1:
        
        #Coppia che contiene la frase più corta e la frase più lunga
        frasi_min_max = ('', '')
        for frase, nome in list_frasi_nomi_1:
            if nome == nome_top:
                #Se non ho ancora trovato nessuna frase
                if len(frasi_min_max[0]) == 0:
                    frasi_min_max = (frase, frase)
                else:
                    #Se trovo una frase più corta della frase più corta trovata 
                    if len(frasi_min_max[0]) > len(frase):
                        frasi_min_max = (frase, frasi_min_max[1])
                    else:
                        if len(frasi_min_max[1]) < len(frase):
                            frasi_min_max = (frasi_min_max[0], frase)
        print("Nome: " + str(nome_top))
        print("---------------- Frase più breve ----------------")
        print(frasi_min_max[0])
        print("---------------- Frase più lunga ----------------")
        print(frasi_min_max[1])
        print("____________________________________________________________")
    print("File 2: ")
    for nome_top in top_nomi_2:
        #Coppia che contiene la frase più corta e la frase più lunga
        frasi_min_max = ('', '')
        for frase, nome in list_frasi_nomi_2:
            if nome == nome_top:
                #Se non ho ancora trovato nessuna frase
                if len(frasi_min_max[0]) == 0:
                    frasi_min_max = (frase, frase)
                else:
                    #Se trovo una frase più corta della frase più corta trovata 
                    if len(frasi_min_max[0]) > len(frase):
                        frasi_min_max = (frase, frasi_min_max[1])
                    else:
                        if len(frasi_min_max[1]) < len(frase):
                            frasi_min_max = (frasi_min_max[0], frase)
        print("Nome: " + str(nome_top))
        print("---------------- Frase più breve ----------------")
        print(frasi_min_max[0])
        print("---------------- Frase più lunga ----------------")
        print(frasi_min_max[1])
        print("____________________________________________________________")

    top_10_persone_1, top_10_luoghi_1, top_10_sostantivi_1, top_10_verbi_1, set_frasi_1 = top_10_all(list_frasi_nomi_1)
    top_10_persone_2, top_10_luoghi_2, top_10_sostantivi_2, top_10_verbi_2, set_frasi_2  = top_10_all(list_frasi_nomi_2)

    print("*** Punto 2.1 ***")
    print("I 10 luoghi più frequenti per nome proprio:")
    print("File 1: ")
    for nome, frequenza in top_10_luoghi_1:
        print("Luogo: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("File 2: ")
    for nome,frequenza in top_10_luoghi_2:
        print("Luogo: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("________________________________________")

    print("*** Punto 2.2 ***")
    print("I 10 persone più frequenti per nome proprio:")
    print("File 1: ")
    for nome, frequenza in top_10_persone_1:
        print("Persona: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("File 2: ")
    for nome,frequenza in top_10_persone_1:
        print("Persona: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("________________________________________")

    print("*** Punto 2.3 ***")
    print("I 10 sostantivi più frequenti:")
    print("File 1: ")
    for nome,frequenza in top_10_sostantivi_1:
        print("Sostantivo: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("File 2: ")
    for nome,frequenza in top_10_sostantivi_2:
        print("Sostantivo: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("________________________________________")

    print("*** Punto 2.4 ***")
    print("I 10 verbi più frequenti:")
    print("File 1: ")
    for nome,frequenza in top_10_verbi_1:
        print("Verbo: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("File 2: ")
    for nome,frequenza in top_10_verbi_2:
        print("Verbo: " + str(nome) + " - Frequenza: " + str(frequenza))
    print("________________________________________")

    #Acquisiamo date, mesi e giorni
    list_date = []
    list_mesi = []
    list_giorni = []

    for frase in set_frasi_1:
    	list_date = list_date + re.findall(regex_date,frase)
    	list_mesi = list_mesi + re.findall(regex_mesi,frase)
    	list_giorni = list_giorni + re.findall(regex_giorni,frase)

    top_mesi_1 = nltk.FreqDist(list_mesi).most_common()
    top_date_1 = nltk.FreqDist(list_date).most_common()
    top_giorni_1 =nltk.FreqDist(list_giorni).most_common()


    print("*** Punto 2.5 ***")
    print("Le Date, i Mesi e i Giorni più frequenti:")
    print("File 1: ")
    for data,frequenza in top_date_1:
        print("Data: " + str(data) + " - Frequenza: " + str(frequenza))
    for mese,frequenza in top_mesi_1:
        print("Mese: " + str(mese) + " - Frequenza: " + str(frequenza))
    for giorno,frequenza in top_giorni_1:
        print("Giorno: " + str(giorno) + " - Frequenza: " + str(frequenza))
    #Acquisiamo date, mesi e giorni
    list_date = []
    list_mesi = []
    list_giorni = []

    for frase in set_frasi_2:
    	list_date = list_date + re.findall(regex_date,frase)
    	list_mesi = list_mesi + re.findall(regex_mesi,frase)
    	list_giorni = list_giorni + re.findall(regex_giorni,frase)

    top_mesi_2 = nltk.FreqDist(list_mesi).most_common()
    top_date_2 = nltk.FreqDist(list_date).most_common()
    top_giorni_2 =nltk.FreqDist(list_giorni).most_common()
    print("File 2: ")
    for data,frequenza in top_date_2:
        print("Data: " + str(data) + " - Frequenza: " + str(frequenza))
    for mese,frequenza in top_mesi_2:
        print("Mese: " + str(mese) + " - Frequenza: " + str(frequenza))
    for giorno,frequenza in top_giorni_2:
        print("Giorno: " + str(giorno) + " - Frequenza: " + str(frequenza))

    #La frase lunga minimo 8 token e massimo 12 con probabilità più alta

    print("*** Punto 2.6 ***")
    print("La frase lunga minimo 8 token e massimo 12 con probabilità più alta:")
    print("File 1: ")
    frase_max, proba_max = prob_markov(nltk.FreqDist(list_list_tokens_1), len_corpus_1, set_frasi_1)
    print("Frase: " + str(frase_max) + " - Probabilità: " + str(proba_max))
    print("File 2: ")
    frase_max, proba_max = prob_markov(nltk.FreqDist(list_list_tokens_2), len_corpus_2, set_frasi_2)
    print("Frase: " + str(frase_max) + " - Probabilità: " + str(proba_max))

main()