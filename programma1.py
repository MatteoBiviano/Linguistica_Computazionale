# encoding: utf-8
import sys
import codecs
import nltk
from nltk import FreqDist
from nltk import bigrams
from collections import Counter
import math
#Lista globale della punteggiatura che non va considerata
punteggiatura=["/","'", "!",";", ",",":","?","-","=",".","\""]

#Lista globale dei placeholder dei sostantivi
sostantivi = ["NN", "NNS", "NNP", "NNPS"]
#Lista globale dei placeholder dei verbi
verbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
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

#La funzione calcola le medie richieste dal punto 2
def medie(frasi_file, list_tokens):

    #Calcolo la media delle frasi

    #Coppia in cui nel primo elemento inseriamo delle lunghezze dei tokens
    # e nel secondo inseriamo il numero di frasi
    temp = (0,0)
    #Per ogni frase nella lista delle frasi
    for frase in frasi_file:
        #Lista di tokens nella frase
        tokens = nltk.word_tokenize(frase)
        n_tokens = len(tokens)
        #Incremento il numero di tokens
        #Incremento il numero di frasi di 1
        temp = (temp[0]+n_tokens, temp[1] +1)
    #Calcolo la media
    media_frasi = float(temp[0]) / float(temp[1])

    # Adesso calcolo la media delle parole 

    #Inserisco nel primo elemento la somma delle lunghezze dei tokens
    # e nel secondo elemento il numero di parole
    temp = (0,0)
    for token in list_tokens:
        #Escludo la punteggiatura
        if token not in punteggiatura:
            #Aggiorno la lunghezza dei tokens e il numero di parole
            len_token = len(token)
            temp = (temp[0] + len_token, temp[1] + 1)
    #Calcolo la media
    media_parole = float(temp[0]) / float(temp[1])
    return media_frasi, media_parole

#Funzione che stampa la distribuzione degli hapax all'aumentare del corpus
# per porzioni incrementali
def hapax_incr(list_tokens):
    #For per calcolare le porzioni incrementali di 1000
    for porzione in range(1000,len(list_tokens),1000):
        tot_hapax = 0
        #lista che viene riempita di volta in volta con il numero di tokens che mi serve
        lista_porzioni_incrementali = []
        for i in range(0, porzione):
            lista_porzioni_incrementali.append(list_tokens[i])
        #controllo la frequenza del token e verifico che sia un hapax
        lista_frequenza = Counter(lista_porzioni_incrementali).items()
        for i in lista_frequenza:
            if i[1]== 1:
                tot_hapax = tot_hapax+1
        vocabolario = len(set(lista_porzioni_incrementali))
        print("Per porzioni incrementali di " + str(porzione)+" il libro ha un vocabolario composto da "+ str(vocabolario)+" parole tipo e "+ str(tot_hapax)+" hapax")

#Funzione che calcola il rapporto tra Sostantivi e Verbi
def rapportoSV(list_tokens_annotati):
    #Coppia il cui primo elemento conta i sostantivi
    # mentre il secondo conta i verbi
    conta = (0, 0)
    for _, annotazione in list_tokens_annotati:
        #Controllo se il token è annotato come sostantivo
        if annotazione in sostantivi:
            #incremento il contatore dei sostantivi
            conta = (conta[0] + 1, conta[1])
        #controllo se il token è un verbo
        if annotazione in verbi:
            #incremento il contatore dei verbi
            conta = (conta[0], conta[1] + 1)
    #restituisco il rapporto tra sostantivi e verbi        
    return (float(conta[0])/float(conta[1]))	

#Funzione che calcola le 10 PoS più frequenti
# e restituisce la lista delle annotazioni
def pos_10(list_tokens_annotati):
    #Lista vuota delle annotazioni
    list_annotazioni = []
    for _,annotazione in list_tokens_annotati:
        #Aggiungo alla lista delle annotazione
        list_annotazioni.append(annotazione)
    pos_distribuzione = FreqDist(list_annotazioni)
    #Salvo la lista delle 10 PoS più frequenti usando il metodo "most_common"
    pos_ordinati = pos_distribuzione.most_common(10)
    dim_pos = len(pos_ordinati)
    for i in range(0, dim_pos):
        print("Part of Speech: " + str(pos_ordinati[i][0]) + " - frequenza: "+ str(pos_ordinati[i][1]))
    return list_annotazioni


#Funzione che calcola i 10 bigrammi con probabilità condizionata, forza associativa massima
def maximization(lista_annotazioni, list_bigrammi, set_bigrammi):
    bigrammi_prob_max = []
    bigrammi_forza_max = []
    #Voglio trovare 10 bigrammi
    for i in range(0, 10):
        #Probabilità massima considerata fin'ora
        probabilita = 0.0        
        forza = 0.0
        lista_pos_prob = []
        lista_pos_forz = []
        for bigramma,_ in bigrammi_prob_max:
            #Inserisco i bigrammi in lista
            lista_pos_prob.append(bigramma)
        for bigramma,_ in bigrammi_forza_max:
            #Inserisco i bigrammi in lista
            lista_pos_forz.append(bigramma)

        for bigramma in set_bigrammi:
            #Escludo bigrammi già trovati
            if bigramma not in lista_pos_prob:
                #Calcolo la frequenza del bigramma selezionato
                frequenza_bigramma = list_bigrammi.count(bigramma)
                #Calcolo la frequenza nel testo
                frequenza_in_testo = lista_annotazioni.count(bigramma[0])
                #Probabilità condizionata dividendo le due frequenze
                prob_cond = float(frequenza_bigramma)/float(frequenza_in_testo)
                #Controllo se ho ottenuto una probabilità condizionale maggiore di quella salvata
                if prob_cond > probabilita:
                    #Aggiorno la probabilità massima
                    probabilità = prob_cond
                    massimo_bigramma_prob = bigramma
            #Escludo bigrammi già trovati
            if bigramma not in lista_pos_forz:
                #Calcolo la frequenza del bigramma selezionato
                frequenza_bigramma = list_bigrammi.count(bigramma)
                #Calcolo la frequenza nel testo del primo elemento
                frequenza_in_testo_1 = lista_annotazioni.count(bigramma[0])
                #Calcolo la frequenza nel testo del secondo elemento
                frequenza_in_testo_2 = lista_annotazioni.count(bigramma[1])
                #Calcolo la LMI - con formula
                mult_freq_bigramma= frequenza_bigramma*len(list_bigrammi)
                mult_freq_testo=frequenza_in_testo_1*frequenza_in_testo_2
                log_freq=math.log((mult_freq_bigramma/mult_freq_testo),2)
                forza_ass = frequenza_bigramma*log_freq
                #Controllo se ho ottenuto una forza associativa maggiore di quella salvata
                if forza_ass > forza:
                    #Aggiorno la probabilità massima
                    forza = forza_ass
                    massimo_bigramma_forz = bigramma            
        #Inserisco il bigramma con probabilità massima
        bigrammi_prob_max.append((massimo_bigramma_prob, probabilita))
        bigrammi_forza_max.append((massimo_bigramma_forz, forza))
    return bigrammi_prob_max, bigrammi_forza_max

#Funzione che estrae i 10 bigrammi con probabilità condizionata massima
# e con forza associativa massima
def bigrammi(lista_annotazioni):
    #Lista dei bigrammi
    list_bigrammi = list(bigrams(lista_annotazioni))
    #Insieme dei bigrammi distinti
    set_bigrammi = set(list_bigrammi)

    bigrammi_prob_max, bigrammi_forza_max = maximization(lista_annotazioni, list_bigrammi, set_bigrammi)

    return bigrammi_prob_max, bigrammi_forza_max

def main():
    print("Progetto di Linguistica computazionale, a.a. 2019/2020\n Author: Clara Casandra, 566830\n PROGRAMMA 1\n")

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

    #Divido i files in tokens e ne salvo la dimensione
    list_tokens_1 = nltk.word_tokenize(raw_1)
    n_tokens_1 = len(list_tokens_1)
    list_tokens_2 = nltk.word_tokenize(raw_2)
    n_tokens_2 = len(list_tokens_2)


    print("*** Punto 1 ***")
    print("Numero totale di frasi: ")
    print("File 1: " + str(n_frasi_1))
    print("File 2: " + str(n_frasi_2))
    
    print("Numero totale di tokens: ")
    print("File 1: " + str(n_tokens_1))
    print("File 2: " + str(n_tokens_2))
    print("________________________________________")

    #Chiamo la funzione che calcola la media delle frasi e delle parole
    media_frasi_1, media_parole_1 = medie(frasi_file_1, list_tokens_1)
    media_frasi_2, media_parole_2 = medie(frasi_file_2, list_tokens_2)
    
    print("*** Punto 2 ***")
    print("Lunghezza media delle frasi in termini di tokens: ")
    print("File: " + str(media_frasi_1))
    print("File 2: " + str(media_frasi_2))
    
    print("Lunghezza media delle parole in termini di caratteri: ")
    print("File 1: " + str(media_parole_1))
    print("File 2: " + str(media_parole_2))
    print("________________________________________")

    #Calcolo il numero di parole tipo
    len_vocabolario_1 = len(set(list_tokens_1))
    len_vocabolario_2 = len(set(list_tokens_2))

    print("*** Punto 3 ***")
    print("Grandezza del vocabolario: ")
    print("File 1: " + str(len_vocabolario_1))
    print("File 2: " + str(len_vocabolario_2))
    print("Distribuzione degli hapax all'aumentare del corpus")
    print("File 1: ")
    print("*** **** *** *** ")
    #Funzione che calcola la distribuzione degli hapax
    hapax_incr(list_tokens_1)
    print("File 2: ")
    hapax_incr(list_tokens_2)
    print("________________________________________")

    #Annotazione del testo
    tokens_annotati_1 = nltk.pos_tag(list_tokens_1)
    tokens_annotati_2 = nltk.pos_tag(list_tokens_2)


    #Calcolo il rapporto tra sostantivi e verbi
    r_file_1 = rapportoSV(tokens_annotati_1)
    r_file_2 = rapportoSV(tokens_annotati_2)

    print("*** Punto 4 ***")
    print("Rapporto tra Sostantivi e verbi: ")
    print("File 1: " + str(r_file_1))
    print("File 2: " + str(r_file_2))
    print("________________________________________")


    print("*** Punto 5 ***")
    print("10 PoS più frequenti: ")
    print("File 1: ")
    list_pos_1 = pos_10(tokens_annotati_1)
    print("File 2: ")
    list_pos_2 = pos_10(tokens_annotati_2)

    print("________________________________________")

    #Bigrammi con probabilità condizionata massima dei due testi
    bigrammi_prob_max_1, bigrammi_forza_max_1 = bigrammi(tokens_annotati_1)
    bigrammi_prob_max_2, bigrammi_forza_max_2 = bigrammi(tokens_annotati_2)
    print("*** Punto 6 ***")
    print("Estrarre ed ordinare i 10 bigrammi PoS con probabilità condizionata massima: ")
    print("File 1: ")
    for i in range(0, len(bigrammi_prob_max_1)):
        print("Bigramma: " + str(bigrammi_prob_max_1[i][0]) + " - Probabilità: " + str(bigrammi_prob_max_1[i][1]))
    print("File 2: ")
    for i in range(0, len(bigrammi_prob_max_2)):
        print("Bigramma: " + str(bigrammi_prob_max_2[i][0]) + " - Probabilità: " + str(bigrammi_prob_max_2[i][1]))

    print("Estrarre ed ordinare i 10 bigrammi PoS con forza associativa massima: ")
    print("File 1: ")
    for i in range(0, len(bigrammi_forza_max_1)):
        print("Bigramma: " + str(bigrammi_forza_max_1[i][0]) + " - Probabilità: " + str(bigrammi_forza_max_1[i][1]))
    print("File 2: ")
    for i in range(0, len(bigrammi_forza_max_2)):
        print("Bigramma: " + str(bigrammi_forza_max_2[i][0]) + " - Probabilità: " + str(bigrammi_forza_max_2[i][1]))


#Controllo che siano stati passati due file


#richiamo la funzione main
main()