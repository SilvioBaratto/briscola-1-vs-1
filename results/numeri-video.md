# Numeri per il video: agente PPO contro gemma4:31b-cloud

> Dati completi: 200 partite su 200, tutte giocate e salvate.

L'agente è `checkpoints/final_model.pt` (rete PPO già addestrata, mossa sempre la più probabile). L'avversario è `gemma4:31b-cloud` servito da Ollama e chiamato solo con LangChain, temperatura 0,1. Il mazzo della partita *i* si mescola con il seme 42 + *i*, e l'agente apre le partite pari, il modello le dispari: 100 partite aperte dall'agente e 100 dal modello. Al modello arriva solo il prompt della mossa (briscola, carta dell'avversario, carte già uscite, mano): le regole del gioco non gli vengono spiegate.

Ogni numero ha accanto il comando che lo ricalcola dai file in `results/partite/`. Rilanciando la valutazione i mazzi e chi apre sono gli stessi, ma il modello è un servizio remoto e può rispondere in modo un po' diverso: le mosse, e quindi i punteggi, possono cambiare di poco.

## In una frase

L'agente addestrato perde contro il modello: 75 vittorie contro 121 sconfitte (37,5% di vittorie, intervallo al 95% da 31,1% a 44,4%).

## I numeri

### 1. Chi ha vinto

- **75 vittorie, 121 sconfitte e 4 pareggi** dell'agente, su 200 partite. Un pareggio è 60 a 60.
- In parole povere: ogni 100 partite l'agente ne vince circa 38 e ne perde circa 60.
- Si ricalcola con: `python -m training.numeri_video --mostra risultato`

### 2. Percentuale di vittorie e margine di errore

- **37,5% di vittorie** (vittorie / partite, i pareggi non contano come vittorie), con un **intervallo di confidenza al 95% da 31,1% a 44,4%**.
- In parole povere: con 200 partite il vero valore sta, con il 95% di fiducia, in quella forbice, e il 50% (testa o croce) cade fuori, sopra la forbice: non è sfortuna, l'agente è davvero peggiore del modello.
- Metodo: intervallo di Wilson, `training/numeri_video.py:wilson_interval`.
- Si ricalcola con: `python -m training.numeri_video --mostra percentuale-vittorie`

### 3. Punti medi e margine

- **Agente 53,1 punti, modello 66,9 punti**, margine **-13,8** per l'agente (su 120 punti in palio, per vincere ne servono 61).
- In parole povere: in una partita media il modello prende circa 14 punti più dell'agente.
- Si ricalcola con: `python -m training.numeri_video --mostra punti-medi`

### 4. Aprire cambia qualcosa?

- **Apre l'agente:** 36 vittorie, 63 sconfitte e 1 pareggi su 100 partite (36,0% di vittorie); punti medi 54,3 a 65,7.
- **Apre il modello:** 39 vittorie, 58 sconfitte e 3 pareggi su 100 partite (39,0% di vittorie); punti medi 51,8 a 68,2.
- **Differenza:** -3,0 punti percentuali per l'agente quando apre, intervallo al 95% da -16,1 a 10,3.
- In parole povere: lo zero cade dentro la forbice: con queste partite non si può dire che aprire sia un vantaggio né uno svantaggio.
- Si ricalcola con: `python -m training.numeri_video --mostra chi-apre`

### 5. La mossa in cui il modello perde più punti

- Carte con punti: *carico* = asso o tre (11 e 10 punti), *figura* = re, cavallo o fante (4, 3 e 2 punti).
- **La più frequente:** rispondere con una figura e perdere la mano: **275 volte**, 838 punti regalati all'agente.
- **La più costosa:** rispondere con un carico e perdere la mano: 175 volte, **1844 punti** regalati.
- In parole povere: ogni volta che il modello gioca una carta con punti e non vince la mano, quei punti vanno all'agente. Questa tabella conta tutte quelle volte, divise per chi apre e per tipo di carta.

| Ruolo | Carta | Volte che ha perso la mano | Punti ceduti | Volte giocata |
|---|---|---:|---:|---:|
| risponde | figura | 275 | 838 | 366 |
| risponde | carico | 175 | 1844 | 271 |
| apre | figura | 153 | 456 | 481 |
| apre | carico | 103 | 1083 | 364 |
| apre | figura di briscola | 16 | 39 | 159 |
| risponde | figura di briscola | 10 | 35 | 138 |
| apre | carico di briscola | 1 | 10 | 119 |

- Si ricalcola con: `python -m training.numeri_video --mostra mosse-che-perdono-punti`

#### Quanti di questi regali erano evitabili

- Quando risponde all'apertura dell'agente, il modello butta una carta con punti e perde la mano pur avendo in mano una carta che vale meno e avrebbe perso lo stesso: **408 volte, 2325 punti** che poteva tenersi. Questa è un'euristica: non dice che sia sempre un errore, dice che il regalo non era obbligato.
- Il caso più frequente: rispondere con una figura, 247 volte (746 punti risparmiabili).

| Carta | Volte | Punti risparmiabili | Volte giocata in risposta |
|---|---:|---:|---:|
| figura | 247 | 746 | 366 |
| carico | 152 | 1549 | 271 |
| figura di briscola | 9 | 30 | 138 |

- Si ricalcola con: `python -m training.numeri_video --mostra regali-evitabili`

### 6. Risposte non valide del modello

- **0 mosse non valide su 4000** mosse del modello (nessun indice di carta utilizzabile dopo fino a 3 tentativi per mossa; i tentativi ripetuti non sono conteggiati a parte).
- In parole povere: alla fine dei tentativi il modello ha sempre dato un indice di carta valido; non abbiamo contato quante mosse hanno avuto bisogno di un secondo o terzo tentativo.
- Si ricalcola con: `python -m training.numeri_video --mostra mosse-non-valide`

### 7. Quanto ci mette a muovere

- **media 0,40 s, mediana 0,38 s, 95° percentile 0,50 s, massimo 8,26 s** per mossa del modello (4000 mosse).
- In parole povere: una partita da 20 mosse costa al modello meno di dieci secondi di attesa, ritentativi compresi.
- Si ricalcola con: `python -m training.numeri_video --mostra secondi-per-mossa`

## Cosa è contro-intuitivo

- **L'agente addestrato perde contro il modello**, e non per sfortuna: 37,5% di vittorie, e anche il limite alto della forbice (44,4%) resta sotto il 50%. Una rete addestrata apposta per questo gioco perde con un modello generico. La partita che lo mostra: partita 10 (apre l'agente): agente 41, modello 79, file `results/partite/partita_010.json`. Il caso peggiore è la partita 141 (apre il modello): agente 2, modello 118, file `results/partite/partita_141.json`.
- **Ma la fortuna pesa:** la miglior partita dell'agente è un 120-0, la partita 143 (apre il modello): agente 120, modello 0, file `results/partite/partita_143.json`. Con una sola partita si può dire tutto: servono le 200.
- **Chi apre non conta, o conta troppo poco per vederlo:** 36,0% di vittorie aprendo contro 39,0% lasciando aprire il modello, con una differenza (-3,0 punti) più piccola dell'errore. È un numero medio su 100 + 100 partite: nessuna partita singola lo mostra, vale la tabella al punto 4.
- **Il modello non dà mai una risposta inutilizzabile:** 0 mosse non valide su 4000 (dopo fino a 3 tentativi per mossa), e in media risponde in 0,40 secondi a mossa. La mossa più lenta (8,3 s) è nella partita 190 (apre l'agente): agente 34, modello 86, file `results/partite/partita_190.json`.
- **Il modello vince anche se regala punti:** 408 regali evitabili (2325 punti) nel complesso. La partita in cui ne ha regalati di più e ha vinto lo stesso: partita 120 (apre l'agente): agente 58, modello 62, file `results/partite/partita_120.json`, con 28 punti risparmiabili.

## Le due partite da mostrare

Scelte con una regola fissa, perché siano tipiche e non estreme: tra le partite vinte (e tra quelle perse) dall'agente, quella con il margine più vicino alla mediana del suo gruppo.

- Vinta dall'agente: partita 11 (apre il modello): agente 70, modello 50, file `results/partite/partita_011.json`; copia in `results/da-mostrare/partita-vinta.json`.
- Persa dall'agente: partita 10 (apre l'agente): agente 41, modello 79, file `results/partite/partita_010.json`; copia in `results/da-mostrare/partita-persa.json`.

Per vederle:

```bash
python -m visualization.render_game --replay results/da-mostrare/partita-vinta.json --output results/da-mostrare/partita-vinta.gif
python -m visualization.render_game --replay results/da-mostrare/partita-persa.json --output results/da-mostrare/partita-persa.gif
```

## Come rifare tutto

```bash
python -m training.evaluate        # gioca le partite mancanti, salva ogni partita appena finita
python -m training.numeri_video    # riscrive questo file e le due partite da mostrare
```
