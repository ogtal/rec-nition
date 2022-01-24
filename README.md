# Rec&nition
Dette repository indeholder kode og modelvægtene til *Rec&nition* (Recognition) algortimen. Den er udviklet af [Analyse og Tal F.M.B.A.](www.ogtal.dk) med støtte fra TryghedsFonden. 

Algoritmen er designet til at klassificere små tekststykker efter om de indeholder anerkendende sprog. Den er blevet brugt til at finde anerkendennde sprog i den offentlige debat, et arbejde som man kan læse om [her](https://www.tryghed.dk/viden/publikationer/trivsel/anerkendelse-i-den-offentlige-debat-paa-facebook). 

## Definitioner af anerkendende sprog

Algortimen er en binær klassifikationsalgortime der vurdere om et kort tekststykke indholdende anerkendende sprog eller ej. Definitionen af anerkendelse kan læses i filen *definitioner.pdf*.

## Beskrivelse af algoritmen

Algortimen er udviklet vha. et annoteret datasæt med 67.188 tekststykker. Datasættet indeholder 14.911 eksempler på anerkendelse og 52.913 eksempler på tekststykker uden anerkendelse. Teksstykkerne er kommentarer og svar afgivet på opslag i en række offentlige Facebook Pages og større grupper. Datasættet er opdelt i et træningsdatasæt (70 %), et evalueringsdatasæt (20 %) og et testdatasæt (10 %).  

Trænings- og evalueringsdatasættet blev brugt til at træne og udvælge den bedste kombination af algoritmearkitektur og hyperparametre. Til det brugte vi den højest macro average F1 score. Efter udvælgelsen af den bedste algoritme blev denne testet på testdatasættet. 

Den bedste model bruger en [dansk electra model](https://huggingface.co/Maltehb/-l-ctra-danish-electra-small-uncased#) som sprogmodel og har et feed forward lag til selve klassificeringen. Se modeldefinitionen i filen `model_def.py`. 

## Resultater
		
Resultaterne for algoritmen på evalueringsdatasættet er: 
 - Macro averace F1 score: 0.8068
 - Precision: 0.8182 
 - Recall: 0.7970 
 - Confusion matrix:

|         | Annoteret ikke anerkendelse | Annoteret anerkendelse  |
| ------------- |:-------------:| :-----:|
| **Ikke anerkendelse iflg. Rec&nition** | 9858 | 732 |
| **Anerkendelse iflg. Rec&nition**      | 997  | 1962 |


Og for testdatasættet:
 - Macro averace F1 score: 0.7510
 - Precision: 0.7340  
 - Recall: 0.7690
 - Confusion matrix:

|         | Annoteret ikke-hadfuldt sprog | Annoteret hadfuldt sprog  |
| ------------- |:-------------:| :-----:|
| **Ikke-hadfuldt sprog iflg. Ha&te** | 4238 | 830 |
| **Hadfuldt sprog iflg. Ha&te**      | 462  | 1084 |


## Brug af algoritmen

For at gør brug af algortimen skal der først installeres *transformers* og *pytorch*:
```bash
pip install torch
pip install transformers

```

Derefter kan modellen bruges tests på enkelte tekststykker ved at køre følgende:

```python
import torch
from transformers import AutoTokenizer
from model_def import ElectraClassifier

text = "Du har en rigtig god pointe"

def load_model():
    model_checkpoint = 'Maltehb/aelaectra-danish-electra-small-cased
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    model = ElectraClassifier(model_checkpoint,2)
    model_path = 'pytorch_model.bin'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    return(model, tokenizer)

def make_prediction(text):
    tokenized_text = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )
    input_ids = tokenized_text['input_ids']
    attention_masks = tokenized_text['attention_mask']
    logits = model(input_ids,attention_masks)
    
    logit,preds = torch.max(logits, dim=1)
    return(int(preds))

model, tokenizer = load_model()
text_clf = make_prediction(text)
```
Hvor *make_predition* returnere klasse 0 hvis teksten vurderes til ikke at indeholde anerkendelse og klassen 1 hvis teksten vurderes at indeholde anerkendelse. 

Funktionerne i *data_prep.py* kan bruges til at lave batch inferens. 

## Kontakt

Spørgsmål til indeholder af dette repository kan sendes til:
 - Ronnie Taarnborg (ronnie@ogtal.dk)
 - Edin Lind Ikanovic (edin@ogtal.dk)

## Tak til:

**Projektets annotører**
 - Ida Marcher Manøe
 - Julie Enevoldsen
 - Nikolaj Meldgaard Christensen
 - Naja Bau Nielsen

**Projektets advisory board**
 - Andreas Birkbak, Associat Professor, TANTlab, AAU 
 - Bolette Sandford Pedersen, Professor, CenterforSprogteknologi, KU
 - Leon Derczynski, Associate Professor, ComputerScience, ITU
 - Marianne Rathje, Seniorforsker, Dansk Sprognævn
 - Michael Bang Petersen, Professor, Institut for Statskundskab, AU
 - Rasmus Rønlev, Adjunkt, Center for Journalistik, SDU

**Vores samarbejdspartnere hos TrygFonden**
 - Anders Hede, Forskningschef
 - Christoffer Elbrønd, Projektchef
 - Christian Nørr, Dokumentarist
 - Peter Pilegaard Hansen, Projektchef

**Danske Open-source teknologi vi står på skuldrene af**
 - The Danish Gigaword Project: Leon Derczynski, Manuel R. Ciosici
 - Ælectra: Malte Højmark-Bertelsen
