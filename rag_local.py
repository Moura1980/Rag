import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


DOCUMENTOS_MEDICOS = [
    # Neurologia
    (
        "Cefaleia pulsátil unilateral (migrânea) é frequentemente acompanhada de fotofobia, "
        "fonofobia e náusea. O tratamento agudo inclui triptanos (sumatriptano 50 mg VO) e "
        "AINEs. A profilaxia é indicada quando os episódios superam 4 dias/mês."
    ),
    (
        "A cefaleia tensional episódica manifesta-se como pressão bilateral em faixa, "
        "intensidade leve a moderada, sem agravamento pela atividade física. "
        "O paracetamol 1 g ou ibuprofeno 400 mg são a primeira linha terapêutica."
    ),
    (
        "Acidente Vascular Cerebral Isquêmico (AVCi): oclusão arterial cerebral provoca déficit "
        "neurológico focal de início súbito. A janela terapêutica para trombólise IV (alteplase "
        "0,9 mg/kg) é de até 4,5 h do início dos sintomas. TC de crânio sem contraste deve ser "
        "realizada imediatamente para excluir hemorragia."
    ),
    (
        "Epilepsia focal com generalização secundária: a crise inicia em foco cortical específico "
        "e pode evoluir para crise tônico-clônica generalizada. O valproato de sódio e a "
        "carbamazepina são antiepilépticos de primeira linha."
    ),
    (
        "Vertigem posicional paroxística benigna (VPPB): sensação rotatória desencadeada por "
        "mudança de posição da cabeça, causada por otólitos deslocados no canal semicircular "
        "posterior. A manobra de Epley tem eficácia de 80-90 % na resolução dos sintomas."
    ),

    # Cardiologia
    (
        "Infarto agudo do miocárdio com supradesnivelamento do segmento ST (IAM-CSST): "
        "dor precordial em aperto irradiada para membro superior esquerdo e mandíbula, "
        "associada a sudorese fria. Conduta imediata: AAS 300 mg, heparina e reperfusão "
        "percutânea primária em < 90 min."
    ),
    (
        "Hipertensão arterial sistêmica (HAS) estágio 2: PA ≥ 160/100 mmHg em duas aferições. "
        "Tratamento farmacológico preferencial: inibidores da ECA (enalapril 5–40 mg/dia) "
        "combinados a diuréticos tiazídicos (hidroclorotiazida 12,5–25 mg/dia)."
    ),
    (
        "Insuficiência cardíaca com fração de ejeção reduzida (ICFEr): dispneia ao esforço, "
        "ortopneia, edema de membros inferiores e B3 à ausculta. "
        "Pilares do tratamento: IECA, betabloqueador e antagonista da aldosterona."
    ),
    (
        "Flutter atrial: ritmo regular com frequência atrial de 250–350 bpm e ventricular de "
        "150 bpm (bloqueio 2:1 típico). ECG: ondas F em 'dentes de serra' em D2, D3, aVF. "
        "Cardioversão elétrica sincronizada é indicada em instabilidade hemodinâmica."
    ),
    (
        "Dissecção aórtica tipo A (Stanford): dor torácica de início abrupto, 'em rasgo', "
        "irradiada para dorso. Complicações incluem tamponamento cardíaco e isquemia miocárdica. "
        "Cirurgia de emergência é mandatória."
    ),

    # Pneumologia
    (
        "Asma brônquica: broncoespasmo reversível com sibilos expiratórios, tosse seca e "
        "dispneia episódica. Crise grave: SpO₂ < 92 %, uso de musculatura acessória. "
        "Tratamento: broncodilatador inalatório (salbutamol) e corticosteroide sistêmico."
    ),
    (
        "Pneumonia adquirida na comunidade (PAC) por Streptococcus pneumoniae: febre, "
        "tosse produtiva com expectoração amarelada, dor pleurítica e consolidação lobar "
        "na radiografia de tórax. Amoxicilina-clavulanato é a opção preferencial em adultos."
    ),
    (
        "Tromboembolismo pulmonar (TEP): dispneia súbita, taquicardia sinusal, hipoxemia e "
        "dor pleurítica. Exame padrão-ouro: angiotomografia pulmonar. "
        "Anticoagulação com heparina de baixo peso molecular deve ser iniciada imediatamente."
    ),
    (
        "Doença Pulmonar Obstrutiva Crônica (DPOC) exacerbada: piora da dispneia basal, "
        "aumento da expectoração e alteração da cor do escarro. Tratamento da exacerbação: "
        "broncodilatador, corticosteroide sistêmico e antibioticoterapia (azitromicina)."
    ),

    # Gastroenterologia
    (
        "Doença do refluxo gastroesofágico (DRGE): pirose e regurgitação ácida, piora em "
        "decúbito dorsal e após refeições. O inibidor de bomba de prótons (omeprazol 20 mg/dia "
        "em jejum) é o tratamento de escolha."
    ),
    (
        "Hemorragia digestiva alta (HDA) varicosa: hematêmese em paciente com hepatopatia "
        "crônica, esplenomegalia e circulação colateral abdominal. Octreotida IV e endoscopia "
        "terapêutica emergencial com ligadura elástica são as condutas prioritárias."
    ),
    (
        "Apendicite aguda: dor migratória para fossa ilíaca direita, febre baixa, náusea e "
        "sinal de Blumberg positivo. Leucocitose com desvio à esquerda. "
        "Tratamento cirúrgico (apendicectomia laparoscópica) é indicado após diagnóstico."
    ),
    (
        "Pancreatite aguda: dor abdominal em barra irradiada para o dorso, náusea e vômitos, "
        "amilase e lipase sérica > 3× o limite superior da normalidade. "
        "Conduta: jejum, hidratação vigorosa IV e analgesia."
    ),

    # Endocrinologia
    (
        "Cetoacidose diabética (CAD): hiperglicemia > 250 mg/dL, pH < 7,30, bicarbonato "
        "< 18 mEq/L e cetonemia positiva. Tratamento: insulina regular IV em bomba, "
        "hidratação com SF 0,9 % e reposição de potássio."
    ),
    (
        "Hipotireoidismo primário: fadiga, ganho de peso, constipação, intolerância ao frio, "
        "bradicardia e mixedema. TSH elevado com T4 livre baixo. "
        "Reposição com levotiroxina sódica, ajustada a cada 6–8 semanas."
    ),
    (
        "Síndrome de Cushing: ganho de peso centrípeto, estrias violáceas largas, hipertensão, "
        "hiperglicemia e osteoporose. Cortisol livre urinário de 24 h e teste de supressão com "
        "dexametasona 1 mg confirmam o diagnóstico."
    ),

    # Infectologia
    (
        "Meningite bacteriana: cefaleia intensa, rigidez de nuca, febre alta e fotofobia "
        "(tríade clássica). Punção lombar: pleocitose neutrofílica, glicose baixa, proteína "
        "elevada. Ceftriaxona 4 g/dia IV + dexametasona 0,15 mg/kg devem ser iniciados "
        "imediatamente."
    ),
    (
        "Dengue com sinais de alarme: dor abdominal intensa, vômitos persistentes, "
        "acumulação de líquidos, sangramento de mucosas, letargia e aumento progressivo do "
        "hematócrito. Hospitalização e hidratação IV são obrigatórias."
    ),
    (
        "Sepse (critérios Sepsis-3): suspeita de infecção + disfunção orgânica aguda "
        "(SOFA ≥ 2 pontos). Bundle da primeira hora: culturas, antibiótico amplo espectro, "
        "30 mL/kg de cristaloide e dosagem de lactato."
    ),
    (
        "Pneumocistose (PCP) em paciente imunocomprometido: dispneia progressiva, tosse "
        "seca, febre e hipoxemia desproporcional ao exame físico. "
        "Tratamento: sulfametoxazol-trimetoprima (SMX-TMP) IV + corticosteroide se PaO₂ < 70 mmHg."
    ),
]


def criar_indice_hnsw(documentos, m=16, ef_construction=200, ef_search=50):
    print("\nPASSO 1 - Construção do índice HNSW")

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = modelo.encode(
        documentos,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dimensao = embeddings.shape[1]
    indice = faiss.IndexHNSWFlat(dimensao, m)
    indice.hnsw.efConstruction = ef_construction
    indice.hnsw.efSearch = ef_search
    indice.add(embeddings)

    print(f"Índice criado com {indice.ntotal} vetores de dimensão {dimensao}.")
    print(f"Parâmetros: M={m}, ef_construction={ef_construction}, ef_search={ef_search}\n")

    return indice, modelo


def gerar_hyde_local(pergunta):
    print("PASSO 2 - Transformação da query com HyDE local")
    print(f"Pergunta original: {pergunta}\n")

    texto = pergunta.lower()

    exemplos = [
        (
            ["dor de cabeça", "latejante", "luz", "enjoo", "náusea"],
            "Paciente apresenta cefaleia pulsátil unilateral associada a fotofobia, "
            "fonofobia e náusea, quadro compatível com migrânea. O tratamento agudo "
            "pode incluir AINEs e triptanos, enquanto a profilaxia é considerada em "
            "casos recorrentes ou incapacitantes.",
        ),
        (
            ["coração", "batendo rápido", "irregular", "sem ar", "falta de ar"],
            "Paciente apresenta palpitações, taquicardia e dispneia, podendo sugerir "
            "arritmia supraventricular, como flutter atrial. A avaliação clínica deve "
            "considerar estabilidade hemodinâmica, eletrocardiograma e necessidade de "
            "cardioversão em casos graves.",
        ),
        (
            ["febre", "tosse", "catarro", "peito", "pneumonia"],
            "Paciente apresenta febre, tosse produtiva, dor pleurítica e sinais de "
            "consolidação pulmonar, quadro compatível com pneumonia adquirida na "
            "comunidade. A abordagem inclui avaliação radiológica e antibioticoterapia.",
        ),
        (
            ["barriga", "dor abdominal", "vômito", "náusea"],
            "Paciente apresenta dor abdominal associada a náuseas e vômitos, exigindo "
            "avaliação de hipóteses como apendicite aguda, pancreatite ou outras causas "
            "gastrointestinais.",
        ),
    ]

    melhor_texto = None
    maior_pontuacao = 0

    for palavras_chave, documento in exemplos:
        pontuacao = sum(palavra in texto for palavra in palavras_chave)
        if pontuacao > maior_pontuacao:
            maior_pontuacao = pontuacao
            melhor_texto = documento

    if melhor_texto is None:
        melhor_texto = (
            "Paciente apresenta sintomas inespecíficos que devem ser interpretados "
            "em linguagem clínica, relacionando sinais, hipóteses diagnósticas e "
            "condutas descritas em manuais médicos."
        )

    print("Documento hipotético:")
    print(melhor_texto + "\n")

    return melhor_texto


def buscar_documentos(documento_hyde, indice, documentos, modelo_embedding, k=10):
    print(f"PASSO 3 - Recuperação dos Top-{k} documentos")

    vetor_query = modelo_embedding.encode(
        [documento_hyde],
        normalize_embeddings=True,
    )
    vetor_query = np.asarray(vetor_query, dtype="float32")

    distancias, ids = indice.search(vetor_query, k)

    resultados = []
    for posicao, (idx, distancia) in enumerate(zip(ids[0], distancias[0]), start=1):
        documento = documentos[idx]
        similaridade = 1.0 - distancia / 2.0
        resultados.append((documento, float(similaridade)))

        trecho = documento[:100].replace("\n", " ")
        print(f"{posicao:02d}. similaridade={similaridade:.4f} | {trecho}...")

    print()
    return resultados


def reranquear_documentos(pergunta, candidatos, top_n=3):
    print(f"PASSO 4 - Re-ranking com CrossEncoder para Top-{top_n}")

    modelo = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pares = [(pergunta, documento) for documento, _ in candidatos]
    scores = modelo.predict(pares)

    reranqueados = sorted(
        zip([documento for documento, _ in candidatos], scores),
        key=lambda item: item[1],
        reverse=True,
    )

    melhores = reranqueados[:top_n]

    for posicao, (documento, score) in enumerate(melhores, start=1):
        print(f"\nDocumento {posicao} | score={score:.4f}")
        print(documento)

    print()
    return melhores


def main():
    pergunta = "dor de cabeça latejante, luz incomodando bastante e enjoo"

    indice, modelo_embedding = criar_indice_hnsw(
        DOCUMENTOS_MEDICOS,
        m=16,
        ef_construction=200,
        ef_search=50,
    )

    documento_hyde = gerar_hyde_local(pergunta)
    candidatos = buscar_documentos(documento_hyde, indice, DOCUMENTOS_MEDICOS, modelo_embedding, k=10)
    top3 = reranquear_documentos(pergunta, candidatos, top_n=3)

    print("CONTEXTO FINAL")
    print("Estes documentos seriam enviados para o LLM gerador:\n")

    for i, (documento, _) in enumerate(top3, start=1):
        print(f"[Contexto {i}]")
        print(documento)
        print()


if __name__ == "__main__":
    main()
