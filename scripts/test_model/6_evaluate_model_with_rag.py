    gpt4_evaluation: EvaluatorResponse
    gemini_evaluation: Optional[EvaluatorResponse]
    consensus_scores: EvaluationScores
    agreement_rate: float
    discrepancies: List[Dict]

# ============================================================================
# CARGA DE DATOS DAIC-WOZ
# ============================================================================

def load_daic_woz_data() -> Tuple[List[TestCase], List[TestCase]]:
    """Cargar datos de DAIC-WOZ (test + dev splits)"""
    
    print("📂 Cargando datos de DAIC-WOZ...")
    
    # Cargar PHQ-8 scores del dev split (tiene los scores)
    dev_df = pd.read_csv(DATA_DIR / "dev_split_Depression_AVEC2017.csv")
    
    # Cargar participant IDs de test y dev
    test_ids_df = pd.read_csv(DATA_DIR / "test_split_Depression_AVEC2017.csv")
    dev_ids = dev_df['Participant_ID'].tolist()
    test_ids = test_ids_df['participant_ID'].tolist()
    
    # Crear diccionario de PHQ-8 scores
    phq8_scores = {}
    for _, row in dev_df.iterrows():
        pid = row['Participant_ID']
        phq8_scores[pid] = row['PHQ8_Score']
    
    # Para test split, necesitamos inferir los scores del full_test_split.csv
    try:
        full_test = pd.read_csv(DATA_DIR / "full_test_split.csv")
        for _, row in full_test.iterrows():
            if 'PHQ8_Score' in row:
                phq8_scores[row['Participant_ID']] = row['PHQ8_Score']
    except:
        print("⚠️  No se encontró full_test_split.csv, usando solo dev scores")
    
    test_cases = []
    dev_cases = []
    
    # Procesar casos de test
    for pid in test_ids:
        case = load_participant_case(pid, phq8_scores.get(pid, 0), 
                                     test_ids_df[test_ids_df['participant_ID']==pid]['Gender'].values[0])
        if case:
            test_cases.append(case)
    
    # Procesar casos de dev
    for pid in dev_ids:
        case = load_participant_case(pid, phq8_scores.get(pid, 0),
                                     dev_df[dev_df['Participant_ID']==pid]['Gender'].values[0])
        if case:
            dev_cases.append(case)
    
    print(f"✅ Cargados {len(test_cases)} casos de test y {len(dev_cases)} casos de dev")
    print(f"📊 Total: {len(test_cases) + len(dev_cases)} casos")
    
    return test_cases, dev_cases

def load_participant_case(participant_id: int, phq8_score: int, gender: int) -> Optional[TestCase]:
    """Cargar caso individual de un participante"""
    
    transcript_file = TRANSCRIPTS_DIR / f"{participant_id}_TRANSCRIPT.csv"
    
    if not transcript_file.exists():
        print(f"⚠️  Transcript no encontrado: {participant_id}")
        return None
    
    # Leer transcript
    with open(transcript_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        lines = list(reader)
    
    # Extraer respuestas del participante
    participant_responses = [
        line['value'] for line in lines 
        if line['speaker'] == 'Participant' and len(line['value'].strip()) > 10
    ]
    
    if not participant_responses:
        return None
    
    # Tomar una respuesta representativa (la más larga del primer tercio)
    first_third = participant_responses[:len(participant_responses)//3 + 1]
    user_input = max(first_third, key=len) if first_third else participant_responses[0]
    
    # Transcript completo (primeros 10 turnos)
    transcript = "\n".join([
        f"{line['speaker']}: {line['value']}" 
        for line in lines[:20]
    ])
    
    # Categorizar según PHQ-8
    category = categorize_by_phq8(phq8_score)
    
    return TestCase(
        participant_id=str(participant_id),
        phq8_score=phq8_score,
        gender=gender,
        transcript=transcript,
        user_input=user_input,
        category=category
    )

def categorize_by_phq8(score: int) -> str:
    """Categorizar caso según PHQ-8 score"""
    if score >= 20:
        return "crisis"
    elif score >= 15:
        return "severe_depression"
    elif score >= 10:
        return "moderate_depression"
    elif score >= 5:
        return "mild_depression"
    else:
        return "low_risk"

# ============================================================================
# BASE DE CONOCIMIENTO (RAG)
# ============================================================================

def create_knowledge_base():
    """Crear base de conocimiento con criterios psicológicos"""
    
    print("📚 Creando base de conocimiento...")
    
    # 1. Criterios de Empatía (Carkhuff ECRS)
    empathy_criteria = {
        "name": "Escala de Respuesta Empática de Carkhuff (ECRS)",
        "description": "Evalúa el nivel de empatía en respuestas terapéuticas",
        "levels": {
            "1": {
                "score": 1.0,
                "name": "Ignora o minimiza",
                "description": "No reconoce el estado emocional, respuesta genérica o irrelevante, tono frío o distante",
                "example": "Todos tenemos problemas. Deberías ser más positivo."
            },
            "2": {
                "score": 2.0,
                "name": "Respuesta superficial",
                "description": "Reconocimiento mínimo de la emoción, sin profundización ni validación, respuesta mecánica",
                "example": "Entiendo que estás triste. ¿Qué más puedo hacer por ti?"
            },
            "3": {
                "score": 3.0,
                "name": "Respuesta intercambiable (OBJETIVO MÍNIMO)",
                "description": "Reconoce la emoción expresada, refleja comprensión básica, validación emocional presente",
                "example": "Veo que te sientes muy abrumado por esta situación. Es completamente válido sentirse así."
            },