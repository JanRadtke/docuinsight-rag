"""
Generates 10 fictional healthcare documents as PDFs for DocuInsight demo.
All data is entirely fictional — no real patients, no medical advice.

Run: python scripts/generate_healthcare_pdfs.py
"""

import json
import os
from fpdf import FPDF

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def sanitize(text: str) -> str:
    """Replace characters outside latin-1 range with ASCII equivalents."""
    return (text
            .replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2026", "...") # ellipsis
            )


def make_pdf(filename: str, title: str, sections: list[tuple[str, str]]):
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, sanitize(heading), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize(body))
        pdf.ln(2)

    path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(path)
    print(f"  Created: {filename}")


DOCS = [
    (
        "patient_therapy_depression_cbt.pdf",
        "Therapy Progress Report — Cognitive Behavioural Therapy (CBT)",
        [
            ("Patient & Setting",
             "Fictional patient: Alex M., 34 years, outpatient setting. "
             "Diagnosis: Major Depressive Disorder (ICD-10: F32.1). "
             "Treating therapist: Dr. S. Hoffmann (fictional). Session count: 12 of 20 planned."),
            ("Treatment Approach",
             "Cognitive Behavioural Therapy (CBT) following the protocol by Beck (1979). "
             "Key techniques applied: behavioural activation, cognitive restructuring, "
             "thought records, and problem-solving training. "
             "Sessions are held weekly, 50 minutes each."),
            ("Progress after 12 Sessions",
             "PHQ-9 score at intake: 18 (moderately severe). Current PHQ-9: 9 (mild). "
             "Patient reports improved sleep quality, reduced anhedonia, and increased "
             "social engagement. Automatic negative thoughts still present but patient "
             "demonstrates growing ability to challenge them using thought records."),
            ("Obstacles & Adjustments",
             "Session 7: Patient reported work-related stress spike leading to temporary "
             "score increase. Adapted plan to include stress-inoculation techniques. "
             "Homework compliance: approximately 70%, satisfactory for this phase."),
            ("Next Steps",
             "Sessions 13–20 will focus on relapse prevention, identifying early warning "
             "signs, and building a personal coping plan. Consideration of step-down to "
             "bi-weekly sessions after session 16 pending continued improvement."),
        ]
    ),
    (
        "patient_sleep_disorder_protocol.pdf",
        "Sleep Disorder Treatment Protocol — CBT-I",
        [
            ("Clinical Background",
             "Fictional patient: Maria K., 29 years. Diagnosis: Chronic Insomnia Disorder "
             "(ICD-10: F51.0) for 18 months. Referral from GP after failed pharmacological "
             "trial (low-dose sedative antihistamine, discontinued after 4 weeks)."),
            ("CBT-I Components",
             "Cognitive Behavioural Therapy for Insomnia (CBT-I) is the first-line treatment "
             "recommended by the AWMF and NICE guidelines. Components in this protocol: "
             "1) Sleep restriction therapy — initial sleep window 5.5 h, expanded weekly. "
             "2) Stimulus control — bed only for sleep and intimacy, no screens. "
             "3) Sleep hygiene education — caffeine cutoff, consistent wake time. "
             "4) Cognitive restructuring of dysfunctional beliefs about sleep."),
            ("Sleep Diary Data (Week 1–4)",
             "Week 1: SOL avg 68 min, WASO avg 42 min, SE 61%. "
             "Week 2: SOL avg 51 min, WASO avg 30 min, SE 70%. "
             "Week 3: SOL avg 34 min, WASO avg 18 min, SE 81%. "
             "Week 4: SOL avg 22 min, WASO avg 12 min, SE 88%. "
             "Target: SE >= 85%. Target reached in week 4."),
            ("Outcome",
             "ISI score reduced from 21 (clinical insomnia, severe) to 8 (subthreshold). "
             "Patient discharged from active treatment. Monthly follow-up for 6 months."),
        ]
    ),
    (
        "diga_mental_health_app_evaluation.pdf",
        "DiGA Evaluation Report — Mindful Companion App (Fictional)",
        [
            ("Overview",
             "This document presents a fictional evaluation of 'Mindful Companion', "
             "a Digital Health Application (DiGA) for mild-to-moderate anxiety and "
             "depression, evaluated for inclusion in the BfArM DiGA directory. "
             "All data is entirely fictional for demonstration purposes."),
            ("Study Design",
             "Randomised controlled trial, n=240 participants (120 intervention, 120 control). "
             "Intervention: App usage for 12 weeks. Control: waitlist. "
             "Primary endpoint: GAD-7 score reduction at week 12. "
             "Secondary endpoints: PHQ-9, WHO-5 wellbeing, app engagement (DAU/MAU)."),
            ("Efficacy Results",
             "GAD-7 mean reduction: intervention -5.2 points, control -1.1 points (p<0.001). "
             "PHQ-9 mean reduction: intervention -4.8 points, control -0.9 points (p<0.001). "
             "Responder rate (>=50% symptom reduction): 61% intervention vs 18% control. "
             "NNT: 2.3. Effect size (Cohen's d): 0.74 (large)."),
            ("Safety & Data Protection",
             "No serious adverse events related to the app. DSGVO-compliant data processing. "
             "Data stored on German servers. Pseudonymisation applied. "
             "No sale of data to third parties. Annual penetration test conducted."),
            ("Conclusion",
             "Mindful Companion demonstrates clinically meaningful and statistically significant "
             "improvements in anxiety and depression symptoms. Recommended for permanent "
             "inclusion in the DiGA directory (Evidenzstufe 1b)."),
        ]
    ),
    (
        "clinical_guideline_anxiety_treatment.pdf",
        "Clinical Guideline — Treatment of Generalised Anxiety Disorder",
        [
            ("Introduction",
             "Generalised Anxiety Disorder (GAD, ICD-10: F41.1) is characterised by excessive, "
             "uncontrollable worry about multiple life domains, accompanied by physical symptoms "
             "such as muscle tension, fatigue, and sleep disturbance. Lifetime prevalence: ~5%. "
             "This fictional guideline summarises recommended treatment steps."),
            ("Step 1: Psychoeducation & Active Monitoring",
             "First contact: provide information about GAD, normalise symptoms, rule out "
             "medical causes (thyroid, cardiac). Active monitoring for 2–4 weeks. "
             "Self-help resources: bibliotherapy, validated apps (e.g. DiGA-listed)."),
            ("Step 2: Psychological Treatment",
             "CBT is the first-line psychological treatment (NICE Grade A evidence). "
             "Key components: worry time, cognitive restructuring of overestimation of threat "
             "and intolerance of uncertainty, relaxation training (PMR), graded exposure. "
             "Minimum 12 sessions recommended for full effect."),
            ("Step 3: Pharmacological Treatment",
             "If psychological treatment insufficient or unavailable: "
             "First-line: SSRIs (sertraline, escitalopram) or SNRIs (venlafaxine, duloxetine). "
             "Second-line: pregabalin (note dependence potential), buspirone. "
             "Avoid long-term benzodiazepine use. Minimum treatment duration 6–12 months."),
            ("Comorbidities",
             "GAD frequently co-occurs with depression (50–60%), other anxiety disorders, "
             "and substance use. Treat predominant condition first. Integrated treatment "
             "preferable when resources allow."),
        ]
    ),
    (
        "patient_onboarding_mental_health_platform.pdf",
        "Patient Onboarding Document — Digital Mental Health Platform",
        [
            ("Welcome",
             "Welcome to CareConnect (fictional). This document explains how our platform "
             "works, what data we collect, and your rights as a user. "
             "CareConnect connects patients with licensed therapists for video, chat, "
             "and asynchronous messaging-based therapy."),
            ("How It Works",
             "1. Complete the intake questionnaire (PHQ-9, GAD-7, demographic information). "
             "2. Our matching algorithm suggests 3 therapists based on your needs and "
             "   availability. You choose. "
             "3. First session is a 20-minute introduction call at no cost. "
             "4. Ongoing therapy: 50-minute sessions, weekly or bi-weekly."),
            ("Data Protection (DSGVO)",
             "We process your health data as a data processor under Art. 28 DSGVO. "
             "Your therapist is the data controller for clinical records. "
             "Data is stored encrypted on EU servers. You may request deletion at any time. "
             "We never sell your data or use it for advertising. "
             "Data retention: clinical notes 10 years per legal requirement (§630f BGB)."),
            ("Your Rights",
             "Right to access (Art. 15 DSGVO), rectification (Art. 16), erasure (Art. 17), "
             "portability (Art. 20). Contact our Data Protection Officer: dpo@careconnect.example"),
            ("Crisis Support",
             "CareConnect is not a crisis service. In an emergency call 112 (EU) or "
             "the Telefonseelsorge: 0800 111 0 111 (free, 24/7). "
             "Our therapists cannot respond outside scheduled sessions."),
        ]
    ),
    (
        "therapy_relapse_prevention_plan.pdf",
        "Relapse Prevention Plan — Post-Treatment Documentation",
        [
            ("Purpose",
             "This fictional relapse prevention plan is developed collaboratively between "
             "therapist and patient at the end of treatment. It serves as a personal "
             "reference to maintain gains and respond early to warning signs."),
            ("Personal Warning Signs",
             "Early signs identified by patient: increased irritability, social withdrawal, "
             "difficulty concentrating at work, skipping morning exercise, "
             "returning to scrolling social media for >2 hours/day."),
            ("Coping Strategies That Worked",
             "1. Behavioural activation: 30-min walk regardless of motivation. "
             "2. Thought record when 'everything is hopeless' thoughts appear. "
             "3. Calling a friend or family member rather than isolating. "
             "4. 5-minute breathing exercise (4-7-8 pattern) for acute anxiety."),
            ("Action Plan",
             "Green (doing well): maintain exercise, social contact, sleep routine. "
             "Amber (1–2 warning signs): reinstate daily thought records, contact GP. "
             "Red (3+ signs or PHQ-9 > 15): contact therapist within 48h for booster session, "
             "consider GP referral for medication review."),
            ("Booster Sessions",
             "One booster session scheduled at 3 months post-discharge. "
             "Further sessions available on self-referral within 12 months."),
        ]
    ),
    (
        "depression_psychoeducation_handout.pdf",
        "Psychoeducation Handout — Understanding Depression",
        [
            ("What is Depression?",
             "Depression (Major Depressive Disorder) is a common mental health condition "
             "characterised by persistent low mood, loss of interest or pleasure (anhedonia), "
             "fatigue, changes in sleep and appetite, difficulties concentrating, and in "
             "severe cases, thoughts of self-harm. It is not a personal weakness or "
             "character flaw — it has biological, psychological, and social causes."),
            ("How Common is it?",
             "Lifetime prevalence: ~16% globally (WHO, 2023 estimate). "
             "Leading cause of disability worldwide. "
             "Women are diagnosed approximately twice as often as men, though men may be "
             "less likely to seek help. First episode most common in young adulthood."),
            ("The Biopsychosocial Model",
             "Biological: genetic vulnerability, neurotransmitter dysregulation (serotonin, "
             "dopamine, noradrenaline), HPA axis dysregulation (cortisol). "
             "Psychological: negative cognitive triad (self, world, future), "
             "rumination, low self-efficacy. "
             "Social: life events, social isolation, socioeconomic stress."),
            ("Effective Treatments",
             "Mild depression: structured self-help, physical exercise (strong evidence), "
             "DiGA, brief CBT. "
             "Moderate-severe: CBT or other evidence-based therapies + antidepressants. "
             "Recovery takes time — most people see improvement within 4–8 weeks of "
             "starting treatment. Completing the full course is important."),
            ("Myths vs Facts",
             "Myth: Antidepressants are addictive. Fact: They are not addictive, though "
             "dose reduction should be gradual. "
             "Myth: You just need to think positive. Fact: Depression changes brain function; "
             "willpower alone is insufficient. "
             "Myth: Therapy is only for severe cases. Fact: Early intervention improves outcomes."),
        ]
    ),
    (
        "compliance_healthcare_data_processing.pdf",
        "Compliance Brief — Healthcare Data Processing under DSGVO & KHZG",
        [
            ("Scope",
             "This fictional compliance brief covers obligations for healthcare providers "
             "processing patient data in Germany under: DSGVO (EU 2016/679), "
             "Bundesdatenschutzgesetz (BDSG), Patientendatenschutzgesetz (PDSG), "
             "and Krankenhauszukunftsgesetz (KHZG) for digitisation investments."),
            ("Legal Basis for Processing",
             "Health data is special category data (Art. 9 DSGVO). "
             "Legal bases: Art. 9(2)(h) — treatment purposes; Art. 9(2)(i) — public interest "
             "in public health; §22(1)(1)(b) BDSG — preventive medicine. "
             "Explicit consent (Art. 6(1)(a), Art. 9(2)(a)) required for research and "
             "secondary purposes."),
            ("Technical & Organisational Measures",
             "Required under Art. 32 DSGVO: encryption at rest (AES-256) and in transit (TLS 1.3), "
             "role-based access control, audit logging, pseudonymisation where possible, "
             "regular penetration testing, data minimisation by design (Art. 25 DSGVO). "
             "KHZG Fördertatbestand 2 requires IT security measures for funded investments."),
            ("Data Subject Rights in Healthcare",
             "Right of access (Art. 15): patients may request their complete health record. "
             "Right to erasure (Art. 17): limited in healthcare — legal retention periods "
             "override erasure requests (§630f BGB: 10 years; §28 RöV: 10 years X-ray). "
             "Right to portability (Art. 20): applies to digitally processed data."),
            ("Breach Notification",
             "Data breaches must be reported to supervisory authority within 72 hours (Art. 33). "
             "Affected patients must be notified without undue delay if high risk (Art. 34). "
             "Healthcare breaches frequently reported to BfDI and state DPAs."),
        ]
    ),
    (
        "mental_health_assessment_phq9_gad7.pdf",
        "Standardised Assessment Report — PHQ-9 and GAD-7 Screening",
        [
            ("Introduction to Screening Tools",
             "The Patient Health Questionnaire-9 (PHQ-9) and Generalised Anxiety Disorder-7 "
             "(GAD-7) are validated, widely used screening instruments in primary and "
             "secondary care. Both are in the public domain and freely available. "
             "They are self-report measures completed by the patient."),
            ("PHQ-9 Scoring Guide",
             "9 items rated 0–3 over the past 2 weeks. Total range: 0–27. "
             "0–4: minimal depression. 5–9: mild. 10–14: moderate. "
             "15–19: moderately severe. 20–27: severe. "
             "Clinical action point: score >= 10 warrants clinical assessment. "
             "Item 9 (self-harm) requires immediate assessment regardless of total score."),
            ("GAD-7 Scoring Guide",
             "7 items rated 0–3. Total range: 0–21. "
             "0–4: minimal anxiety. 5–9: mild. 10–14: moderate. 15–21: severe. "
             "Clinical action point: score >= 10 warrants clinical assessment. "
             "Sensitivity 89%, specificity 82% for GAD at cut-off 10."),
            ("Use in Digital Health Applications",
             "PHQ-9 and GAD-7 are commonly used in DiGA as primary endpoints in clinical trials "
             "and as ongoing monitoring tools within apps. "
             "Change scores (pre–post) are recommended over single timepoint scores. "
             "Minimal Clinically Important Difference (MCID): PHQ-9: 5 points; GAD-7: 4 points."),
            ("Limitations",
             "Screening tools do not replace clinical diagnosis. "
             "PHQ-9 has higher sensitivity than specificity — false positives occur. "
             "Cultural and language adaptations should be used for non-German-speaking patients. "
             "Repeated administration may lead to practice effects; interpret cautiously."),
        ]
    ),
    (
        "cbt_worksheet_thought_record.pdf",
        "CBT Worksheet — Thought Record (Cognitive Restructuring)",
        [
            ("What is a Thought Record?",
             "A thought record (also: 'dysfunctional thought record') is a core CBT tool "
             "used to identify and challenge automatic negative thoughts. "
             "It helps patients become aware of the link between situations, thoughts, "
             "emotions, and behaviours, and to develop more balanced perspectives."),
            ("The 7-Column Format",
             "Column 1 — Situation: What happened? Where, when, with whom? "
             "Column 2 — Automatic Thought: What went through your mind? Rate belief 0–100%. "
             "Column 3 — Emotion: What did you feel? Rate intensity 0–100%. "
             "Column 4 — Evidence For: Facts that support the automatic thought. "
             "Column 5 — Evidence Against: Facts that contradict it. "
             "Column 6 — Balanced Thought: A more balanced perspective (rate belief 0–100%). "
             "Column 7 — Outcome: Re-rate emotion intensity."),
            ("Example (Fictional)",
             "Situation: Boss did not say hello in the corridor. "
             "Automatic thought: 'She is angry with me, I am going to be fired.' (belief: 85%). "
             "Emotion: Anxiety 80%, Shame 60%. "
             "Evidence for: She seemed distracted, I made an error last week. "
             "Evidence against: She is usually busy in the morning, she praised my work yesterday, "
             "I have had no formal warning. "
             "Balanced thought: 'She was probably busy. If there was an issue she would tell me.' (belief: 70%). "
             "Outcome: Anxiety 35%, Shame 20%."),
            ("When to Use",
             "Use thought records when you notice a sudden shift in mood, "
             "when you catch yourself catastrophising or mind-reading, "
             "or when preparing for a difficult situation in advance (anticipatory anxiety). "
             "Daily practice for the first 4 weeks of CBT is recommended for best results."),
            ("Tips for Therapists",
             "Introduce thought records in session first — do not assign as homework without "
             "in-session practice. Review completed records collaboratively. "
             "Common pitfalls: patients writing opinions as evidence, skipping the balanced thought. "
             "Socratic questioning is more effective than direct correction."),
        ]
    ),
]


TESTSET = [
    {
        "question": "What are the components of CBT-I for treating insomnia?",
        "reference_truth": (
            "CBT-I includes sleep restriction therapy (initial sleep window 5.5 h, expanded weekly), "
            "stimulus control (bed only for sleep and intimacy, no screens), sleep hygiene education "
            "(caffeine cutoff, consistent wake time), and cognitive restructuring of dysfunctional "
            "beliefs about sleep."
        ),
        "must_contain": ["sleep restriction", "stimulus control", "sleep hygiene"],
        "intent": "SEARCH"
    },
    {
        "question": "What does a PHQ-9 score of 15 indicate and what action should be taken?",
        "reference_truth": (
            "A PHQ-9 score of 15 falls in the 'moderately severe depression' range (15-19). "
            "It warrants clinical assessment. Item 9 (self-harm) requires immediate assessment "
            "regardless of total score."
        ),
        "must_contain": ["moderately severe", "clinical assessment"],
        "intent": "SEARCH"
    },
    {
        "question": "What DSGVO legal bases apply to processing patient health data in Germany?",
        "reference_truth": (
            "Health data is special category data under Art. 9 DSGVO. Legal bases include: "
            "Art. 9(2)(h) for treatment purposes, Art. 9(2)(i) for public interest in public health, "
            "and S22(1)(1)(b) BDSG for preventive medicine. Explicit consent under Art. 6(1)(a) and "
            "Art. 9(2)(a) is required for research and secondary purposes."
        ),
        "must_contain": ["Art. 9", "DSGVO", "special category"],
        "intent": "SEARCH"
    },
    {
        "question": "How does a DiGA demonstrate clinical efficacy for BfArM approval?",
        "reference_truth": (
            "A DiGA must demonstrate clinical efficacy through a randomised controlled trial. "
            "Primary endpoints are validated scales such as GAD-7 or PHQ-9. Results must show "
            "statistically significant improvement versus control. The fictional Mindful Companion "
            "achieved GAD-7 reduction of -5.2 vs -1.1 (p<0.001), Cohen's d 0.74, NNT 2.3."
        ),
        "must_contain": ["GAD-7", "randomised", "BfArM"],
        "intent": "SEARCH"
    },
    {
        "question": "What are the early warning signs and action steps in a relapse prevention plan for depression?",
        "reference_truth": (
            "Early warning signs include increased irritability, social withdrawal, difficulty "
            "concentrating, skipping exercise, and excessive social media use. The action plan uses "
            "a traffic light system: Green - maintain routines; Amber (1-2 signs) - reinstate thought "
            "records and contact GP; Red (3+ signs or PHQ-9 > 15) - contact therapist within 48h."
        ),
        "must_contain": ["warning signs", "thought record", "PHQ-9"],
        "intent": "SEARCH"
    },
    {
        "question": "How does a CBT thought record work and what are its 7 columns?",
        "reference_truth": (
            "A thought record identifies and challenges automatic negative thoughts. The 7 columns are: "
            "1) Situation, 2) Automatic Thought (belief 0-100%), 3) Emotion (intensity 0-100%), "
            "4) Evidence For, 5) Evidence Against, 6) Balanced Thought, 7) Outcome (re-rate emotion). "
            "Daily practice for the first 4 weeks of CBT is recommended."
        ),
        "must_contain": ["automatic thought", "evidence", "balanced thought"],
        "intent": "SEARCH"
    },
    {
        "question": "Compare the first-line treatments for GAD: psychological vs pharmacological",
        "reference_truth": (
            "Psychological first-line: CBT (NICE Grade A evidence) with worry time, cognitive "
            "restructuring, relaxation training, and graded exposure, minimum 12 sessions. "
            "Pharmacological first-line: SSRIs (sertraline, escitalopram) or SNRIs (venlafaxine, "
            "duloxetine), minimum 6-12 months. Benzodiazepines should be avoided long-term."
        ),
        "must_contain": ["CBT", "SSRI", "benzodiazepine"],
        "intent": "COMPARE"
    },
    {
        "question": "Summarise the key data protection rights patients have under DSGVO",
        "reference_truth": (
            "Patients have: right of access (Art. 15) to their complete health record, right to "
            "rectification (Art. 16), right to erasure (Art. 17) - though limited in healthcare by "
            "legal retention periods (e.g. 10 years under S630f BGB), right to data portability "
            "(Art. 20) for digitally processed data."
        ),
        "must_contain": ["Art. 15", "erasure", "portability"],
        "intent": "SUMMARIZE"
    },
]


def generate_testset():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "testset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(TESTSET, f, indent=2, ensure_ascii=False)
    print(f"  Created: data/testset.json ({len(TESTSET)} test cases)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {len(DOCS)} healthcare PDFs in {OUTPUT_DIR}/")
    for filename, title, sections in DOCS:
        make_pdf(filename, title, sections)
    print()
    print("Generating data/testset.json ...")
    generate_testset()
    print("Done.")


if __name__ == "__main__":
    main()
