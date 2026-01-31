# minimal implementation of Barry Robson's split-complex probabilistic reasoning approach 
# for use in medical knowledge representation and inference (beyond limitations of Bayes Nets).

# in other words, it's a toy prototype of Barry Robson’s algebraic approach to 
# epistemic reasoning in medicine, trying to reduce
# complex math to understandable Python.

# This script creates a split-complex vector representation of several patient states, where:
#  Real part = confirmed/strong evidence (diagnosis, symptoms, labs), with weights
#  Dual part = uncertain or contradictory evidence (e.g. possible diagnosis)
#   => Epistemically transparent

# The script:
# - Aggregates semantic triples into a patient state vector
# - Defines a medical guideline vector
# - Computes agreement and contradiction between the two split-complex vectors
# - Updates the patient vector with new evidence

# This is the Dirac-Q-UEL idea that Robson promotes: use vector algebra over a logic 
# space where contradiction isn't thrown away, but encoded, preserved, and updated.

# Math foundation for such vector algebra: it follows Barry Robson’s Q-UEL/Dirac logic, described here: 
# Steven Deckelman & Barry Robson (2014) Split-complex numbers and Dirac bra-kets.
# Communications in Information and Systems 14:135-59 

# Moritz 1x1 Notizen
# - wie definieren wir die Knowledge Base, zB rare disorders
#- wie funktioniert das Updating (etwas konkreter), a) knowledge base update, b) patient state / trajectory, c) how new data can decrease epistemic uncertainty
#- start with random decisions in simulations; when it can transition to the real world (key methods to compare with and how )
#- weighting, more serously
#- how split vectors are calculated


"""
The core algebra:
    - Uses split-complex numbers: z = a + b·h with h2 = +1
    - Inner product encodes both semantic similarity and epistemic tension:
        - Real part: vector agreement between confirmed facts
        - Dual part: contradiction / uncertainty interaction
Contradictions are preserved algebraically (not discarded), enabling Bayesian-like updates over time without loss of epistemic history.
"""

import numpy as np # for vectors
# note that Python's inherent representation of
# complex numbers is not for split-conplex algebra
# (see above 2014 paper)

class SplitComplexVector:
    def __init__(self, real, dual):
        self.real = np.array(real)
        self.dual = np.array(dual)

        # real: confirmed or strongly-supported facts (e.g., confirmed diagnosis)
        # dual: uncertain, conflicting, or potential information (e.g., possible diagnosis that is not confirmed)

    def inner_product(self, other):
        real_part = np.dot(self.real, other.real) + np.dot(self.dual, other.dual)
        dual_part = np.dot(self.real, other.dual) + np.dot(self.dual, other.real)
        return SplitComplexVector(real_part, dual_part)

        # Computes a Dirac-style inner product.
        # real_part: alignment of confirmed evidence (agreement of facts)
        # dual_part: epistemic tension (how much uncertain/contradictory evidence interacts 
        #  with confirmed evidence). Interaction between knowns and hypotheses

    def __repr__(self):
        return f"SplitComplexVector(real={self.real}, dual={self.dual})"

        # Pretty-print for debugging and inspection.


rdf_sample = """
<Patient123>  <hasDiagnosis>           <Hypertension>
<Patient123>  <hasSymptom>             <Fatigue>
<Patient123>  <hasLabResult>           <HbA1cHigh>
<Patient123>  <possibleDiagnosis>      <Diabetes>
<Patient123>  <conflictingDiagnosis>   <Anemia>

<Patient456>  <hasDiagnosis>           <Diabetes>
<Patient456>  <hasLabResult>           <HbA1cHigh>
<Patient456>  <hasSymptom>             <WeightLoss>
<Patient456>  <possibleDiagnosis>      <Hyperthyroidism>

<Patient789>  <hasSymptom>       <Tremor>
<Patient789>  <hasLabResult>     <NormalTSH>
<Patient789>  <possibleDiagnosis> <ParkinsonsDisease>
<Patient789>  <conflictingDiagnosis> <Hyperthyroidism>
"""

# each semantic triple becomes an epistemic quantum in this algebra.
# Represents clinical facts for one patient in RDF-triple style.


tokens = rdf_sample.replace(">", "> ").split()
# triples = list(zip(tokens[::3], tokens[1::3], tokens[2::3]))

from collections import defaultdict

# Group triples per patient (subject)
patient_triples = defaultdict(list)
for subj, pred, obj in zip(tokens[::3], tokens[1::3], tokens[2::3]):
    patient_id = subj.strip('<>')
    patient_triples[patient_id].append((pred, obj))


# Tokenizes and slices the RDF data into 3-tuples: subject–predicate–object. Per patient.


def get_weight(pred, obj):
    if pred == "hasDiagnosis":
        return 1.0
    elif pred == "hasLabResult":
        return 0.8
    elif pred == "hasSymptom":
        return 0.6
    elif pred == "possibleDiagnosis":
        return 0.4
    elif pred == "conflictingDiagnosis":
        return 0.3
    else:
        return 0.5

# Encodes medical evidence grading heuristics — a key idea from Robson’s PICO-based work.
#  Stronger evidence = heavier weight in the vector space.
# so, vectors are then evidence-aware (via weighting)

# mirror GRADE criteria in evidence-based medicine: strong vs weak recommendations, 
#  confirmed vs possible findings.

# fake vectors for testing (weighted by predicate)
concept_vectors = {
    'hasDiagnosis:Hypertension':       np.array([1, 0, 0, 0, 0, 0]),
    'hasSymptom:Fatigue':              np.array([0, 1, 0, 0, 0, 0]),
    'hasLabResult:HbA1cHigh':          np.array([0, 0, 1, 0, 0, 0]),
    'possibleDiagnosis:Diabetes':      np.array([0, 0, 0, 1, 0, 0]),
    'hasDiagnosis:Diabetes':           np.array([1, 0, 0, 1, 0, 0]),
}

concept_vectors.update({
    'conflictingDiagnosis:Anemia':     np.array([0, 0, 0, 0, 1, 0]),
    'hasSymptom:WeightLoss':           np.array([0, 0, 0, 0, 0, 1]),
    'possibleDiagnosis:Hyperthyroidism': np.array([0, 0, 0, 0.3, 0.1, 0.6]),
})



def encode_weighted_vector(pred, obj):
    weight = get_weight(pred.strip("<>"), obj.strip("<>"))
    key = f"{pred.strip('<>')}:{obj.strip('<>')}"
    base_vec = concept_vectors.get(key, np.zeros(6))
    return weight * base_vec

# Turns (predicate, object) into a weighted vector from the knowledge base.

# ✅ Combine split vectors
def combine_split_vectors(split_vectors):
    total_real = sum((v.real for v in split_vectors), start=np.zeros_like(split_vectors[0].real))
    total_dual = sum((v.dual for v in split_vectors), start=np.zeros_like(split_vectors[0].dual))
    return SplitComplexVector(total_real, total_dual)

# Adds all real parts and all dual parts across vectors.
#  Result: aggregate state of the patient.

def build_patient_state(triples):
    split_vectors = []
    for pred, obj in triples:
        vec = encode_weighted_vector(pred, obj)
        if pred.strip("<>") in ["possibleDiagnosis", "conflictingDiagnosis"]:
            scv = SplitComplexVector(real=np.zeros_like(vec), dual=vec)
        else:
            scv = SplitComplexVector(real=vec, dual=np.zeros_like(vec))
        split_vectors.append(scv)
    return combine_split_vectors(split_vectors)

patient_states = {}
for pid, triples in patient_triples.items():
    patient_states[pid] = build_patient_state(triples)

print("Data loaded, building patient states.")

# building a patient state for each patient


    
    

# You now create multiple epistemic states, which lets you:
#  Compare different patients to the same guideline
#  Evaluate disagreement between guideline updates
#  Cluster patients with similar real–dual ratios

# Very simple toy embedding: each concept maps to a 6D semantic vector.
# Later, used to generate real and dual parts of patient/guideline vectors.



# # ⬇️⬇️ Create list of SplitComplexVectors ⬇️⬇️
# split_vectors = []
# for subj, pred, obj in triples:
#     vec = encode_weighted_vector(pred, obj)
#     if pred.strip("<>") in ["possibleDiagnosis", "conflictingDiagnosis"]:
#         scv = SplitComplexVector(real=np.zeros_like(vec), dual=vec)
#     else:
#         scv = SplitComplexVector(real=vec, dual=np.zeros_like(vec))
#     split_vectors.append(scv)

# Depending on the predicate:
#  If confirmed (e.g. hasDiagnosis) → store in real.
#  If uncertain/conflicting (e.g. possibleDiagnosis) → store in dual.
# This implements the epistemic encoding core to Robson’s logic.



# patient_state = combine_split_vectors(split_vectors)
#print("\n✅ Combined patient state vector:")
#print(patient_state)

# ✅ Define a hypothetical guideline vector
guideline_real = concept_vectors['hasDiagnosis:Hypertension'] + concept_vectors['hasSymptom:Fatigue']
guideline_dual = np.zeros(6)
guideline_vector = SplitComplexVector(guideline_real, guideline_dual)
print("\n✅ Combined guideline vector:")
print(guideline_vector)

# A guideline encoding: “Patient should have hypertension + fatigue”
# No contradictions/uncertainty → dual = 0

# ✅ Do Dirac-style inner product (inference)
# score = patient_state.inner_product(guideline_vector)
# print("\n📊 Inference result:")
# print("Agreement (real):", score.real)
# print("Contradiction (dual):", score.dual)

# score = patient_state.inner_product(guideline_vector)
# print("\n📊 Patient vs Guideline Inference")
# print("✅ Agreement (real part):", score.real)
# print("⚠️  Epistemic contradiction (dual part):", score.dual)

for pid, state in patient_states.items():
    score = state.inner_product(guideline_vector)
    print(f"\n📊 Patient {pid} vs Guideline:")
    print("✅ Agreement (real):", score.real)
    print("⚠️  Epistemic contradiction (dual):", score.dual)
    match_quality = np.linalg.norm(score.real)
    contradiction_level = np.linalg.norm(score.dual)
    print(f"📈 Match Score: {match_quality:.2f} | ⚠️ Contradiction Score: {contradiction_level:.2f}")



# Compare Patient vs Guideline
# real part = how well the patient's confirmed state matches the guideline.
# dual part = interaction of confirmed vs. uncertain evidence.
#  High value = epistemic tension. Low value = clean match

# A strong real match with low dual score suggests:
#   - The patient fits the guideline closely.
#   - There is low epistemic tension.
#   => Good candidate for standard care.

# A weak real match with high dual score suggests:
#   - Unusual case with uncertain/conflicting information.
#   - May require personalized, out-of-guideline care.
#   => Good candidate for rare disease research, MDT review.

print("New data! Update knowledge about each patient's state, if they have an update.")

# ✅ Optional: Update with new knowledge

# Extend concept vectors with any new update concepts
concept_vectors.update({
    'hasDiagnosis:Diabetes':           np.array([1, 0, 0, 1, 0, 0]),
    'conflictingDiagnosis:Anemia':     np.array([0, 0, 0, 0, 1, 0]),
    'hasLabResult:ElevatedCRP':        np.array([0.2, 0.1, 0.4, 0, 0, 0.3]),
})

patient_updates = {
    'Patient123': ('hasDiagnosis', 'Diabetes'),
    'Patient456': ('conflictingDiagnosis', 'Anemia'),
    'Patient789': ('hasLabResult', 'ElevatedCRP'),
}

print("\n🔁 Updating patient states considering new knowledge:")
for pid, (pred, obj) in patient_updates.items():
    if pid in patient_states:
        vec = encode_weighted_vector(f"<{pred}>", f"<{obj}>")
        if pred in ["possibleDiagnosis", "conflictingDiagnosis"]:
            new_info_vec = SplitComplexVector(real=np.zeros_like(vec), dual=vec)
        else:
            new_info_vec = SplitComplexVector(real=vec, dual=np.zeros_like(vec))

        updated_state = patient_states[pid].inner_product(new_info_vec)
        print(f"\n🧬 Updated state for {pid}:")
        print("Updated Real:", updated_state.real)
        print("Updated Dual:", updated_state.dual)
    else:
        print(f"\n⚠️ No known state for patient {pid}. Skipping update.")



# new_info_vec = SplitComplexVector(real=concept_vectors['hasDiagnosis:Diabetes'], dual=np.zeros(6))
# updated_state = patient_state.inner_product(new_info_vec)
# print("\n🔁 Updated patient state after new knowledge:")
# print(updated_state)

# This demonstrates Bayesian-style updating via algebra.
# Can also allow future logic like contradiction resolution.

# Enables learning ecosystems: Contradictions aren’t bugs — they’re learning opportunities. 
# As new evidence arrives, the inner product updates the state algebraically (as shown in 
# the updated_state logic).



# Contradictions are not discarded but encoded in the dual part. Contradiction and uncertainty are algebraically visible and can be modulated and updated. In other words: This handles epistemic encoding: If the predicate implies uncertainty (e.g., “possibleDiagnosis”, “conflictingDiagnosis”), we put the vector in the dual component. Otherwise (e.g., confirmed diagnosis or lab results), we store it in the real component
    
# Then, each triple becomes a split-complex vector, e.g.:
# Real: [0.9, -1.2, 0.4, ...] + Dual: [0.0, 0.0, 0.0, ...] h
# Real: [0.0, 0.0, 0.0, ...] + Dual: [-1.3, 0.8, ...] h




# shows semantic similarity and epistemic tension

# A high real + low dual = “good match, low contradiction.” A low real + high dual = “epistemic tension — possibly a misleading match.”

# weighted by medical evidence


