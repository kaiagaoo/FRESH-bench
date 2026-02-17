# **Research Proposal: FreshRAG**

## **A Causal Benchmark for Understanding How Content Freshness Reduces Hallucination in AI Search Systems**

---

 **Date:** February 2026  
 **Advisor:** \[Advisor Name\]

---

## **Abstract**

While Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding Large Language Model (LLM) outputs in external knowledge, the relationship between content freshness and hallucination remains poorly understood. Existing benchmarks like HoH demonstrate that outdated information degrades RAG performance, but they do not isolate the causal mechanisms through which content freshness reduces hallucination. We propose **FreshRAG**, a novel benchmark designed to systematically investigate the causal pathways between content freshness and hallucination reduction in AI search systems. Our benchmark introduces three key innovations: (1) a **temporal gradient framework** that controls freshness as a continuous treatment variable, (2) a **multi-mechanism annotation schema** that labels the specific causal pathway through which freshness affects each query, and (3) a **counterfactual evaluation protocol** that measures what would have happened under different freshness conditions. FreshRAG will enable researchers to move beyond correlational findings ("fresh content improves accuracy") toward actionable causal insights ("fresh content reduces hallucination primarily by resolving knowledge conflicts, not by improving retrieval relevance").

---

## **1\. Introduction**

### **1.1 Motivation**

Large Language Models have revolutionized information retrieval, but they remain fundamentally limited by their parametric knowledge—facts encoded during training that inevitably become stale. Retrieval-Augmented Generation (RAG) addresses this by retrieving relevant documents at inference time, grounding model outputs in external knowledge sources.

However, a critical yet understudied problem emerges: **the knowledge bases that RAG systems retrieve from also contain outdated information**. When a user asks "Who is the CEO of OpenAI?", the retrieval system may return documents from different time periods—some current, some outdated—creating conflicting signals that can lead to hallucination.

Recent work by Ouyang et al. (2025) introduced HoH, the first benchmark to systematically study how outdated information affects RAG performance. Their findings are striking: even state-of-the-art models like Llama-70B experience a 24% performance drop when just one outdated passage is introduced. More concerning, outdated information is more harmful than missing information—models become confidently wrong rather than appropriately uncertain.

### **1.2 The Gap: Correlation vs. Causation**

While HoH demonstrates the *correlation* between outdated information and degraded performance, it does not answer the deeper question: **through what causal mechanisms does content freshness reduce hallucination?**

Consider three possible causal pathways:

1. **Knowledge Conflict Resolution**: Fresh content eliminates contradictions between retrieved passages, reducing the model's confusion about which information to trust.

2. **Temporal Grounding**: Fresh content provides explicit temporal signals (dates, "currently", "as of 2025") that help models identify which information is valid.

3. **Parametric Override**: Fresh content provides stronger evidence that overrides the model's outdated parametric knowledge, preventing the model from defaulting to stale training data.

Understanding *which* mechanism dominates—and under what conditions—is essential for designing effective content refresh strategies. A company with limited resources should know: Is it better to refresh content more frequently (mechanism 1), add explicit timestamps (mechanism 2), or focus on facts that contradict the model's training data (mechanism 3)?

### **1.3 Research Questions**

This proposal addresses three primary research questions:

**RQ1 (Causal Mechanisms):** Through what causal pathways does content freshness reduce hallucination in RAG systems? Can we decompose the total effect into distinct mechanisms?

**RQ2 (Moderating Factors):** How do domain characteristics (rate of change, factual density), query characteristics (specificity, temporal sensitivity), and model characteristics (size, recency of training) moderate the freshness-hallucination relationship?

**RQ3 (Optimal Refresh Strategies):** Given resource constraints, what content refresh strategies maximize hallucination reduction? Is there a point of diminishing returns?

### **1.4 Contributions**

This research will make the following contributions:

1. **FreshRAG Benchmark**: The first benchmark specifically designed to isolate causal mechanisms between content freshness and hallucination, featuring 50,000+ QA pairs with temporal gradient annotations and mechanism hypotheses.

2. **Hypothesis-Driven Validation Framework**: A rigorous methodology that generates mechanism hypotheses and validates them through controlled experiments, rather than relying on unverifiable annotations as ground truth.

3. **Mechanism Effect Quantification**: Controlled experiments that isolate and measure the actual effect of each causal pathway (KC, TS, PO), enabling evidence-based mechanism attribution.

4. **Empirical Findings**: Comprehensive experiments revealing which mechanisms dominate, under what conditions, and for which models—including novel insights on Grok-2's unique training on real-time data.

5. **Practical Guidelines**: Actionable recommendations for content refresh strategies in production AI search systems, grounded in validated causal understanding.

---

## **2\. Related Work**

### **2.1 Temporal Knowledge in LLMs**

The knowledge outdating problem in LLMs has been well-documented. Lazaridou et al. (2021) showed that language models struggle with time-sensitive facts, while Kasai et al. (2022) introduced RealTimeQA, a benchmark for evaluating models on questions about current events. FreshLLMs (Vu et al., 2024\) extended this by augmenting LLMs with search engine results, demonstrating significant improvements on time-sensitive queries.

However, these works focus on whether LLMs *can* answer time-sensitive questions, not on the mechanisms through which fresh information helps.

### **2.2 RAG Evaluation Benchmarks**

The RAG evaluation landscape has expanded rapidly. KILT (Petroni et al., 2021\) provides a unified benchmark for knowledge-intensive tasks, while BEIR (Thakur et al., 2021\) evaluates retrieval across diverse domains. More recently, CRAG (Yang et al., 2024\) introduced a comprehensive RAG benchmark with complex queries.

Most relevant to our work is HoH (Ouyang et al., 2025), which specifically evaluates RAG robustness against outdated information. HoH makes two key contributions: (1) a dynamic QA dataset that tracks factual changes over time, and (2) a mock search engine that maintains both current and historical documents. Our work builds directly on HoH but shifts focus from measuring *impact* to understanding *mechanisms*.

### **2.3 Causal Inference in NLP**

Causal inference methods have been increasingly applied to NLP. Feder et al. (2021) survey causal inference for NLP, emphasizing the importance of counterfactual reasoning. Keith et al. (2020) discuss causal inference from text, while Veitch et al. (2021) use text embeddings for causal inference in observational studies.

Our work applies causal thinking to the freshness-hallucination relationship, designing experiments that can isolate distinct causal pathways.

### **2.4 Hallucination in LLMs**

Hallucination—generating plausible but factually incorrect content—remains a critical challenge. Ji et al. (2023) provide a comprehensive survey, categorizing hallucinations as intrinsic (contradicting the source) or extrinsic (unverifiable from the source). Manakul et al. (2023) introduce SelfCheckGPT for detecting hallucinations, while Min et al. (2023) study factuality in long-form generation.

Our work connects hallucination research with temporal knowledge, investigating how content freshness specifically affects different types of hallucination.

---

## **3\. FreshRAG Benchmark Design**

### **3.1 Overview**

FreshRAG consists of three integrated components:

1. **FreshRAG-QA**: A dynamic question-answering dataset with temporal gradient annotations  
2. **FreshRAG-Corpus**: A document collection with controlled freshness variations  
3. **FreshRAG-Eval**: A counterfactual evaluation protocol for causal analysis

### **3.2 Causal Framework**

We formalize the freshness-hallucination relationship using a causal graph:

Freshness (F) ──┬──→ Knowledge Conflict (KC) ──→ Hallucination (H)  
                │  
                ├──→ Temporal Signals (TS) ────→ Hallucination (H)  
                │  
                └──→ Parametric Override (PO) ─→ Hallucination (H)

Where:

* **F (Freshness)**: The recency of retrieved content relative to the query time  
* **KC (Knowledge Conflict)**: Whether retrieved passages contain contradictory information  
* **TS (Temporal Signals)**: Presence of explicit temporal markers in content  
* **PO (Parametric Override)**: Whether retrieved content contradicts model's parametric knowledge  
* **H (Hallucination)**: Whether the model generates factually incorrect output

This framework enables us to decompose the total effect of freshness on hallucination into three distinct pathways.

### **3.3 Hypothesis-Driven Validation Approach**

A key methodological challenge is that causal mechanisms are not directly observable—we cannot simply "label" which mechanism is at play for a given QA pair. Instead, we adopt a **hypothesis-driven approach**:

1. **Hypothesis Generation**: For each QA pair, we use LLM to *predict* which mechanism is likely dominant based on observable features (e.g., entity fame, presence of conflicting sources online, temporal language in documents).

2. **Experimental Validation**: We design controlled experiments that isolate each mechanism's effect:

   * **KC Effect**: Compare performance with vs. without conflicting documents  
   * **TS Effect**: Compare performance with vs. without timestamps  
   * **PO Effect**: Compare retrieval-augmented vs. parametric-only responses  
3. **Mechanism Attribution**: For each QA pair, we identify the *actual* dominant mechanism based on which experimental manipulation produces the largest effect.

4. **Hypothesis Validation**: We report the agreement between predicted and actual dominant mechanisms, treating this as an empirical finding rather than assuming annotation accuracy.

This approach is more scientifically rigorous than treating LLM annotations as ground truth, as we validate hypotheses through experimental evidence rather than relying on unverifiable labels.

### **3.3 FreshRAG-QA: Dataset Construction**

#### **3.3.1 Data Sources**

We will collect data from multiple sources to ensure domain diversity:

| Domain | Source | Change Rate | Example Facts |
| ----- | ----- | ----- | ----- |
| People | Wikipedia \+ Wikidata | Medium | Job titles, affiliations, relationships |
| Organizations | Wikipedia \+ SEC filings | Medium | Leadership, financials, products |
| Sports | Wikipedia \+ ESPN | High | Records, rankings, team rosters |
| Science | Wikipedia \+ arXiv | Low | Discoveries, consensus positions |
| Policy | Government websites | Medium | Laws, regulations, officials |
| Products | Company websites | High | Prices, features, availability |

#### **3.3.2 Temporal Gradient Construction**

For each factual change, we construct a temporal gradient with 5 time points:

T-4: 12+ months before change (stable old fact)  
T-3: 6 months before change  
T-2: 1 month before change  
T-1: 1 week before change  
T0:  Change occurs  
T+1: 1 week after change  
T+2: 1 month after change  
T+3: 6 months after change (stable new fact)

This enables us to study:

* Pre-change stability (T-4 to T-1)  
* Transition period dynamics (T-1 to T+1)  
* Post-change stability (T+1 to T+3)

#### **3.3.3 Question Generation Pipeline**

We adapt HoH's diff \+ LLM approach with extensions for causal annotation:

**Step 1: Factual Change Detection**

\# Use token-level diff on Wikipedia snapshots  
changes \= diff(wiki\_snapshot\_t1, wiki\_snapshot\_t2)

\# Filter for semantic changes (not just wording)  
factual\_changes \= llm\_filter(changes, prompt="""  
    Classify this change:  
    \- FACTUAL\_UPDATE: Core fact changed (e.g., CEO changed)  
    \- NUMERIC\_UPDATE: Number changed (e.g., population)  
    \- ADDITION: New information added  
    \- DELETION: Information removed  
    \- WORDING: Same fact, different words  
    Return FACTUAL\_UPDATE, NUMERIC\_UPDATE, ADDITION, or DELETION only.  
""")

**Step 2: Question Generation**

for change in factual\_changes:  
    question \= llm\_generate(prompt=f"""  
        Generate a natural question that would be answered differently   
        before vs. after this change:  
          
        Old text: {change.old\_text}  
        New text: {change.new\_text}  
          
        Requirements:  
        \- Question should be something a real user might ask  
        \- Answer should be unambiguous at each time point  
        \- Avoid questions that explicitly reference time  
    """)  
      
    old\_answer \= extract\_answer(change.old\_text, question)  
    new\_answer \= extract\_answer(change.new\_text, question)

**Step 3: Mechanism Hypothesis Generation**

Rather than treating LLM predictions as ground truth labels, we generate *hypotheses* about which mechanism is likely dominant for each QA pair. These hypotheses will be validated through controlled experiments.

for qa\_pair in generated\_pairs:  
    hypothesis \= llm\_predict(prompt=f"""  
        This fact changed: {qa\_pair.old\_answer} → {qa\_pair.new\_answer}  
          
        Based on observable features, PREDICT which mechanism is most likely   
        to be important for fresh content to help a model answer correctly:  
          
        A) CONFLICT\_RESOLUTION: Multiple sources likely exist online with   
           different answers; fresh content would clarify which is current  
        B) TEMPORAL\_GROUNDING: Fresh content would contain explicit time   
           markers that help identify validity  
        C) PARAMETRIC\_OVERRIDE: The old fact is likely in LLM training data;   
           fresh content would need to override parametric knowledge  
        D) COVERAGE: The new information is genuinely new (not replacing old);   
           this is about availability, not freshness  
          
        Observable indicators to consider:  
        \- Is this entity/fact famous? (suggests PO)  
        \- Are there likely many dated sources online? (suggests KC)  
        \- Would timestamps significantly disambiguate? (suggests TS)  
          
        Return your PREDICTION (to be validated experimentally):  
    """)

**Important**: These are hypotheses, not ground truth. We will validate them by measuring actual mechanism effects through controlled experiments (Section 4).

#### **3.3.4 Quality Control**

* **Human Validation**: Sample 10% of generated QA pairs for human review  
* **Answer Verification**: Cross-check answers against Wikidata for entity facts  
* **Temporal Consistency**: Verify that answers are indeed different at T-1 vs T+1  
* **Naturalness Filtering**: Remove questions that sound artificial or contrived

#### **3.3.5 Dataset Statistics (Target)**

| Metric | Target |
| ----- | ----- |
| Total QA pairs | 50,000 |
| Unique factual changes | 15,000 |
| Domains covered | 6 |
| Time span | 2020-2025 |
| Avg. questions per change | 3.3 |
| Human-validated subset | 5,000 |

### **3.4 FreshRAG-Corpus: Document Collection**

#### **3.4.1 Multi-Temporal Document Store**

For each factual change, we maintain documents from multiple time points:

Document Store Structure:  
├── entity\_123/  
│   ├── 2023-01-01\_wikipedia.txt  
│   ├── 2023-06-01\_wikipedia.txt  
│   ├── 2024-01-01\_wikipedia.txt  
│   ├── 2024-06-01\_wikipedia.txt  
│   └── 2025-01-01\_wikipedia.txt

#### **3.4.2 Controlled Freshness Scenarios**

We define 5 retrieval scenarios to isolate causal mechanisms:

| Scenario | Retrieved Documents | Tests |
| ----- | ----- | ----- |
| S1: Fresh Only | Only T+3 documents | Baseline with perfect freshness |
| S2: Stale Only | Only T-4 documents | Baseline with outdated info |
| S3: Mixed (Fresh Dominant) | 3 fresh \+ 1 stale | Knowledge conflict with fresh majority |
| S4: Mixed (Stale Dominant) | 1 fresh \+ 3 stale | Knowledge conflict with stale majority |
| S5: Mixed (Equal) | 2 fresh \+ 2 stale | Maximum knowledge conflict |

#### **3.4.3 Timestamp Manipulation**

To isolate the temporal grounding mechanism, we create document variants:

* **With Timestamps**: "As of January 2025, the CEO is..."  
* **Without Timestamps**: "The CEO is..."  
* **Misleading Timestamps**: Old content with recent dates (adversarial)

### **3.5 FreshRAG-Eval: Counterfactual Evaluation Protocol**

#### **3.5.1 Evaluation Metrics**

**Primary Metrics:**

**Factual Accuracy (FA)**: Whether the answer matches ground truth at query time

 FA \= 1 if answer matches current\_ground\_truth else 0

1. 

**Hallucination Rate (HR)**: Proportion of confident but incorrect answers

 HR \= P(confident AND incorrect) / P(confident)

2. 

**Appropriate Uncertainty (AU)**: Whether model expresses uncertainty when it should

 AU \= P(uncertain | conflicting\_evidence)

3. 

**Mechanism-Specific Metrics:**

**Conflict Resolution Score (CRS)**: Accuracy improvement from S5 to S1

 CRS \= FA(S1) \- FA(S5)  \# Higher \= better at resolving conflicts

4. 

**Temporal Grounding Score (TGS)**: Accuracy improvement from adding timestamps

 TGS \= FA(with\_timestamps) \- FA(without\_timestamps)

5. 

**Parametric Override Score (POS)**: Accuracy on facts that contradict training data

 POS \= FA | (fact contradicts parametric knowledge)

6. 

#### **3.5.2 Mechanism Validation Protocol**

Since causal mechanisms cannot be directly observed, we validate mechanism hypotheses through experimental manipulation:

**For each QA pair, compute actual mechanism effects:**

def compute\_actual\_mechanism\_effects(qa\_pair, model):  
    """  
    Run controlled experiments to measure actual effect of each mechanism.  
    """  
      
    \# KC Effect: Does removing conflict improve accuracy?  
    KC\_effect \= (  
        evaluate(model, qa\_pair, scenario='S1\_no\_conflict') \-  
        evaluate(model, qa\_pair, scenario='S5\_max\_conflict')  
    )  
      
    \# TS Effect: Do timestamps improve accuracy?  
    TS\_effect \= (  
        evaluate(model, qa\_pair, scenario='S1\_with\_timestamp') \-  
        evaluate(model, qa\_pair, scenario='S1\_no\_timestamp')  
    )  
      
    \# PO Effect: Does retrieval override parametric knowledge?  
    \# First, get model's parametric answer (no retrieval)  
    parametric\_answer \= model.generate(qa\_pair.question, context=None)  
    parametric\_correct \= is\_correct(parametric\_answer, qa\_pair.new\_answer)  
      
    \# Then, get retrieval-augmented answer  
    retrieval\_answer \= model.generate(qa\_pair.question, context=fresh\_docs)  
    retrieval\_correct \= is\_correct(retrieval\_answer, qa\_pair.new\_answer)  
      
    \# PO effect is only relevant when parametric answer is wrong  
    if not parametric\_correct:  
        PO\_effect \= retrieval\_correct \- parametric\_correct  \# 0 or 1  
    else:  
        PO\_effect \= 0  \# No override needed  
      
    return {  
        'KC\_effect': KC\_effect,  
        'TS\_effect': TS\_effect,  
        'PO\_effect': PO\_effect,  
        'actual\_dominant': get\_dominant\_mechanism(KC\_effect, TS\_effect, PO\_effect)  
    }

**Validate hypotheses against actual effects:**

def validate\_mechanism\_hypotheses(qa\_pairs, results):  
    """  
    Compare predicted mechanisms with experimentally-determined actual mechanisms.  
    """  
      
    predictions \= \[qa\['predicted\_mechanism'\] for qa in qa\_pairs\]  
    actuals \= \[results\[qa\['id'\]\]\['actual\_dominant'\] for qa in qa\_pairs\]  
      
    \# Overall hypothesis accuracy  
    accuracy \= sum(p \== a for p, a in zip(predictions, actuals)) / len(predictions)  
      
    \# Per-mechanism precision/recall  
    for mechanism in \['KC', 'TS', 'PO'\]:  
        precision \= precision\_score(actuals, predictions, pos\_label=mechanism)  
        recall \= recall\_score(actuals, predictions, pos\_label=mechanism)  
          
    \# Confusion matrix: which mechanisms are confused with which?  
    confusion \= confusion\_matrix(actuals, predictions)  
      
    return {  
        'hypothesis\_accuracy': accuracy,  
        'confusion\_matrix': confusion,  
        'per\_mechanism\_metrics': {...}  
    }

**Key Metrics:**

| Metric | Description | Purpose |
| ----- | ----- | ----- |
| Hypothesis Accuracy | % of cases where predicted \= actual dominant mechanism | Validate LLM prediction quality |
| Mechanism Effect Size | Cohen's d for each mechanism's experimental effect | Quantify mechanism importance |
| Effect Consistency | Correlation of effects across models | Test if mechanisms are model-agnostic |

#### **3.5.3 Counterfactual Analysis**

For each QA pair, we compute counterfactual outcomes:

def counterfactual\_analysis(qa\_pair, model):  
    results \= {}  
      
    \# Factual outcome  
    results\['fresh\_only'\] \= evaluate(model, qa\_pair, scenario='S1')  
    results\['stale\_only'\] \= evaluate(model, qa\_pair, scenario='S2')  
      
    \# Counterfactual: What if we had fresh content?  
    results\['counterfactual\_gain'\] \= results\['fresh\_only'\] \- results\['stale\_only'\]  
      
    \# Mechanism attribution  
    results\['conflict\_effect'\] \= evaluate(model, qa\_pair, scenario='S1') \- \\  
                                  evaluate(model, qa\_pair, scenario='S5')  
    results\['timestamp\_effect'\] \= evaluate(model, qa\_pair, scenario='S1\_with\_ts') \- \\  
                                   evaluate(model, qa\_pair, scenario='S1\_no\_ts')  
      
    return results

#### **3.5.3 Causal Effect Decomposition**

Using the causal framework, we decompose total freshness effect:

Total Effect \= Direct Effect \+ Indirect Effects

TE(F→H) \= DE(F→H) \+ IE(F→KC→H) \+ IE(F→TS→H) \+ IE(F→PO→H)

We estimate each component through controlled experiments:

1. **IE(F→KC→H)**: Compare S1 vs S5 (varying conflict, holding timestamps constant)  
2. **IE(F→TS→H)**: Compare with vs without timestamps (holding content constant)  
3. **IE(F→PO→H)**: Compare facts in vs not in training data (stratified analysis)

---

## **4\. Experimental Design**

### **4.1 Models Under Evaluation**

We will evaluate models across different scales and architectures:

**Closed-Source:**

* GPT-4o  
* GPT-4o-mini  
* Claude 3.5 Sonnet  
* Gemini 1.5 Pro  
* Grok-2 (xAI) — *particularly interesting due to training on real-time X/Twitter data*

**Open-Source:**

* Llama 3.1 (8B, 70B)  
* Qwen 2.5 (7B, 72B)  
* Mistral Large

**Note on Grok:** xAI's Grok models are trained with access to real-time X (Twitter) data, making them a unique test case for our freshness study. We hypothesize that Grok may show different patterns in the Parametric Override (PO) mechanism, as its training data is more recent than other models.

### **4.2 Retrieval Systems**

We test multiple retrieval approaches:

1. **Dense Retrieval**: Contriever, BGE-large  
2. **Sparse Retrieval**: BM25  
3. **Hybrid**: RRF combination of dense \+ sparse  
4. **Time-Aware**: BM25 with Gaussian temporal decay (HoH's approach)

### **4.3 Experimental Conditions**

**Experiment 1: Mechanism Isolation**

* Goal: Quantify contribution of each causal pathway  
* Design: Full factorial across scenarios S1-S5 × timestamp conditions  
* Analysis: ANOVA with mechanism as factor

**Experiment 2: Temporal Gradient Effects**

* Goal: Characterize freshness decay curve  
* Design: Vary document age from T-4 to T+3  
* Analysis: Regression with age as continuous predictor

**Experiment 3: Domain Heterogeneity**

* Goal: Identify domain-specific freshness sensitivities  
* Design: Stratified analysis by domain  
* Analysis: Mixed-effects models with domain as random effect

**Experiment 4: Model Scaling**

* Goal: Test whether larger models handle staleness better  
* Design: Compare across model sizes within families  
* Analysis: Scaling law regression

**Experiment 5: Adversarial Freshness**

* Goal: Test robustness to misleading temporal signals  
* Design: Old content with fake recent timestamps  
* Analysis: Compare adversarial vs honest timestamp conditions

### **4.4 Statistical Analysis Plan**

* **Effect Sizes**: Report Cohen's d for all comparisons  
* **Confidence Intervals**: Bootstrap 95% CIs for all metrics  
* **Multiple Comparisons**: Bonferroni correction for family-wise error  
* **Causal Identification**: Use do-calculus to verify identifiability of causal effects

---

## **5\. Expected Outcomes and Impact**

### **5.1 Expected Findings**

Based on pilot studies and related work, we hypothesize:

**H1**: Knowledge conflict resolution (KC) is the dominant mechanism, accounting for \>50% of the total freshness effect.

**H2**: Temporal grounding (TS) has a stronger effect for larger models, as they better utilize explicit temporal signals.

**H3**: Parametric override (PO) is most important for "famous" facts (e.g., CEO of major companies) that are likely in training data.

**H4**: The freshness-hallucination relationship follows a logarithmic decay curve, with most benefit from the first week of freshness.

**H5 (Methodological)**: LLM-generated mechanism hypotheses will achieve 60-70% accuracy when validated against experimental results, with systematic patterns in prediction errors (e.g., over-predicting PO for famous entities).

**H6 (Grok-specific)**: Grok-2, trained on real-time X/Twitter data, will show different PO patterns than other models—specifically, lower PO effect for social-media-discussed facts due to more recent parametric knowledge.

### **5.2 Practical Implications**

Our findings will inform content refresh strategies:

1. **Prioritization**: If KC dominates, prioritize refreshing facts with multiple conflicting sources online  
2. **Timestamp Strategy**: If TS is significant, adding explicit timestamps may be a low-cost intervention  
3. **Frequency Optimization**: The decay curve will indicate optimal refresh intervals  
4. **Domain-Specific Policies**: Different domains may require different refresh cadences

### **5.3 Broader Impact**

**For Researchers:**

* A rigorous benchmark for studying temporal dynamics in RAG  
* A causal framework applicable to other RAG robustness questions

**For Practitioners:**

* Evidence-based guidelines for content maintenance  
* Cost-benefit framework for refresh investments

**For the Field:**

* Advancing from correlational to causal understanding of RAG behavior  
* Connecting RAG evaluation with causal inference methodology

---

## **6\. Timeline and Milestones**

| Phase | Duration | Milestones |
| ----- | ----- | ----- |
| **Phase 1: Data Collection** | Months 1-3 | • Collect Wikipedia snapshots (2020-2025) • Set up diff pipeline • Initial factual change extraction |
| **Phase 2: Dataset Construction** | Months 3-5 | • Generate QA pairs • Annotate causal mechanisms • Human validation of 5K samples |
| **Phase 3: Benchmark Infrastructure** | Months 5-6 | • Build FreshRAG-Corpus • Implement retrieval scenarios • Develop evaluation harness |
| **Phase 4: Experiments** | Months 6-8 | • Run all 5 experiments • Statistical analysis • Iterate on findings |
| **Phase 5: Analysis & Writing** | Months 8-10 | • Causal effect decomposition • Practical guidelines • Paper writing |
| **Phase 6: Release** | Month 10 | • Open-source benchmark • Documentation • Community feedback |

---

## **7\. Resources Required**

### **7.1 Compute**

* **LLM API Costs**: \~$5,000 for question generation and mechanism annotation  
* **Model Evaluation**: \~$3,000 for closed-source model APIs  
* **Open-Source Inference**: 200 GPU-hours on A100 (available through Berkeley cluster)

### **7.2 Data**

* Wikipedia dumps: Free (Wikimedia Foundation)  
* Wikidata: Free (CC0 license)  
* Additional sources: May require web scraping or API access

### **7.3 Human Annotation**

* 5,000 QA pairs × 3 annotators × $0.50/annotation \= $7,500  
* Platform: Prolific or MTurk with quality controls

### **7.4 Total Budget**

| Category | Cost |
| ----- | ----- |
| LLM APIs | $8,000 |
| Human Annotation | $7,500 |
| Miscellaneous | $1,500 |
| **Total** | **$17,000** |

---

## **8\. Risks and Mitigation**

| Risk | Likelihood | Impact | Mitigation |
| ----- | ----- | ----- | ----- |
| Insufficient factual changes in data | Low | High | Use multiple sources; extend time range |
| Annotation quality issues | Medium | Medium | Multiple annotators; clear guidelines; pilot study |
| Causal mechanisms not separable | Medium | High | Design additional controlled experiments; accept partial identification |
| API cost overruns | Low | Low | Budget buffer; use open-source models where possible |
| Timeline delays | Medium | Medium | Prioritize core experiments; defer extensions |

---

## **9\. Ethical Considerations**

### **9.1 Data Privacy**

* Use only publicly available data (Wikipedia, public websites)  
* No personal information beyond public figures

### **9.2 Potential Misuse**

* Benchmark could be used to exploit model weaknesses  
* Mitigation: Focus on defense (improving freshness) not attack

### **9.3 Bias Considerations**

* Wikipedia has known biases in coverage  
* Mitigation: Stratify analysis by domain; acknowledge limitations

### **9.4 Environmental Impact**

* LLM inference has carbon footprint  
* Mitigation: Efficient experimental design; reuse cached results

---

## **10\. Conclusion**

Understanding the causal mechanisms through which content freshness reduces hallucination is essential for building robust AI search systems. FreshRAG will provide the research community with a rigorous benchmark for investigating these mechanisms, moving beyond correlational findings toward actionable causal insights. By decomposing the total freshness effect into distinct pathways—knowledge conflict resolution, temporal grounding, and parametric override—we will enable more targeted and cost-effective content refresh strategies.

The benchmark, along with our empirical findings and practical guidelines, will be released as open-source resources, contributing to the broader goal of building AI systems that provide accurate, up-to-date information to users.

---

## **References**

Feder, A., Keith, K. A., Manzoor, E., Pryzant, R., Sridhar, D., Wood-Doughty, Z., ... & Yang, D. (2021). Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. arXiv preprint arXiv:2109.00725.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 1-38.

Kasai, J., Kasai, K., Qin, J., Wang, L., & Zettlemoyer, L. (2022). RealTime QA: What's the answer right now? arXiv preprint arXiv:2207.13332.

Keith, K. A., Jensen, D., & O'Connor, B. (2020). Text and causal inference: A review of using text to remove confounding from causal estimates. ACL 2020\.

Lazaridou, A., Kuncoro, A., Gribovskaya, E., Agrawal, D., Liska, A., Terber, T., ... & Blunsom, P. (2021). Mind the gap: Assessing temporal generalization in neural language models. NeurIPS 2021\.

Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. EMNLP 2023\.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W. T., Koh, P. W., ... & Hajishirzi, H. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. EMNLP 2023\.

Ouyang, J., Pan, T., Cheng, M., Yan, R., Luo, Y., Lin, J., & Liu, Q. (2025). HoH: A dynamic benchmark for evaluating the impact of outdated information on retrieval-augmented generation. ACL 2025\.

Petroni, F., Piktus, A., Fan, A., Lewis, P., Yazdani, M., De Cao, N., ... & Riedel, S. (2021). KILT: A benchmark for knowledge intensive language tasks. NAACL 2021\.

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. NeurIPS 2021 Datasets and Benchmarks.

Veitch, V., Sridhar, D., & Blei, D. (2021). Using text embeddings for causal inference. AISTATS 2021\.

Vu, T., Iyyer, M., Wang, X., Constant, N., Wei, J., Wei, J., ... & Luong, T. (2024). FreshLLMs: Refreshing large language models with search engine augmentation. ACL 2024\.

Yang, X., Sun, K., Xin, H., Sun, Y., Bhalla, N., Chen, X., ... & Jiang, Z. (2024). CRAG: Comprehensive RAG benchmark. arXiv preprint arXiv:2406.04744.

---

## **Appendix A: Causal Mechanism Taxonomy**

### **A.1 Knowledge Conflict Resolution (KC)**

**Definition**: Fresh content helps by eliminating or clarifying conflicts between multiple retrieved documents that contain different answers to the same question.

**Indicators**:

* Multiple documents exist online with different answers  
* The fact has changed at least once in the past 2 years  
* Search results would naturally return a mix of old and new content

**Example**:

* Question: "Who is the CEO of Twitter?"  
* Old answer (in many documents): "Jack Dorsey" or "Parag Agrawal"  
* New answer: "Linda Yaccarino"  
* Mechanism: Fresh content clarifies that Linda Yaccarino is the *current* CEO

### **A.2 Temporal Grounding (TS)**

**Definition**: Fresh content helps by providing explicit temporal signals (dates, "currently", "as of 2025") that help models identify which information is valid at query time.

**Indicators**:

* Fresh documents contain explicit timestamps or temporal language  
* The model can correctly answer if given temporal context  
* Removing temporal markers degrades performance

**Example**:

* Document A: "The population is 8.3 million"  
* Document B: "As of 2024, the population is 8.5 million"  
* Mechanism: Document B's explicit date helps the model trust it over Document A

### **A.3 Parametric Override (PO)**

**Definition**: Fresh content helps by providing evidence strong enough to override the model's parametric knowledge (facts encoded during training).

**Indicators**:

* The old fact is likely in the model's training data (famous entity, widely reported)  
* Without retrieval, the model would answer with the old fact  
* Fresh content must "convince" the model to change its answer

**Example**:

* Question: "What is the latest iPhone model?"  
* Model's parametric knowledge: "iPhone 15" (from training data)  
* Fresh content needed to update to: "iPhone 16"  
* Mechanism: Strong evidence in retrieved content overrides training

### **A.4 Coverage (C)**

**Definition**: Fresh content helps simply because the information didn't exist before—not about freshness per se, but about availability.

**Indicators**:

* The fact is genuinely new (new product, new discovery)  
* There are no "old" versions of this fact  
* This is not a case of outdated → updated, but absent → present

**Example**:

* Question: "What is Apple's new Vision Pro feature announced in 2025?"  
* Mechanism: This information simply didn't exist before; it's not replacing outdated info

**Note**: Coverage cases are included for completeness but are analyzed separately, as they don't involve the freshness-staleness dynamic that is the focus of this benchmark.

---

## **Appendix B: Annotation Guidelines**

### **B.1 Task Overview**

You will review question-answer pairs derived from factual changes and assign:

1. A primary causal mechanism (KC, TS, PO, or C)  
2. A confidence score (1-5)  
3. Optional secondary mechanism if applicable

### **B.2 Decision Flowchart**

Is this a NEW fact (never existed before)?  
├── Yes → COVERAGE (C)  
└── No → Continue

Is the OLD fact likely in major LLMs' training data?  
├── Yes → Likely PARAMETRIC\_OVERRIDE (PO)  
├── No → Likely CONFLICT\_RESOLUTION (KC)  
└── Unsure → Continue

Would adding a timestamp to documents significantly help?  
├── Yes → Likely TEMPORAL\_GROUNDING (TS)  
└── No → Default to KC or PO based on above

### **B.3 Examples with Annotations**

\[Detailed examples would be provided to annotators\]

---

## **Appendix C: Dataset Sample**

### **C.1 Sample QA Pair**

{  
  "id": "freshrag\_00001",  
  "question": "Who is the CEO of OpenAI?",  
  "entity": "OpenAI",  
  "domain": "Organizations",  
    
  "temporal\_data": {  
    "change\_date": "2024-11-17",  
    "old\_answer": "Sam Altman",  
    "old\_answer\_valid\_until": "2024-11-17",  
    "interim\_answer": "Mira Murati (interim)",  
    "interim\_valid": "2024-11-17 to 2024-11-21",  
    "new\_answer": "Sam Altman",  
    "new\_answer\_valid\_from": "2024-11-21"  
  },  
    
  "causal\_annotation": {  
    "primary\_mechanism": "PARAMETRIC\_OVERRIDE",  
    "confidence": 4,  
    "secondary\_mechanism": "CONFLICT\_RESOLUTION",  
    "rationale": "Sam Altman as OpenAI CEO is widely known and likely in training data. During the brief period of change, models would need strong evidence to override this. Additionally, many conflicting news articles exist from this period."  
  },  
    
  "documents": {  
    "T-4": "doc\_openai\_2023\_01.txt",  
    "T-1": "doc\_openai\_2024\_11\_15.txt",  
    "T+1": "doc\_openai\_2024\_11\_20.txt",  
    "T+3": "doc\_openai\_2025\_01.txt"  
  }  
}

---

*End of Proposal*

