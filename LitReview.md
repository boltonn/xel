# NLP Review

## Cross-lingual Papers
1. **Entity Linking**
    * **Unsupervised Entity Linking with Abstract Meaning Representation**: [[paper]](https://www.aclweb.org/anthology/N15-1119.pdf)
        * manually map AMR entity to DBpedia entity types
        * use a set of rules to construct a *Knowledge Network* where the mention in English is linked to other nodes (entities, categories, times) mostly by AMR relationships but also things like DBpedia and Freebas typed relationships, typed hyperlinks, etc.
        * for linking they use the jaccard similarity of the tokens (stopwords removed). it then re-ranked doing the same thing with coherent mentions or other entities within a window (this makes it more robust to overall popularity and context of other entities)
    * **Cross-lingual Name Tagging and Linking for 282 Languages**: [[paper]](https://www.aclweb.org/anthology/P17-1178.pdf)
        * create Wiki dataset: 2756 entity mentions w/ one of 139 entity types from AMR corpus (in English); they map those to YAGO entity types by taking highest Pointwise Mutual Information (PMI) across all AMR entity types; then they use 62k DBpedia properties as features to classify Wikipedia pages as an entity type (ex: if population is a propert, it likely a GPE entity type). Doing this across three levels (conll, AMR, YAGO) they get 10mil English wiki pages. Note that AMR is only in English at the time and unrepresentative.
        * they use LSTM+CRF w/ some heuristic features (stemming, capitalization, etc.) for NER
        * for candidate generation they take all ngram token/word combinations of the source name mention, then use GIZA++ to get word for word translations to the target language (English), $t_i$. Candidates are obtained on what they call a *surface form dictionary* from the KB properties (redirects, aliases, names, acronyms, etc.), called $e_j$. I think this means exact match with maybe some lemmatization beforehand
        * for linking they use the [[paper above]](https://www.aclweb.org/anthology/N15-1119.pdf) which computes similarity between all possible Knowledge Networks of $t_i$ and $e_j$, denoted $g(t_i)$ and $g(e_j)$ based on saliency, similarity, and coherence
        * found 10k mentions lead to above 80% F-score
    * **Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation**: [[paper]](https://arxiv.org/pdf/1601.01343v4.pdf)
        * use a skip-gram model to map words and entities in the same semantic space
            1) word2vec based on text 2) similar to Wikipedia Link-based Measure (WLM) to get relatedness among entities or coherence based on links, this is trained as predicting an entity given it's links 3) anchor context model to bridge text and entity embeddings by also encorporating the sum of the word2vec embeddings of the other enity/anchors in the text and their context within a window
        * also use a prior on how prevalent the entity is in the Wikipedia corpus
    * **MAG**: A Multilingual, Knowledge-base Agnostic and Deterministic Entity Linking Approach: [[paper]](https://arxiv.org/pdf/1707.05288v3.pdf)
        * candidate generation: preprocessing (remove punctuation, adds space between lower and upper case, etc.); acronyms map to all possible entities; tri-gram similarity post stemming which based off pseudo code looks like the entities the mention and possible candidate have in common (this is done after the preprocessing&acronyms and then also one based on context or tf-idf)
        * then for disambiguation they use the KB to form a graph and use PageRank to rank the candidates
        * downside of this seems to Low Resource Languages (LRL) since you're only working in the source language
    * **Neural Cross-Lingual Entity Linking**: [[paper]](https://arxiv.org/pdf/1712.01813v1.pdf)
        * system only (or at least mostly) based on contextual embeddings of the mention via its context within a window and the potential candidate's embedding using the source page (for embeddings they use LSTMs w/ **Multilingual Canonical Correlation Analysis** or MultiCCA to project monolingual embeddings into the same space)
    * **End-to-End Neural Entity Linking**: [[paper]](https://arxiv.org/pdf/1808.07699v2.pdf)
        * for disambiguation they embed the mention using a lstm-based contextual embedding which are then compared to word2vec embeddings of candidates using dot product similarity for what they call a *local score*; they then keep these $e$ and $m$ combinations and get some semblence of coherence or a *global score* by taking the average of mention embeddings and taking a similarity for each embedding
        * candidate selection is using a pre-defined prior $P(e|m)$ from this [[paper]](https://www.aclweb.org/anthology/D17-1277.pdf) and they found a coreference heuristic to help 1-5%
    * **XLEMS**- Joint Multilingual Supervision for Cross-lingual Entity Linking: [[paper]](https://arxiv.org/pdf/1809.07657v1.pdf) *
        * their model encodes the mention using a *local context* (they used fasttext which is a CNN on token n-grams) and concatenate this with a *document context* which is sum linear combination of all other mention embeddings in the documents. These are concatenated to form the mention embedding
        * the *Entity Context (EC) Loss* is then the softmax over dot product of mention and entity embedding; then two more losses  *Type Context (TC) Loss* and  *Type Entity (TE) Loss* are incorporated to capture the prediction of the mention type (ie. sports_team) (this is the binary crossentropy and the entity type is assumed to be the same as the mention type; also the types come from Freebase or YAGO)
        * for candidate generation they use prior probabilites which again is just the counts for each time a mention maps to a title, and if this doesnt worj they have a second dictionary where they do the same thing but for tokens of the mention [[paper]](https://www.aclweb.org/anthology/N16-1072.pdf); they also point out that transliteration could help recall
    * **PBEL**- Zero-shot Neural Transfer for Cross-lingual Entity Linking: [[paper]](https://arxiv.org/pdf/1811.04154v1.pdf) *
        * *PBEL* (Pivot Based Entity Linking): idea is that because bilingual lexicons work well for HRL but poorly for for LRL, they use a bilingual lexicon to transfer the mention to the closest related HRL (typically in same language family with same alphabet) and then use neural models to link those enties to English e.g. Marathi to Hindi using lexicon and then Hindi to English using the cross-lingual model
        * they actually use a bidirectional character LSTM on $m$, $e_{HRL}$, and $e_{EN}$. Then they take the cosine similarities $sim(m, e_{HRL})$ and $sim(e_{HRL}, e_{EN})$ and then take the maximum  (if no mapping $e_{HRL} -> e_{EN}$ exists then first sim is set to $-\infty$); with this as the loss they treat the true entity, $e_{EN}$, as the positive and every other entity in the batch as a negative. Also recognizing the alphabets are different, they train the LSTMs on phonemes (converted into IPA using Epitran)
        * to get the closest HRL to the source language they use the URIEL databases; also they use PanPhon which add some more phonetic features not captured in IPA like nasal, strident, voice, etc.
        * seem to be a lot of downsides like not taking context into account, not using coherence or other mentions, and still relying on good links between the HRL and English, but still very unique
    * **BURN**- Towards Zero-resource Cross-lingual Entity Linking: [[paper]](https://www.aclweb.org/anthology/D19-6127.pdf), [[code]](https://github.com/shuyanzhou/pbel_plus)
        * candidate generation:
            * *WIKIMENTION* (more accurate)- is similar to the prior method where we find the entity in the source language most heavily linked to the mention in the source language and then use the mapping of $e_{SRC}$ to  $e_{EN}$; very heavily correlated w/ amount of data so LRL recalls are very bad (<.5) typically
            * *PIVOTING* (more robust)- like above where we utilize the closest HRL and get a mapping to English usually in the form of an LSTM
            * b/c *PIVOTING* isnt probabilistic they suggest taking a weighted softmax over top-*n* candidates and then you can average w/ *WIKIMENTION*
            * improves *gold candidate recall* by as much as 15%
        * disambiguation:
            * *unary* (considering all candidate entites): <br>
                1) target entity's co-ocurrances with the candidate entities<br>
                2) prior- number of times the entity is in the dataset (salience)<br>
                3) exact match- mention coreference w/ entity
            * *binary* (considering candidate entity and other linked entities in the doc): <br>
                1) co-occurance<br>
                2) Positive Pointwise Mutual Information (PPMI) [[paper]](https://dl.acm.org/doi/pdf/10.5555/89086.89095)<br>
                3) word2vec similarity<br>
                4) hyperlink count ($e_j$ mentioned in $e_i$)<br>
        * use Gradient Boosted Regression Tress for training GBRTs; get SOTA as well as improving LRL XEL by quite a bit (~15% for most languages)
            
    * **RELIC**- Learning Cross-Context Entity Representations From Text: [[paper]](https://arxiv.org/pdf/2001.03765v1.pdf) *
        * this is just for English but I feel like it should extend with the recent advancements in cross lingual embeddings
        * basic idea: use BERT to embed the mention using the context and embed the abstract of all entities (64 tokens for both); then take cosine similarity
        * worth noting the dataset is Wikidata and they test it on CoNLL-Aida and TAC-KBP 2010; they also acheive close to SOTA on entity typing on the FIGMENT dataset
    * **Improving Candidate Generation for Low-resource Cross-lingual Entity Linking**: [[paper]](https://arxiv.org/ftp/arxiv/papers/2003/2003.01343.pdf), [[code]](https://github.com/shuyanzhou/pbel_plus)
        * background: for monolingual linking, either string similariy or using entity-mention lookup tables provide high recalls, in 90%; for LRL they use LRL-English gazeteers or a neural model to measure similarity between a closely related HRL and English
        * contributions: <br>
            * 1) problem: neural models often trained on entity-entity<br>
                1) solution: train models $mention_{LRL}->entity$ <br>
            * 2) problem: model not good enough (b/c short text)
                2) solution: replace LSTM w/ character n-gram model
        * average gain of 16.9% on gold candidate recall (formula nice in the paper) and 7.9% on XEL
        
    * **Zero-shot Entity Linking with Dense Entity Retrieval**: [[paper]](https://arxiv.org/pdf/1911.03814.pdf), [[code]](https://github.com/facebookresearch/BLINK), [[slides]](https://speakerdeck.com/izuna385/zero-shot-entity-linking-with-dense-entity-retrieval-unofficial-slides-and-entity-linking-future-directions?slide=71)
    * **Entity Linking with a Knowledge Base: Issues, Techniques, and Solutions**: [[paper]](http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/TKDE14-entitylinking.pdf)
    
    * **Zero-Shot Entity Linking by Reading Entity Descriptions**: [[paper]](https://www.aclweb.org/anthology/P19-1335.pdf)
    
## Data:
* **DBpedia**: [2016](https://wiki.dbpedia.org/downloads-2016-10), [2018](https://wiki.dbpedia.org/Datasets)
* **VoxEL**: [[paper]](http://aidanhogan.com/docs/voxel_multilingual_entity_linking.pdf), [[data]](https://figshare.com/articles/VoxEL/6539675) very small and probably out fo date
* **Charagram transliteration WikiData**: [[pbel repo]](https://github.com/shuyanzhou/pbel_plus) $m_{LRL}$ -> $e_{EN}$ and collection of aliases and phonetic conversions; good for potential candidate generation
* **TAC KBP**: [[paper]](http://www.lrec-conf.org/proceedings/lrec2012/pdf/278_Paper.pdf)
* **AIDA-YAGO2**: English CONLL 2003 entity dataset w/ YAGO2, Freebase, and Wikipedia linking and type annotations [[data]](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads)

* **Repositories**: [[TAGME]](https://github.com/hasibi/TAGME-Reproducibility), [[NLP-progress]](http://nlpprogress.com/english/entity_linking.html)
<br>

2. **Coreference Resolution**
    * **Neural Cross-Lingual Coreference Resolution And Its Application To Entity Linking**: [[paper]](https://arxiv.org/pdf/1806.10201.pdf)