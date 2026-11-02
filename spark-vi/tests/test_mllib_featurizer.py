"""DagGatedFeaturizer (spec R12): per-domain vocabularies, vectors and labels."""
import pytest
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import SparseVector
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from spark_vi.mllib import DagGatedFeaturizer, DagGatedFeaturizerModel
from spark_vi.mllib.topic import OnlineLDAEstimator


def _docs(spark):
    return spark.createDataFrame(
        [(1, ["a", "b", "a"], [10, 11]),
         (2, ["b", "c"], [11]),
         (3, [], None),
         (4, None, [12, 10, 10])],
        "doc_id long, words array<string>, codes array<int>")


def _by_doc(df, *cols):
    return {r["doc_id"]: tuple(r[c] for c in cols) for r in df.collect()}


def _dense(v: SparseVector, vocab, ):
    return {vocab[i]: c for i, c in zip(v.indices, v.values)}


def test_array_domains_string_and_int(spark):
    docs = _docs(spark)
    m = (DagGatedFeaturizer().addArrayDomain("words", "words")
         .addArrayDomain("codes", "codes", outputCol="codes_vec")).fit(docs)
    assert m.domains() == ["words", "codes"]
    assert m.featuresCols() == ["words_features", "codes_vec"]
    vw, vc = m.vocabularies()
    assert sorted(vw) == ["a", "b", "c"] and vw[0] in ("a", "b")
    assert sorted(vc) == ["10", "11", "12"] and vc[0] == "10"  # int terms -> str ids
    out = _by_doc(m.transform(docs), "words_features", "codes_vec")
    assert _dense(out[1][0], vw) == {"a": 2.0, "b": 1.0}
    assert _dense(out[4][1], vc) == {"12": 1.0, "10": 2.0}
    # empty and null documents: all-zero vectors of the domain width
    for d, i in ((3, 0), (4, 0), (3, 1)):
        assert out[d][i].size == len(m.vocabularies()[i]) and out[d][i].numNonzeros() == 0
    assert "__dgf_terms_0" not in m.transform(docs).columns


def test_table_domains_with_and_without_counts(spark):
    docs = spark.createDataFrame([(1,), (2,), (3,)], "doc_id long")
    rows = spark.createDataFrame(
        [(1, "x"), (1, "x"), (1, "y"), (2, "y"), (9, "z")], "doc_id long, term string")
    counted = spark.createDataFrame(
        [(1, "p", 3), (2, "p", 1), (2, "q", 2), (2, None, 5)],
        "pid long, code string, n long")
    m = (DagGatedFeaturizer()
         .addTableDomain("rows", rows)
         .addTableDomain("counted", counted, termCol="code", countCol="n",
                         docIdCol="pid")).fit(docs)
    vr, vc = m.vocabularies()
    assert sorted(vr) == ["x", "y"]  # doc 9 is not a document: its "z" is ignored
    assert vc == ["p", "q"]  # p: 4 occurrences, q: 2
    out = _by_doc(m.transform(docs), "rows_features", "counted_features")
    assert _dense(out[1][0], vr) == {"x": 2.0, "y": 1.0}
    assert _dense(out[1][1], vc) == {"p": 3.0}
    assert _dense(out[2][1], vc) == {"p": 1.0, "q": 2.0}
    assert out[3][0].size == 2 and out[3][0].numNonzeros() == 0
    assert out[3][1].size == 2 and out[3][1].numNonzeros() == 0


def test_non_integer_count_fails_by_name(spark):
    docs = spark.createDataFrame([(1,)], "doc_id long")
    bad = spark.createDataFrame([(1, "p", 1.5)], "doc_id long, term string, n double")
    f = DagGatedFeaturizer().addTableDomain("events", bad, countCol="n")
    with pytest.raises(Exception, match="non-negative integers"):
        f.fit(docs)


def test_mixed_array_and_table_domains(spark):
    docs = _docs(spark).select("doc_id", "words")
    table = spark.createDataFrame([(2, "t1"), (4, "t2"), (4, "t1")],
                                  "doc_id long, term string")
    m = (DagGatedFeaturizer().addArrayDomain("words", "words")
         .addTableDomain("events", table)).fit(docs)
    out = m.transform(docs)
    assert out.count() == 4
    got = _by_doc(out, "words_features", "events_features")
    ve = m.vocabularies()[1]
    assert _dense(got[4][1], ve) == {"t1": 1.0, "t2": 1.0}
    assert got[1][1].numNonzeros() == 0 and got[1][1].size == len(ve)
    assert set(out.columns) == {"doc_id", "words", "words_features", "events_features"}


def test_min_df_max_df_and_vocab_size(spark):
    docs = spark.createDataFrame(
        [(i, w) for i, w in enumerate([["a", "b", "c"], ["a", "b"], ["a", "d", "d", "d", "d"]])],
        "doc_id long, words array<string>")
    m = DagGatedFeaturizer(minDF=2.0).addArrayDomain("w", "words").fit(docs)
    assert sorted(m.vocabularies()[0]) == ["a", "b"]
    m = DagGatedFeaturizer(vocabSize=1).addArrayDomain("w", "words").fit(docs)
    assert m.vocabularies()[0] == ["d"]  # most frequent overall (d: 4, a: 3)
    m = DagGatedFeaturizer(maxDF=2.0).addArrayDomain("w", "words").fit(docs)
    assert "a" not in m.vocabularies()[0]
    with pytest.raises(ValueError, match="'w' has an empty vocabulary"):
        DagGatedFeaturizer(minDF=5.0).addArrayDomain("w", "words").fit(docs)
    out = m.transform(docs).collect()
    assert all(r["w_features"].size == len(m.vocabularies()[0]) for r in out)


def test_labels_from_array_column_keep_type_and_background(spark):
    docs = spark.createDataFrame(
        [(1, ["a"], [1, 2]), (2, ["b"], []), (3, ["a"], None)],
        "doc_id long, words array<string>, nodes array<int>")
    m = (DagGatedFeaturizer().addArrayDomain("w", "words")
         .setLabelsInputCol("nodes")).fit(docs)
    assert m.labelsCol() == "labels"
    out = m.transform(docs)
    assert out.schema["labels"].dataType == ArrayType(IntegerType())
    got = _by_doc(out, "labels")
    assert got == {1: ([1, 2],), 2: ([],), 3: ([],)}


def test_labels_from_long_table_str_and_int_ids(spark):
    docs = spark.createDataFrame([(1, ["a"]), (2, ["b"]), (3, ["a"])],
                                 "doc_id long, words array<string>")
    str_labels = spark.createDataFrame(
        [(1, "1"), (1, "N:7"), (2, "1"), (2, None)], "doc_id long, node string")
    m = (DagGatedFeaturizer().addArrayDomain("w", "words")
         .setLabelsTable(str_labels).setLabelsOutputCol("nodes")).fit(docs)
    out = m.transform(docs)
    assert m.labelsCol() == "nodes"
    assert out.schema["nodes"].dataType == ArrayType(StringType())
    got = {k: sorted(v[0]) for k, v in _by_doc(out, "nodes").items()}
    assert got == {1: ["1", "N:7"], 2: ["1"], 3: []}

    int_labels = spark.createDataFrame([(5, 1), (1, 2), (1, 2)], "pid long, n long")
    m = (DagGatedFeaturizer().addArrayDomain("w", "words")
         .setLabelsTable(int_labels, nodeCol="n", docIdCol="pid")).fit(docs)
    out = m.transform(docs)
    assert out.schema["labels"].dataType == ArrayType(LongType())  # not cast to str
    got = _by_doc(out, "labels")
    assert got == {1: ([2, 2],), 2: ([],), 3: ([],)}  # duplicates kept (R2)


def test_transform_new_docs_unseen_terms_dropped(spark):
    train = spark.createDataFrame([(1, ["a", "b"]), (2, ["b"])],
                                  "doc_id long, words array<string>")
    tab = spark.createDataFrame([(1, "x"), (2, "y")], "doc_id long, term string")
    m = (DagGatedFeaturizer().addArrayDomain("w", "words")
         .addTableDomain("t", tab)).fit(train)
    sizes = [len(v) for v in m.vocabularies()]
    new = spark.createDataFrame([(7, ["a", "zzz", "zzz"])],
                                "doc_id long, words array<string>")
    m.setDomainTable("t", spark.createDataFrame([(7, "y"), (7, "unseen")],
                                                "doc_id long, term string"))
    r = m.transform(new).collect()[0]
    assert r["w_features"].size == sizes[0] and r["t_features"].size == sizes[1]
    assert _dense(r["w_features"], m.vocabularies()[0]) == {"a": 1.0}
    assert _dense(r["t_features"], m.vocabularies()[1]) == {"y": 1.0}


def test_copy_has_its_own_tables_and_registration_errors(spark):
    tab = spark.createDataFrame([(1, "x")], "doc_id long, term string")
    f = DagGatedFeaturizer().addTableDomain("t", tab)
    g = f.copy()
    g.setDomainTable("t", spark.createDataFrame([(1, "y")], "doc_id long, term string"))
    assert f._tables["t"] is tab
    with pytest.raises(ValueError, match="already registered"):
        f.addArrayDomain("t", "words")
    with pytest.raises(ValueError, match="no table domain"):
        f.setDomainTable("nope", tab)
    with pytest.raises(ValueError, match="no domains"):
        DagGatedFeaturizer().fit(spark.createDataFrame([(1,)], "doc_id long"))


def test_pipeline_feeds_a_topic_estimator(spark):
    docs = spark.createDataFrame(
        [(i, ["a", "b", "c"][: 1 + i % 3]) for i in range(12)],
        "doc_id long, words array<string>")
    feat = DagGatedFeaturizer().addArrayDomain("w", "words")
    lda = OnlineLDAEstimator(k=2, maxIter=2, seed=1, featuresCol="w_features")
    pm = Pipeline(stages=[feat, lda]).fit(docs)
    out = pm.transform(docs)
    assert "topicDistribution" in out.columns and out.count() == 12
    assert isinstance(pm.stages[0], DagGatedFeaturizerModel)


def test_save_load_round_trip(spark, tmp_path):
    docs = _docs(spark)
    lab = spark.createDataFrame([(1, 3), (2, 4)], "doc_id long, node int")
    tab = spark.createDataFrame([(1, "x", 2)], "doc_id long, term string, n int")
    m = (DagGatedFeaturizer(minTF=1.0, docIdCol="doc_id")
         .addArrayDomain("words", "words")
         .addTableDomain("t", tab, countCol="n")
         .setLabelsTable(lab)).fit(docs)
    path = f"file://{tmp_path}/feat"
    m.write().overwrite().save(path)
    m2 = DagGatedFeaturizerModel.load(path)
    assert m2.vocabularies() == m.vocabularies()
    assert m2.domains() == m.domains() and m2.featuresCols() == m.featuresCols()
    assert m2.labelsCol() == m.labelsCol() and m2.uid == m.uid
    with pytest.raises(ValueError, match="'t' reads a long table"):
        m2.transform(docs)
    m2.setDomainTable("t", tab).setLabelsTable(lab)
    a = _by_doc(m.transform(docs), "words_features", "t_features", "labels")
    b = _by_doc(m2.transform(docs), "words_features", "t_features", "labels")
    assert a == b

    # the unfitted estimator persists its Params (tables re-registered after load)
    est = DagGatedFeaturizer(vocabSize=7).addArrayDomain("words", "words")
    est.write().overwrite().save(f"file://{tmp_path}/est")
    est2 = DagGatedFeaturizer.load(f"file://{tmp_path}/est")
    assert est2.getVocabSize() == 7 and est2.domains() == ["words"]

    # inside a PipelineModel
    pm = Pipeline(stages=[DagGatedFeaturizer().addArrayDomain("words", "words")]).fit(docs)
    pm.write().overwrite().save(f"file://{tmp_path}/pipe")
    pm2 = PipelineModel.load(f"file://{tmp_path}/pipe")
    assert pm2.stages[0].vocabularies() == pm.stages[0].vocabularies()
