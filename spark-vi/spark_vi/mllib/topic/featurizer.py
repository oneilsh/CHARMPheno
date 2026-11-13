"""DagGatedFeaturizer: the prep stage that builds a topic model's inputs (spec R12).

The role ``Tokenizer`` + ``CountVectorizer`` play before MLlib's ``LDA``, for
documents whose terms come in several named DOMAINS and that carry LABELS (node ids
of a label DAG). It turns the tables a user has into the columns the DAG-gated
estimator reads:

- per domain, one sparse count vector column (``featuresCols()``), its width the
  size of the vocabulary learned for that domain (``vocabularies()``);
- one array column of node ids per document (``labelsCol()``, spec R2).

Inputs, per domain, are either

(a) an ARRAY column of terms on the documents DataFrame (``array<string>`` or
    ``array<int>``; each element is one occurrence), registered with
    :meth:`DagGatedFeaturizer.addArrayDomain`; or
(b) a LONG TABLE ``(doc_id, term[, count])``, registered with
    :meth:`DagGatedFeaturizer.addTableDomain`. Without a count column each row is one
    occurrence; with one, the count is the occurrence multiplicity (a non-negative
    integer; anything else fails the job by name rather than being truncated).

Labels come from an array column on the documents (``setLabelsInputCol``) or a long
table ``(doc_id, node)`` (``setLabelsTable``), collected per document. Node ids are
kept exactly as given (their Spark type is preserved: ``1`` and ``"1"`` are never
conflated by a cast). A document with no labels gets an EMPTY array (a background
document); a document absent from a domain gets an all-zero vector of that domain's
width.

The documents DataFrame passed to ``fit`` / ``transform`` defines the documents (one
row each, ``docIdCol`` unique); long tables are LEFT-joined onto it by ``docIdCol``,
so table rows for documents not in it are ignored.

Vocabulary learning is ``pyspark.ml.feature.CountVectorizer`` on each domain's
per-document term array (``minDF``, ``maxDF``, ``vocabSize``, ``minTF`` with its
semantics, applied to every domain alike). Term ids are strings in the vocabulary
(``CountVectorizer`` stores strings; ``array<int>`` terms are cast), so a domain's
vocabulary is a ``list[str]`` ordered by descending corpus frequency.

Long tables are DataFrames, not Params. They are held on the stage instance (and
carried by ``copy()`` and into the fitted model), used by ``fit`` and by the fitted
model's ``transform``. To transform OTHER documents whose terms live in other long
tables, replace them on the model: ``model.setDomainTable(name, df)`` /
``model.setLabelsTable(df)`` (same column names as registered). Tables are not
persisted: after ``load``, re-register each table-backed domain's table before
``transform`` (the error names the domain otherwise).

Example::

    feat = (DagGatedFeaturizer(docIdCol="doc_id", minDF=2, vocabSize=5000)
            .addArrayDomain("words", inputCol="tokens")
            .addTableDomain("events", events_long, termCol="event", countCol="n")
            .setLabelsTable(labels_long, nodeCol="node"))
    prep = feat.fit(docs)
    data = prep.transform(docs)
    est = (DagGatedLDAEstimator()        # or DagGatedLDAEstimator.recommended()
           .setDag(dag)                   # {child: parent | [parents]} in label ids
           .setFeaturesCols(prep.featuresCols())
           .setVocabularies(prep.vocabularies())
           .setLabelsCol(prep.labelsCol()))
    model = est.fit(data)

Persistence: both stages are ``MLWritable`` / ``MLReadable`` (``Pipeline.save`` /
``PipelineModel.load`` work). The model writes the standard Params metadata plus its
vocabularies as ``<path>/data/vocabularies.json``, both written from the driver through
``spark_vi.io.fs``, so a local path or any Hadoop-filesystem URI works.

Domain-neutral: nothing here knows what a document, term or node means.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from pyspark import keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import (
    DefaultParamsReader,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
)

from spark_vi.mllib._common import (DriverParamsWritable, read_params_metadata,
                                    refuse_existing_path, write_params_metadata)

_TERMS_TMP = "__dgf_terms_{}"
_DOC_TMP = "__dgf_doc"


class _DagGatedFeaturizerParams(Params):
    """Params shared by the featurizer and its fitted model.

    The per-domain layout (``domainSpecs``) and the labels source (``labelsSpec``)
    are JSON strings so they persist with the standard Params metadata; set them
    through the ``add*Domain`` / ``setLabels*`` methods, not directly.
    """

    docIdCol = Param(
        Params._dummy(), "docIdCol",
        "document id column on the documents DataFrame (and the default document id "
        "column of every long table); needed only when a long table is registered. "
        "Default 'doc_id'.",
        typeConverter=TypeConverters.toString)
    domainSpecs = Param(
        Params._dummy(), "domainSpecs",
        "JSON list of the ordered domains: name, kind ('array' or 'table'), input "
        "column(s), output vector column. Set via addArrayDomain / addTableDomain. "
        "Default '[]' (no domains).",
        typeConverter=TypeConverters.toString)
    labelsSpec = Param(
        Params._dummy(), "labelsSpec",
        "JSON description of the labels source (array column or long table), or "
        "'null' for none. Set via setLabelsInputCol / setLabelsTable. Default "
        "'null'.",
        typeConverter=TypeConverters.toString)
    labelsOutputCol = Param(
        Params._dummy(), "labelsOutputCol",
        "output column: array of node ids per document (empty = background). "
        "Default 'labels' (the estimator's default labelsCol).",
        typeConverter=TypeConverters.toString)
    minDF = Param(
        Params._dummy(), "minDF",
        "per domain: minimum number (>= 1) or fraction (< 1) of documents a term must "
        "appear in to enter the vocabulary (CountVectorizer semantics). Default 1.",
        typeConverter=TypeConverters.toFloat)
    maxDF = Param(
        Params._dummy(), "maxDF",
        "per domain: maximum number (>= 1) or fraction (< 1) of documents a term may "
        "appear in (CountVectorizer semantics). Default 2^63 - 1 (no limit).",
        typeConverter=TypeConverters.toFloat)
    minTF = Param(
        Params._dummy(), "minTF",
        "per document: a term counted fewer than this many times (>= 1) or below this "
        "fraction of the document's domain length (< 1) is dropped from that "
        "document's vector (CountVectorizer semantics; applied at transform). "
        "Default 1.",
        typeConverter=TypeConverters.toFloat)
    vocabSize = Param(
        Params._dummy(), "vocabSize",
        "per domain: maximum vocabulary size, the most frequent terms kept. "
        "Default 2^18.",
        typeConverter=TypeConverters.toInt)

    def __init__(self, *args):
        super().__init__(*args)
        self._setDefault(
            docIdCol="doc_id", domainSpecs="[]", labelsSpec="null",
            labelsOutputCol="labels", minDF=1.0, maxDF=float(2 ** 63 - 1),
            minTF=1.0, vocabSize=1 << 18)
        self._tables: Dict[str, DataFrame] = {}
        self._labels_table: Optional[DataFrame] = None

    # -- read access ----------------------------------------------------------
    def _specs(self) -> List[dict]:
        return json.loads(self.getOrDefault(self.domainSpecs))

    def _labels(self) -> Optional[dict]:
        return json.loads(self.getOrDefault(self.labelsSpec))

    def domains(self) -> List[str]:
        """The domain names, in order."""
        return [s["name"] for s in self._specs()]

    def featuresCols(self) -> List[str]:
        """The output vector column of each domain, aligned with ``domains()``."""
        return [s["outputCol"] for s in self._specs()]

    def labelsCol(self) -> str:
        """The output labels column (array of node ids per document)."""
        return self.getOrDefault(self.labelsOutputCol)

    def getDocIdCol(self) -> str:
        return self.getOrDefault(self.docIdCol)

    def getMinDF(self) -> float:
        return self.getOrDefault(self.minDF)

    def getMaxDF(self) -> float:
        return self.getOrDefault(self.maxDF)

    def getMinTF(self) -> float:
        return self.getOrDefault(self.minTF)

    def getVocabSize(self) -> int:
        return self.getOrDefault(self.vocabSize)

    def setDocIdCol(self, value: str):
        return self._set(docIdCol=value)

    def setLabelsOutputCol(self, value: str):
        return self._set(labelsOutputCol=value)

    # -- tables (not Params) --------------------------------------------------
    def setDomainTable(self, name: str, table: DataFrame):
        """Replace the long table of an already-registered table domain.

        Same column names as registered. Used to point a fitted model at the long
        table of new documents, or to re-attach a table after ``load``.
        """
        spec = {s["name"]: s for s in self._specs()}.get(name)
        if spec is None or spec["kind"] != "table":
            raise ValueError(f"no table domain named {name!r}; table domains: "
                             f"{[s['name'] for s in self._specs() if s['kind'] == 'table']}")
        self._tables[name] = table
        return self

    def setLabelsTable(self, table: DataFrame, nodeCol: Optional[str] = None,
                       docIdCol: Optional[str] = None):
        """Labels from a long table ``(doc_id, node)``, one row per (document, node).

        On a fitted model (or with the labels table already registered) the column
        names may be omitted to keep the registered ones.
        """
        prev = self._labels()
        if prev is not None and prev["kind"] == "table":
            nodeCol = nodeCol if nodeCol is not None else prev["nodeCol"]
            docIdCol = docIdCol if docIdCol is not None else prev["docIdCol"]
        self._set(labelsSpec=json.dumps({
            "kind": "table", "nodeCol": nodeCol or "node", "docIdCol": docIdCol}))
        self._labels_table = table
        return self

    def setLabelsInputCol(self, inputCol: str):
        """Labels from an array column of node ids on the documents DataFrame."""
        self._set(labelsSpec=json.dumps({"kind": "array", "inputCol": inputCol}))
        self._labels_table = None
        return self

    def copy(self, extra=None):
        that = super().copy(extra)
        # Params.copy is a shallow copy; give the copy its own table registry so
        # setDomainTable on one does not repoint the other.
        that._tables = dict(self._tables)
        that._labels_table = self._labels_table
        return that

    # -- assembly shared by fit and transform ---------------------------------
    def _assemble(self, docs: DataFrame) -> DataFrame:
        """Per domain, a temporary ``array<string>`` term column; plus the labels.

        Long tables are reduced to one term array per document and left-joined on
        ``docIdCol``; a document a domain never mentions gets an empty array, which
        CountVectorizer turns into an all-zero vector of the domain's width.
        """
        doc_id = self.getDocIdCol()
        out = docs
        for i, spec in enumerate(self._specs()):
            tmp = _TERMS_TMP.format(i)
            if spec["kind"] == "array":
                col = F.col(spec["inputCol"]).cast(ArrayType(StringType()))
                out = out.withColumn(tmp, F.filter(col, lambda x: x.isNotNull()))
            else:
                table = self._tables.get(spec["name"])
                if table is None:
                    raise ValueError(
                        f"domain {spec['name']!r} reads a long table, but none is "
                        f"registered on this stage (tables are not persisted); call "
                        f"setDomainTable({spec['name']!r}, df) first")
                if doc_id not in docs.columns:
                    raise ValueError(f"documents DataFrame has no {doc_id!r} column "
                                     f"(docIdCol), needed to join long tables")
                out = out.join(_table_to_arrays(table, spec, doc_id, tmp),
                               on=doc_id, how="left")
            out = out.withColumn(
                tmp, F.coalesce(F.col(tmp), F.array().cast(ArrayType(StringType()))))

        labels = self._labels()
        lab_out = self.labelsCol()
        if labels is None:
            return out
        if labels["kind"] == "array":
            col = F.col(labels["inputCol"])
            elem = docs.schema[labels["inputCol"]].dataType
            if not isinstance(elem, ArrayType):
                raise TypeError(f"labels column {labels['inputCol']!r} must be an "
                                f"array column, got {elem.simpleString()}")
            filtered = F.filter(col, lambda x: x.isNotNull())
            return out.withColumn(
                lab_out, F.coalesce(filtered, F.array().cast(elem)))
        table = self._labels_table
        if table is None:
            raise ValueError("labels read a long table, but none is registered on "
                             "this stage (tables are not persisted); call "
                             "setLabelsTable(df) first")
        if doc_id not in docs.columns:
            raise ValueError(f"documents DataFrame has no {doc_id!r} column "
                             f"(docIdCol), needed to join the labels table")
        tdoc = labels["docIdCol"] or doc_id
        node_type = table.schema[labels["nodeCol"]].dataType
        agg = (table
               .where(F.col(labels["nodeCol"]).isNotNull())
               .groupBy(F.col(tdoc).alias(doc_id))
               .agg(F.collect_list(labels["nodeCol"]).alias(_DOC_TMP)))
        if lab_out in out.columns:
            out = out.drop(lab_out)
        out = out.join(agg, on=doc_id, how="left")
        return (out.withColumn(lab_out, F.coalesce(
                    F.col(_DOC_TMP), F.array().cast(ArrayType(node_type))))
                .drop(_DOC_TMP))


def _table_to_arrays(table: DataFrame, spec: dict, doc_id: str, tmp: str) -> DataFrame:
    """Long table ``(doc, term[, count])`` -> ``(doc_id, tmp: array<string>)``.

    With a count, each term is repeated ``count`` times (``array_repeat``), so the
    array is exactly what an array-column input of the same document would be and
    CountVectorizer's minDF / minTF / frequency ordering apply unchanged. A count that
    is null, negative or non-integral fails the job naming the domain: truncating it
    would silently change the corpus.
    """
    tdoc = spec.get("docIdCol") or doc_id
    term = F.col(spec["termCol"]).cast(StringType())
    if spec.get("countCol"):
        c = F.col(spec["countCol"])
        n = F.when(c.isNull() | (c < 0) | (c != F.floor(c)), F.raise_error(F.lit(
            f"domain {spec['name']!r}: count column {spec['countCol']!r} must hold "
            f"non-negative integers"))).otherwise(c.cast(IntegerType()))
        piece = F.array_repeat(term, n)
    else:
        piece = F.array(term)
    return (table
            .where(F.col(spec["termCol"]).isNotNull())
            .select(F.col(tdoc).alias(doc_id), piece.alias(tmp))
            .groupBy(doc_id)
            .agg(F.flatten(F.collect_list(tmp)).alias(tmp)))


class DagGatedFeaturizer(_DagGatedFeaturizerParams, Estimator, DriverParamsWritable):
    """Learn per-domain vocabularies and build vector + label columns (spec R12).

    See the module docstring for the input kinds and the example. ``fit(docs)``
    learns one vocabulary per domain with ``CountVectorizer`` semantics and returns a
    :class:`DagGatedFeaturizerModel`. Persisting the estimator saves its Params only;
    long tables are re-registered after ``load`` (``setDomainTable`` /
    ``setLabelsTable``).
    """

    @keyword_only
    def __init__(self, *, docIdCol="doc_id", labelsOutputCol="labels", minDF=1.0,
                 maxDF=float(2 ** 63 - 1), minTF=1.0, vocabSize=1 << 18):
        super().__init__()
        self._set(**self._input_kwargs)

    def setMinDF(self, value: float):
        return self._set(minDF=value)

    def setMaxDF(self, value: float):
        return self._set(maxDF=value)

    def setMinTF(self, value: float):
        return self._set(minTF=value)

    def setVocabSize(self, value: int):
        return self._set(vocabSize=value)

    def _add_domain(self, spec: dict):
        specs = self._specs()
        name = spec["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("a domain name must be a non-empty string")
        if name in [s["name"] for s in specs]:
            raise ValueError(f"domain {name!r} is already registered")
        spec["outputCol"] = spec.get("outputCol") or f"{name}_features"
        if spec["outputCol"] in [s["outputCol"] for s in specs]:
            raise ValueError(f"output column {spec['outputCol']!r} is already used")
        self._set(domainSpecs=json.dumps(specs + [spec]))
        return self

    def addArrayDomain(self, name: str, inputCol: str, outputCol: Optional[str] = None):
        """Add a domain read from an array column of terms on the documents.

        The output vector column defaults to ``<name>_features``.
        """
        return self._add_domain({"name": name, "kind": "array", "inputCol": inputCol,
                                 "outputCol": outputCol})

    def addTableDomain(self, name: str, table: DataFrame, termCol: str = "term",
                       countCol: Optional[str] = None, docIdCol: Optional[str] = None,
                       outputCol: Optional[str] = None):
        """Add a domain read from a long table ``(doc_id, term[, count])``.

        ``docIdCol`` defaults to the featurizer's ``docIdCol``; without ``countCol``
        each row is one occurrence. The output column defaults to
        ``<name>_features``.
        """
        self._add_domain({"name": name, "kind": "table", "termCol": termCol,
                          "countCol": countCol, "docIdCol": docIdCol,
                          "outputCol": outputCol})
        self._tables[name] = table
        return self

    def _fit(self, dataset: DataFrame) -> "DagGatedFeaturizerModel":
        specs = self._specs()
        if not specs:
            raise ValueError("no domains registered; call addArrayDomain / "
                             "addTableDomain first")
        assembled = self._assemble(dataset).persist()
        try:
            vocabs = []
            for i, spec in enumerate(specs):
                cv = CountVectorizer(
                    inputCol=_TERMS_TMP.format(i), outputCol=spec["outputCol"],
                    minDF=self.getMinDF(), maxDF=self.getMaxDF(),
                    minTF=self.getMinTF(), vocabSize=self.getVocabSize())
                vocab = list(cv.fit(assembled).vocabulary)
                if not vocab:  # CountVectorizer only warns; a 0-wide domain is useless
                    raise ValueError(
                        f"domain {spec['name']!r} has an empty vocabulary "
                        f"(minDF={self.getMinDF()}, maxDF={self.getMaxDF()}); "
                        f"relax them or drop the domain")
                vocabs.append(vocab)
        finally:
            assembled.unpersist()
        model = DagGatedFeaturizerModel(vocabs)
        self._copyValues(model)
        model._tables = dict(self._tables)
        model._labels_table = self._labels_table
        return model


class DagGatedFeaturizerModel(_DagGatedFeaturizerParams, Model, MLReadable, MLWritable):
    """Fitted prep stage: per-domain vocabularies; ``transform`` emits the columns.

    ``transform(docs)`` adds one ``SparseVector`` column per domain (``featuresCols()``,
    size ``len(vocabularies()[m])``; out-of-vocabulary terms dropped) and, if a labels
    source is registered, the ``labelsCol()`` array column. The temporary columns are
    dropped; the documents' own columns pass through.
    """

    def __init__(self, vocabularies: Optional[List[List[str]]] = None):
        super().__init__()
        self._vocabularies = [list(v) for v in (vocabularies or [])]

    def vocabularies(self) -> List[List[str]]:
        """Each domain's vocabulary (term ids, vector index order), aligned with
        ``featuresCols()``."""
        return [list(v) for v in self._vocabularies]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        specs = self._specs()
        if len(specs) != len(self._vocabularies):
            raise ValueError(f"{len(specs)} domain(s) but {len(self._vocabularies)} "
                             f"vocabularies")
        out = self._assemble(dataset)
        for i, (spec, vocab) in enumerate(zip(specs, self._vocabularies)):
            tmp = _TERMS_TMP.format(i)
            cvm = CountVectorizerModel.from_vocabulary(
                vocab, inputCol=tmp, outputCol=spec["outputCol"],
                minTF=self.getMinTF())
            out = cvm.transform(out).drop(tmp)
        return out

    def write(self) -> MLWriter:
        return _FeaturizerModelWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return _FeaturizerModelReader(cls)


class _FeaturizerModelWriter(MLWriter):
    """Params metadata (``<path>/metadata``) + vocabularies JSON (``<path>/data/
    vocabularies.json``), both written from the driver through ``spark_vi.io.fs``
    (any local path or Hadoop URI; a Spark write job would put a driver-local path on
    the executors' disks). Metadata last. Refuses an existing path unless
    ``write().overwrite()`` (MLlib's rule)."""

    def __init__(self, instance: DagGatedFeaturizerModel):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        from spark_vi.io import fs
        uri = fs.qualify(path)
        refuse_existing_path(path, uri)
        fs.write_text(fs.join(uri, "data", "vocabularies.json"),
                      json.dumps([list(v) for v in self.instance._vocabularies]))
        write_params_metadata(self.instance, uri, self.sc)


class _FeaturizerModelReader(MLReader):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def load(self, path: str) -> DagGatedFeaturizerModel:
        from spark_vi.io import fs
        uri = fs.qualify(path)
        metadata = read_params_metadata(uri)
        vocabs: List[List[str]] = json.loads(
            fs.read_text(fs.join(uri, "data", "vocabularies.json")))
        model = self.cls(vocabs)
        model._resetUid(metadata["uid"])
        DefaultParamsReader.getAndSetParams(model, metadata)
        return model
