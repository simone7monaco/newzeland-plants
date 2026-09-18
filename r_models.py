
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import warnings

import numpy as np
import pandas as pd


class RBackendUnavailable(RuntimeError):
    pass


def _coerce_categorical_columns(
    data: pd.DataFrame,
    categorical_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a copy whose categorical-like columns convert to R factors.

    ``pandas2ri`` maps pandas ``category`` columns to R factors.  Explicitly
    performing the conversion avoids version-dependent treatment of pandas
    object/string columns by rpy2 and by the downstream R package.
    """
    converted = data.copy(deep=True)
    if categorical_columns is None:
        columns = converted.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()
    else:
        columns = [str(column) for column in categorical_columns]
        missing = sorted(set(columns).difference(converted.columns))
        if missing:
            raise ValueError(f"Unknown categorical columns: {missing}")

    for column in columns:
        converted[column] = converted[column].astype("category")
    return converted


@dataclass
class MICE:
    """Exact Python front-end to R's mice(..., method='cart').
    Allows other methods, but raises an error if the method is not 'cart'.

    This class intentionally delegates the imputation itself to the R packages
    ``mice`` and ``rpart``. That is what makes it behaviorally identical to the
    R implementation, including rpart's tree construction, factor handling,
    donor/class sampling, and R's RNG.

    Parameters mirror the most commonly used arguments of ``mice::mice``.
    Extra keyword arguments can be passed with ``mice_kwargs`` and are forwarded
    unchanged to ``mice::mice`` (therefore also to ``mice.impute.cart`` where
    applicable, e.g. ``minbucket`` and ``cp``).
    """

    m: int = 5
    maxit: int = 5
    seed: Optional[int] = None
    print_flag: bool = False
    minbucket: int = 5
    cp: float = 1e-4
    visit_sequence: Optional[Any] = None
    predictor_matrix: Optional[Any] = None
    where: Optional[Any] = None
    method: str = "cart"
    ignore: Optional[Iterable[bool]] = None
    mice_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.method != "cart":
            raise ValueError("method must be 'cart'")
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.maxit < 0:
            raise ValueError("maxit must be >= 0")
        if self.minbucket < 1:
            self.minbucket = 1  # same normalization as mice.impute.cart
        if self.cp < 0:
            raise ValueError("cp must be >= 0")
        self._mids = None
        self._columns: list[str] | None = None
        self._index: pd.Index | None = None
        self._r = None
        self._mice = None

    @staticmethod
    def _load_r():
        try:
            from rpy2 import robjects
            from rpy2.robjects import default_converter, pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr
        except Exception as exc:
            raise RBackendUnavailable(
                "rpy2 is not available. Install R, then `pip install rpy2`, "
                "and install the R packages `mice` and `rpart`."
            ) from exc

        try:
            mice = importr("mice")
            importr("rpart")
        except Exception as exc:
            raise RBackendUnavailable(
                "R is reachable through rpy2, but R packages `mice` and/or "
                "`rpart` are missing. In R run: install.packages(c('mice','rpart'))."
            ) from exc

        return robjects, default_converter, pandas2ri, localconverter, mice

    @staticmethod
    def _validate_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        if data.columns.has_duplicates:
            raise ValueError("column names must be unique")
        if len(data.columns) == 0:
            raise ValueError("data must contain at least one column")
        if data.index.has_duplicates:
            raise ValueError("row index must uniquely identify species")
        return _coerce_categorical_columns(data)

    @staticmethod
    def _to_r_matrix(robjects, obj: Any, *, logical: bool = False):
        arr = np.asarray(obj)
        if arr.ndim != 2:
            raise ValueError("expected a 2-D matrix")
        flat = arr.ravel(order="F")
        if logical:
            vec = robjects.BoolVector([bool(x) for x in flat])
        else:
            vec = robjects.IntVector([int(x) for x in flat])
        return robjects.r["matrix"](vec, nrow=arr.shape[0], ncol=arr.shape[1])

    def fit(self, data: pd.DataFrame) -> "MICE":
        data = self._validate_dataframe(data)
        (
            robjects,
            default_converter,
            pandas2ri,
            localconverter,
            mice,
        ) = self._load_r()

        # pandas categorical columns are converted by rpy2 to R factors. This is
        # essential: mice.impute.cart branches on whether y is an R factor.
        with localconverter(default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)

        kwargs: dict[str, Any] = {
            "data": r_data,
            "m": int(self.m),
            "method": self.method,
            "maxit": int(self.maxit),
            "printFlag": bool(self.print_flag),
            "minbucket": int(self.minbucket),
            "cp": float(self.cp),
        }

        if self.seed is not None:
            kwargs["seed"] = int(self.seed)

        if self.visit_sequence is not None:
            if isinstance(self.visit_sequence, str):
                kwargs["visitSequence"] = self.visit_sequence
            else:
                kwargs["visitSequence"] = robjects.StrVector(
                    [str(x) for x in self.visit_sequence]
                )

        if self.predictor_matrix is not None:
            pm = np.asarray(self.predictor_matrix)
            expected = (data.shape[1], data.shape[1])
            if pm.shape != expected:
                raise ValueError(
                    f"predictor_matrix must have shape {expected}, got {pm.shape}"
                )
            r_pm = self._to_r_matrix(robjects, pm, logical=False)
            r_pm.rownames = robjects.StrVector(list(map(str, data.columns)))
            r_pm.colnames = robjects.StrVector(list(map(str, data.columns)))
            kwargs["predictorMatrix"] = r_pm

        if self.where is not None:
            wh = np.asarray(self.where)
            if wh.shape != data.shape:
                raise ValueError(f"where must have shape {data.shape}, got {wh.shape}")
            r_where = self._to_r_matrix(robjects, wh, logical=True)
            r_where.rownames = robjects.StrVector(list(map(str, data.index)))
            r_where.colnames = robjects.StrVector(list(map(str, data.columns)))
            kwargs["where"] = r_where

        if self.ignore is not None:
            ign = list(self.ignore)
            if len(ign) != len(data):
                raise ValueError("ignore must have one element per row")
            kwargs["ignore"] = robjects.BoolVector([bool(x) for x in ign])

        if self.mice_kwargs:
            overlap = set(kwargs).intersection(self.mice_kwargs)
            if overlap:
                raise ValueError(
                    "mice_kwargs duplicates explicitly managed arguments: "
                    + ", ".join(sorted(overlap))
                )
            kwargs.update(self.mice_kwargs)

        self._mids = mice.mice(**kwargs)
        self._columns = list(map(str, data.columns))
        self._index = data.index.copy()
        self._r = (robjects, default_converter, pandas2ri, localconverter)
        self._mice = mice
        return self

    def complete(self, action: int = 1) -> pd.DataFrame:
        if (
            self._mids is None
            or self._r is None
            or self._mice is None
            or self._index is None
            or self._columns is None
        ):
            raise RuntimeError("fit() must be called before complete()")
        if not 1 <= action <= self.m:
            raise ValueError(f"action must be between 1 and {self.m}")

        robjects, default_converter, pandas2ri, localconverter = self._r
        r_complete = robjects.r["complete"](self._mids, action=int(action))
        with localconverter(default_converter + pandas2ri.converter):
            out = robjects.conversion.rpy2py(r_complete)
        out = pd.DataFrame(out)
        out.columns = self._columns
        if len(out) != len(self._index):
            raise RuntimeError(
                f"mice returned {len(out)} rows for an input with {len(self._index)} rows"
            )
        out.index = self._index
        return out

    def transform(self, data: pd.DataFrame, maxit: int = 1) -> list[pd.DataFrame]:
        """Impute new rows using the model specification stored in ``mids``.

        This delegates to ``mice::mice.mids(newdata=...)``.  The fitted method,
        predictor matrix and RNG state are inherited from the original fit.
        """
        if self._mids is None or self._r is None or self._mice is None or self._columns is None:
            raise RuntimeError("fit() must be called before transform()")
        if maxit < 1:
            raise ValueError("maxit must be >= 1")

        data = self._validate_dataframe(data)
        if list(map(str, data.columns)) != self._columns:
            raise ValueError(
                "new data columns and ordering must match the data used by fit()"
            )

        robjects, default_converter, pandas2ri, localconverter = self._r
        with localconverter(default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)

        transformed_mids = robjects.r["mice.mids"](
            self._mids,
            newdata=r_data,
            maxit=int(maxit),
            printFlag=bool(self.print_flag),
        )

        completed = []
        for action in range(1, self.m + 1):
            r_complete = robjects.r["complete"](transformed_mids, action=action)
            with localconverter(default_converter + pandas2ri.converter):
                out = robjects.conversion.rpy2py(r_complete)
            out = pd.DataFrame(out)
            if out.shape != data.shape:
                raise RuntimeError(
                    f"mice.mids returned shape {out.shape}, expected {data.shape}"
                )
            out.columns = data.columns
            out.index = data.index
            completed.append(out)
        return completed
    
    def fit_transform(self, data: pd.DataFrame) -> list[pd.DataFrame]:
        self.fit(data)
        return [self.complete(i) for i in range(1, self.m + 1)]

    # TODO: da rilanciare
    def estimate_parameter_count(self) -> int:
        """Approximate this fit's model complexity, for AIC-style diagnostics only.

        ``mice(..., method='cart')`` fits one ``rpart`` tree per incomplete column at
        every iteration and discards it, so there is no single fitted model to
        introspect. As a proxy, this refits one representative rpart tree per
        imputed column -- using the exact predictor set, minbucket and cp mice
        used -- on that column's observed rows, and sums the leaf (terminal
        node) counts across columns. Leaf count is the standard measure of a
        regression tree's effective number of parameters (each leaf is one
        locally fitted constant). Because it depends on the observed training
        data, it naturally varies fold-to-fold and seed-to-seed; average over
        repeated fits rather than trusting a single value.
        """
        if self._mids is None or self._r is None or self._mice is None or self._columns is None:
            raise RuntimeError("fit() must be called before estimate_parameter_count()")
        robjects, default_converter, pandas2ri, localconverter = self._r
        r = robjects.r
        from rpy2.robjects.packages import importr

        importr("rpart")

        with localconverter(default_converter + pandas2ri.converter):
            original_data = robjects.conversion.rpy2py(self._mids.rx2("data"))
        original_data = pd.DataFrame(original_data)
        original_data.columns = self._columns

        predictor_matrix = np.asarray(self._mids.rx2("predictorMatrix"))
        methods = [str(value) for value in self._mids.rx2("method")]

        total_leaves = 0
        for column_index, method in enumerate(methods):
            if method != "cart":
                continue
            target_column = self._columns[column_index]
            predictors = [
                self._columns[i] for i, flag in enumerate(predictor_matrix[column_index]) if flag
            ]
            observed = original_data.dropna(subset=[target_column])
            if not predictors or len(observed) <= self.minbucket or observed[target_column].nunique() < 2:
                total_leaves += 1  # a constant prediction: one effective parameter
                continue

            subset = _coerce_categorical_columns(observed[[target_column] + predictors])
            with localconverter(default_converter + pandas2ri.converter):
                r_observed = robjects.conversion.py2rpy(subset)
            formula = robjects.Formula(
                f"`{target_column}` ~ " + " + ".join(f"`{predictor}`" for predictor in predictors)
            )
            control = r["rpart.control"](minbucket=int(self.minbucket), cp=float(self.cp))
            tree = r["rpart"](formula, data=r_observed, control=control)
            leaf_flags = [str(value) == "<leaf>" for value in tree.rx2("frame").rx2("var")]
            total_leaves += max(sum(leaf_flags), 1)
        return int(total_leaves)

    @property
    def mids(self):
        """The underlying R `mids` object, for advanced diagnostics in rpy2."""
        if self._mids is None:
            raise RuntimeError("fit() must be called before accessing mids")
        return self._mids


@dataclass
class MissForest:
    """Python front-end to the reference R implementation of missForest.

    Only the random seed and categorical-column declaration are managed by the
    wrapper.  All algorithmic defaults are deliberately left to the installed
    R package.  Additional named arguments can be passed unchanged through
    ``missforest_kwargs`` (for example ``ntree``, ``maxiter`` or ``backend``).
    """

    categorical_columns: Optional[Iterable[str]] = None
    seed: Optional[int] = None
    missforest_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._fit_result = None
        self._columns: list[str] | None = None
        self._index: pd.Index | None = None
        self._r = None
        self._missforest = None
        self.package_version: str | None = None

    @staticmethod
    def _load_r():
        try:
            from rpy2 import robjects
            from rpy2.robjects import default_converter, pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr
        except Exception as exc:
            raise RBackendUnavailable(
                "rpy2 is not available. Install R, then `pip install rpy2`, "
                "and install the R package `missForest`."
            ) from exc

        try:
            missforest = importr("missForest")
        except Exception as exc:
            raise RBackendUnavailable(
                "R is reachable through rpy2, but R package `missForest` is "
                "missing. In R run: install.packages('missForest')."
            ) from exc

        return robjects, default_converter, pandas2ri, localconverter, missforest

    @staticmethod
    def _validate_dataframe(
        data: pd.DataFrame,
        categorical_columns: Optional[Iterable[str]],
    ) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        if data.columns.has_duplicates:
            raise ValueError("column names must be unique")
        if data.index.has_duplicates:
            raise ValueError("row index must uniquely identify species")
        if len(data.columns) == 0:
            raise ValueError("data must contain at least one column")

        converted = _coerce_categorical_columns(data, categorical_columns)
        unsupported = [
            column
            for column in converted.columns
            if not (
                pd.api.types.is_numeric_dtype(converted[column])
                or isinstance(converted[column].dtype, pd.CategoricalDtype)
            )
        ]
        if unsupported:
            raise TypeError(
                "missForest accepts only numeric or categorical columns; "
                f"unsupported columns: {unsupported}"
            )
        return converted

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._validate_dataframe(data, self.categorical_columns)
        (
            robjects,
            default_converter,
            pandas2ri,
            localconverter,
            missforest,
        ) = self._load_r()

        with localconverter(default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)

        if self.seed is not None:
            robjects.r["set.seed"](int(self.seed))

        kwargs: dict[str, Any] = {"xmis": r_data}
        if self.missforest_kwargs:
            overlap = set(kwargs).intersection(self.missforest_kwargs)
            if overlap:
                raise ValueError(
                    "missforest_kwargs duplicates explicitly managed arguments: "
                    + ", ".join(sorted(overlap))
                )
            kwargs.update(self.missforest_kwargs)

        result = missforest.missForest(**kwargs)
        r_imputed = result.rx2("ximp")
        with localconverter(default_converter + pandas2ri.converter):
            out = robjects.conversion.rpy2py(r_imputed)
        out = pd.DataFrame(out)

        if out.shape != data.shape:
            raise RuntimeError(
                f"missForest returned shape {out.shape}, expected {data.shape}"
            )
        out.columns = data.columns
        out.index = data.index

        self._fit_result = result
        self._columns = list(map(str, data.columns))
        self._index = data.index.copy()
        self._r = (robjects, default_converter, pandas2ri, localconverter)
        self._missforest = missforest
        self.package_version = str(robjects.r("as.character(packageVersion('missForest'))")[0])
        return out

    @property
    def fit_result(self):
        """The underlying R ``missForest`` result, including OOB errors."""
        if self._fit_result is None:
            raise RuntimeError("fit_transform() must be called before accessing fit_result")
        return self._fit_result


@dataclass
class Rphylopars:
    """Train-only Rphylopars fit with frozen-parameter test reconstruction.

    ``fit`` estimates the ancestral mean, phylogenetic covariance, optional
    phenotypic covariance, and optional Pagel lambda/kappa using *only* the
    supplied training rows. ``transform`` accepts test rows only, appends the
    stored training rows internally, and reconstructs the test tips while all
    fitted parameters are frozen.

    Rphylopars exposes ``phylocov_fixed``, ``phenocov_fixed`` and
    ``model_par_fixed``, but its public ``phylopars`` entry point still
    re-estimates the root mean even when those parameters are fixed. To prevent
    test observations from changing that mean, ``transform`` uses the fixed-
    parameter call only to construct Rphylopars' three-point inputs, then calls
    the package's internal ``tp`` routine with ``fixed_mu`` set to the value
    learned by ``fit``. This is an internal Rphylopars API, so the installed
    package version is recorded and the required result fields are validated.

    The implementation assumes one row per species, which is the setting used
    by the outer species-level cross-validation pipeline.
    """

    method: str = "BM"
    phylopars_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.method not in ("BM", "lambda", "kappa"):
            raise ValueError("method must be one of 'BM', 'lambda', 'kappa'")
        self._fit_result = None
        self._columns: list[str] | None = None
        self._tip_labels: list[str] | None = None
        self._train_data: pd.DataFrame | None = None
        self._tree = None
        self._r_mu = None
        self._r_phylocov = None
        self._r_phenocov = None
        self._r_model_par = None
        self._last_transform_fit = None
        self._last_transform_result = None
        self._r = None
        self._rphylopars = None
        self.mu_: pd.Series | None = None
        self.phylocov_: pd.DataFrame | None = None
        self.phenocov_: pd.DataFrame | None = None
        self.model_par_: float | None = None
        self.reml_: bool | None = None
        self._phylo_correlated = True
        self.package_version: str | None = None

    @staticmethod
    def _load_r():
        try:
            from rpy2 import robjects
            from rpy2.robjects import default_converter, pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr
        except Exception as exc:
            raise RBackendUnavailable(
                "rpy2 is not available. Install R, then `pip install rpy2`, "
                "and install the R packages `Rphylopars` and `ape`."
            ) from exc

        try:
            rphylopars = importr("Rphylopars")
            ape = importr("ape")
        except Exception as exc:
            raise RBackendUnavailable(
                "R is reachable through rpy2, but the R packages `Rphylopars` "
                "and/or `ape` are missing. In R run: "
                "install.packages(c('Rphylopars', 'ape'))."
            ) from exc

        return robjects, default_converter, pandas2ri, localconverter, rphylopars, ape

    @staticmethod
    def _validate_trait_data(trait_data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(trait_data, pd.DataFrame):
            raise TypeError("trait_data must be a pandas.DataFrame indexed by species name")
        if len(trait_data) == 0:
            raise ValueError("trait_data must contain at least one species")
        if len(trait_data.columns) == 0:
            raise ValueError("trait_data must contain at least one trait column")
        if trait_data.index.has_duplicates:
            raise ValueError("trait_data index must uniquely identify species")
        if trait_data.columns.has_duplicates:
            raise ValueError("trait_data column names must be unique")

        out = trait_data.copy(deep=True)
        out.index = pd.Index(map(str, out.index), name=trait_data.index.name)
        out.columns = list(map(str, out.columns))
        if out.index.has_duplicates:
            raise ValueError("species names must remain unique after conversion to strings")
        if out.columns.has_duplicates:
            raise ValueError("trait names must remain unique after conversion to strings")

        non_numeric = [
            column for column in out.columns
            if not pd.api.types.is_numeric_dtype(out[column])
        ]
        if non_numeric:
            raise TypeError(
                "Rphylopars accepts numeric traits only; non-numeric columns: "
                f"{non_numeric}"
            )
        out = out.astype(float)
        invalid = np.isinf(out.to_numpy(dtype=float, copy=False))
        if invalid.any():
            raise ValueError("trait_data may contain finite values or NaN, but not +/-inf")
        return out

    @staticmethod
    def _to_r_trait_frame(
        trait_data: pd.DataFrame,
        robjects,
        default_converter,
        pandas2ri,
        localconverter,
    ):
        r_trait_data = trait_data.copy(deep=True)
        r_trait_data.insert(0, "species", trait_data.index.astype(str))
        r_trait_data = r_trait_data.reset_index(drop=True)
        with localconverter(default_converter + pandas2ri.converter):
            return robjects.conversion.py2rpy(r_trait_data)

    @staticmethod
    def _list_names(robjects, obj: Any) -> list[str]:
        names = robjects.r["names"](obj)
        return [str(name) for name in names]

    @staticmethod
    def _matrix_to_frame(matrix: Any, columns: list[str], label: str) -> pd.DataFrame:
        values = np.asarray(matrix, dtype=float)
        expected = (len(columns), len(columns))
        if values.shape != expected:
            raise RuntimeError(f"Rphylopars {label} has shape {values.shape}, expected {expected}")
        return pd.DataFrame(values, index=columns, columns=columns)

    @staticmethod
    def _extract_reconstruction(
        anc_recon: Any,
        tip_labels: list[str],
        columns: list[str],
    ) -> pd.DataFrame:
        recon = np.asarray(anc_recon, dtype=float)
        if recon.ndim != 2:
            raise RuntimeError(f"anc_recon must be two-dimensional, got shape {recon.shape}")
        if recon.shape[0] < len(tip_labels) or recon.shape[1] != len(columns):
            raise RuntimeError(
                "Rphylopars returned an incompatible reconstruction: "
                f"shape {recon.shape}, expected at least "
                f"({len(tip_labels)}, {len(columns)})"
            )
        return pd.DataFrame(
            recon[:len(tip_labels), :],
            index=tip_labels,
            columns=columns,
        )

    @staticmethod
    def _filter_tree_tips(
        trait_data: pd.DataFrame,
        tip_labels: list[str],
        data_role: str,
    ) -> pd.DataFrame:
        """Drop input rows that Rphylopars cannot associate with a tree tip."""
        missing = sorted(set(trait_data.index).difference(tip_labels))
        if not missing:
            return trait_data
        warnings.warn(
            f"Rphylopars removed {len(missing)} {data_role} species not present "
            f"in the phylogenetic tree: {missing}",
            UserWarning,
            stacklevel=3,
        )
        return trait_data.drop(index=missing)

    def fit(self, trait_data: pd.DataFrame, tree_path: str | Path) -> "Rphylopars":
        """Fit the phylogenetic model.

        Parameters
        ----------
        trait_data : pandas.DataFrame indexed by species name (matching the
            tree's tip labels), with one column per numeric trait. Missing
            values must be encoded as ``NaN``.
        tree_path : path to a Newick (``.nwk``) tree file.
        """
        trait_data = self._validate_trait_data(trait_data)
        (
            robjects,
            default_converter,
            pandas2ri,
            localconverter,
            rphylopars,
            ape,
        ) = self._load_r()

        tree = ape.read_tree(str(tree_path))
        tip_labels = [str(label) for label in tree.rx2("tip.label")]
        trait_data = self._filter_tree_tips(trait_data, tip_labels, "training")
        if trait_data.empty:
            raise ValueError(
                "Rphylopars has no training species present in the phylogenetic tree"
            )
        all_missing = trait_data.columns[trait_data.isna().all(axis=0)].tolist()
        if all_missing:
            raise ValueError(
                "Every fitted trait needs at least one training observation after "
                "removing species absent from the tree; all-missing training traits: "
                f"{all_missing}"
            )

        r_df = self._to_r_trait_frame(
            trait_data,
            robjects,
            default_converter,
            pandas2ri,
            localconverter,
        )

        kwargs: dict[str, Any] = {"trait_data": r_df, "tree": tree, "model": self.method}
        if self.phylopars_kwargs:
            managed = {
                "trait_data",
                "tree",
                "model",
                "ret_args",
                "ret_level",
                "phylocov_fixed",
                "phenocov_fixed",
                "model_par_fixed",
                "phenocov_list",
            }
            overlap = managed.intersection(self.phylopars_kwargs)
            if overlap:
                raise ValueError(
                    "phylopars_kwargs duplicates wrapper-managed arguments: "
                    + ", ".join(sorted(overlap))
                )
            kwargs.update(self.phylopars_kwargs)

        self._fit_result = rphylopars.phylopars(**kwargs)
        columns = list(trait_data.columns)
        fit_names = self._list_names(robjects, self._fit_result)
        required = {"mu", "pars", "model", "anc_recon", "tree", "REML"}
        missing_fields = sorted(required.difference(fit_names))
        if missing_fields:
            raise RuntimeError(
                "The installed Rphylopars result lacks required fields: "
                f"{missing_fields}"
            )

        r_mu = self._fit_result.rx2("mu")
        mu = np.asarray(r_mu, dtype=float).reshape(-1)
        if mu.shape != (len(columns),):
            raise RuntimeError(
                f"Rphylopars mu has shape {mu.shape}, expected {(len(columns),)}"
            )
        if not np.isfinite(mu).all():
            raise RuntimeError("Rphylopars returned a non-finite ancestral mean")

        r_pars = self._fit_result.rx2("pars")
        pars_names = self._list_names(robjects, r_pars)
        if "phylocov" not in pars_names:
            raise RuntimeError("Rphylopars result does not contain pars$phylocov")
        r_phylocov = r_pars.rx2("phylocov")
        phylocov = self._matrix_to_frame(r_phylocov, columns, "phylocov")
        if not np.isfinite(phylocov.to_numpy()).all():
            raise RuntimeError("Rphylopars returned a non-finite phylocov")

        r_phenocov = None
        phenocov = None
        if "phenocov" in pars_names:
            r_phenocov = r_pars.rx2("phenocov")
            phenocov = self._matrix_to_frame(r_phenocov, columns, "phenocov")
            if not np.isfinite(phenocov.to_numpy()).all():
                raise RuntimeError("Rphylopars returned a non-finite phenocov")

        r_model = self._fit_result.rx2("model")
        model_names = self._list_names(robjects, r_model)
        r_model_par = None
        model_par = None
        if self.method != "BM":
            if self.method not in model_names:
                raise RuntimeError(
                    f"Rphylopars result does not contain model${self.method}"
                )
            r_model_par = r_model.rx2(self.method)
            model_values = np.asarray(r_model_par, dtype=float).reshape(-1)
            if model_values.shape != (1,):
                raise RuntimeError(
                    f"Rphylopars {self.method} has shape {model_values.shape}, expected (1,)"
                )
            model_par = float(model_values[0])
            if not np.isfinite(model_par):
                raise RuntimeError(f"Rphylopars returned a non-finite {self.method}")

        reml_values = np.asarray(self._fit_result.rx2("REML")).reshape(-1)
        if reml_values.shape != (1,):
            raise RuntimeError(
                f"Rphylopars REML has shape {reml_values.shape}, expected (1,)"
            )
        reml = bool(reml_values[0])

        self._columns = columns
        self._tip_labels = tip_labels
        self._train_data = trait_data.copy(deep=True)
        self._tree = tree
        self._r_mu = r_mu
        self._r_phylocov = r_phylocov
        self._r_phenocov = r_phenocov
        self._r_model_par = r_model_par
        self._r = (robjects, default_converter, pandas2ri, localconverter)
        self._rphylopars = rphylopars
        self.mu_ = pd.Series(mu, index=columns, name="mu")
        self.phylocov_ = phylocov
        self.phenocov_ = phenocov
        self.model_par_ = model_par
        self.reml_ = reml
        self._phylo_correlated = bool(
            (self.phylopars_kwargs or {}).get("phylo_correlated", True)
        )
        self.package_version = str(
            robjects.r("as.character(packageVersion('Rphylopars'))")[0]
        )
        return self

    def _check_parameters_stayed_fixed(self, frozen_fit: Any) -> None:
        """Fail loudly if Rphylopars did not retain a supplied fixed parameter."""
        if self.phylocov_ is None or self._r is None:
            raise RuntimeError("fit() must be called before checking parameters")
        if self.method != "BM" and self.model_par_ is None:
            raise RuntimeError(
                f"Rphylopars {self.method} fit has no model parameter to validate"
            )
        robjects = self._r[0]
        r_pars = frozen_fit.rx2("pars")
        pars_names = self._list_names(robjects, r_pars)
        if "phylocov" not in pars_names:
            raise RuntimeError("Frozen Rphylopars call omitted pars$phylocov")
        observed_phylocov = np.asarray(r_pars.rx2("phylocov"), dtype=float)
        if not np.allclose(
            observed_phylocov,
            self.phylocov_.to_numpy(),
            rtol=1e-8,
            atol=1e-10,
            equal_nan=True,
        ):
            raise RuntimeError("Rphylopars changed phylocov during transform")

        if self.phenocov_ is not None:
            if "phenocov" not in pars_names:
                raise RuntimeError("Frozen Rphylopars call omitted pars$phenocov")
            observed_phenocov = np.asarray(r_pars.rx2("phenocov"), dtype=float)
            if not np.allclose(
                observed_phenocov,
                self.phenocov_.to_numpy(),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            ):
                raise RuntimeError("Rphylopars changed phenocov during transform")

        if self.method != "BM":
            r_model = frozen_fit.rx2("model")
            model_names = self._list_names(robjects, r_model)
            if self.method not in model_names:
                raise RuntimeError(
                    f"Frozen Rphylopars call omitted model${self.method}"
                )
            observed_model_par = float(
                np.asarray(r_model.rx2(self.method), dtype=float).reshape(-1)[0]
            )
            if not np.isclose(
                observed_model_par,
                float(self.model_par_),
                rtol=1e-8,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Rphylopars changed {self.method} during transform"
                )

    def transform(self, test_trait_data: pd.DataFrame) -> pd.DataFrame:
        """Impute test rows with training parameters and ancestral mean frozen.

        ``test_trait_data`` must contain test species only. For leave-one-trait-
        out evaluation, construct a fresh copy for each target trait and mask
        all representations of that target before calling this method. The
        method appends the unchanged training table internally and returns only
        the test-species rows in their original order.
        """
        if (
            self._fit_result is None
            or self._r is None
            or self._rphylopars is None
            or self._tip_labels is None
            or self._columns is None
            or self._train_data is None
            or self._tree is None
            or self.mu_ is None
            or self._r_phylocov is None
            or self.reml_ is None
        ):
            raise RuntimeError("fit() must be called before transform()")

        test_trait_data = self._validate_trait_data(test_trait_data)
        if list(test_trait_data.columns) != self._columns:
            raise ValueError(
                "test trait columns and ordering must exactly match the training data"
            )
        overlap = sorted(set(test_trait_data.index).intersection(self._train_data.index))
        if overlap:
            raise ValueError(
                "transform() accepts held-out species only; species also present in "
                f"training: {overlap[:10]}"
            )
        test_trait_data = self._filter_tree_tips(
            test_trait_data, self._tip_labels, "testing"
        )
        if test_trait_data.empty:
            return pd.DataFrame(index=test_trait_data.index, columns=self._columns, dtype=float)

        robjects, default_converter, pandas2ri, localconverter = self._r
        combined = pd.concat([self._train_data, test_trait_data], axis=0)
        r_df = self._to_r_trait_frame(
            combined,
            robjects,
            default_converter,
            pandas2ri,
            localconverter,
        )

        # This call builds the missingness- and species-specific three-point
        # inputs. All estimable covariance/evolution parameters are fixed, and
        # both EM and BFGS are disabled. Its own reconstructed mean is ignored.
        kwargs: dict[str, Any] = {
            "trait_data": r_df,
            "tree": self._tree,
            "model": self.method,
            "phylocov_fixed": self._r_phylocov,
            "phylo_correlated": self._phylo_correlated,
            "REML": self.reml_,
            "skip_optim": True,
            "skip_EM": True,
            "usezscores": False,
        }
        if self._r_phenocov is not None:
            kwargs["phenocov_fixed"] = self._r_phenocov
        if self._r_model_par is not None:
            kwargs["model_par_fixed"] = self._r_model_par

        frozen_fit = self._rphylopars.phylopars(**kwargs)
        self._check_parameters_stayed_fixed(frozen_fit)
        frozen_names = self._list_names(robjects, frozen_fit)
        if "threepoint_calc" not in frozen_names:
            raise RuntimeError(
                "The installed Rphylopars version did not return threepoint_calc; "
                "a frozen-mu transform cannot be guaranteed"
            )

        threepoint_args = frozen_fit.rx2("threepoint_calc")
        argument_names = self._list_names(robjects, threepoint_args)
        required_args = {"fixed_mu", "ret_level"}
        missing_args = sorted(required_args.difference(argument_names))
        if missing_args:
            raise RuntimeError(
                "Rphylopars threepoint_calc lacks required arguments: "
                f"{missing_args}"
            )
        tp_kwargs = {
            name: threepoint_args.rx2(name)
            for name in argument_names
        }
        r_mu_vector = robjects.FloatVector(
            [float(value) for value in self.mu_.to_numpy()]
        )
        tp_kwargs["fixed_mu"] = robjects.r["matrix"](
            r_mu_vector,
            nrow=len(self._columns),
            ncol=1,
        )
        tp_kwargs["ret_level"] = 3

        tp = robjects.r["getFromNamespace"]("tp", "Rphylopars")
        frozen_result = tp(**tp_kwargs)
        result_names = self._list_names(robjects, frozen_result)
        if "anc_recon" not in result_names:
            raise RuntimeError("Rphylopars::tp did not return anc_recon")

        reconstructed = self._extract_reconstruction(
            frozen_result.rx2("anc_recon"),
            self._tip_labels,
            self._columns,
        )
        self._last_transform_fit = frozen_fit
        self._last_transform_result = frozen_result
        return reconstructed.loc[test_trait_data.index, self._columns].copy()

    def fit_transform(self, trait_data: pd.DataFrame, tree_path: str | Path) -> pd.DataFrame:
        """Legacy one-shot reconstruction of the same rows used for fitting.

        Use ``fit(train, tree).transform(masked_test)`` for leakage-controlled
        outer-fold evaluation.
        """
        self.fit(trait_data, tree_path)
        if (
            self._fit_result is None
            or self._tip_labels is None
            or self._columns is None
            or self._train_data is None
        ):
            raise RuntimeError("Rphylopars fit did not produce a reconstruction")
        reconstructed = self._extract_reconstruction(
            self._fit_result.rx2("anc_recon"),
            self._tip_labels,
            self._columns,
        )
        # ``fit`` may have removed input species absent from the tree. Return
        # precisely the retained input rows, matching fit_transform's contract.
        return reconstructed.loc[self._train_data.index, self._columns].copy()

    @property
    def fit_result(self):
        """The underlying R `phylopars` fit object, for advanced diagnostics in rpy2."""
        if self._fit_result is None:
            raise RuntimeError("fit() must be called before accessing fit_result")
        return self._fit_result
