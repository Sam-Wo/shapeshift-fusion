"""The Gram-matrix fold transforms and multi-alpha ridge reproduce sklearn PCA / TruncatedSVD / Pipeline."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fusion import engine as E
from fusion import synthetic


def _abs_close(a, b):
    np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=1e-6, atol=1e-6)


def test_gram_pca_matches_sklearn_pca():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 40)) @ np.diag(np.linspace(3, 0.1, 40))
    S, Ev = np.arange(60), np.arange(60, 80)
    Zs, Ze = E.gram_pca(X @ X.T, S, Ev, 5, centered=True)
    p = PCA(5, svd_solver="full").fit(X[S])
    _abs_close(Zs, p.transform(X[S]))
    _abs_close(Ze, p.transform(X[Ev]))


def test_gram_svd_matches_truncated_svd():
    rng = np.random.default_rng(1)
    X = (rng.random((70, 50)) < 0.2).astype(float)
    S, Ev = np.arange(50), np.arange(50, 70)
    Zs, Ze = E.gram_pca(X @ X.T, S, Ev, 6, centered=False)
    t = TruncatedSVD(6, algorithm="arpack").fit(X[S])
    _abs_close(Zs, t.transform(X[S]))
    _abs_close(Ze, t.transform(X[Ev]))


def test_ridge_multi_matches_sklearn():
    rng = np.random.default_rng(2)
    Xs, Xe = rng.standard_normal((50, 8)), rng.standard_normal((10, 8))
    y = rng.standard_normal(50)
    alphas = np.array([0.5, 3.0, 40.0])
    P = E.ridge_multi(Xs, y, Xe, alphas)
    for a, pr in zip(alphas, P):
        np.testing.assert_allclose(pr, Ridge(alpha=a).fit(Xs, y).predict(Xe), rtol=1e-8, atol=1e-10)
    Y = np.column_stack([y, -y, y ** 2])
    PM = E.ridge_multi(Xs, Y, Xe, alphas)
    for b in range(3):
        np.testing.assert_allclose(PM[:, :, b], E.ridge_multi(Xs, Y[:, b], Xe, alphas), atol=1e-10)


def test_fusion_design_matches_sklearn_pipeline():
    """Arm F for one split == sklearn Pipeline(ColumnTransformer[OneHot, TruncatedSVD, impute, PCA, PCA],
    StandardScaler, Ridge) fitted on the training rows only."""
    prob, _ = synthetic.make(n=90, seed=3)
    S, Ev = np.arange(70), np.arange(70, 90)
    arm = E.Arm("F", moa=True, chem=True, nuis=True, blocks=("morph", "tx"), k=5)
    feat = E.Featurizer(prob)
    Xs, Xe = E.design(feat, arm, S, Ev, "k", 16, 5, None)
    mine = E.ridge_multi(Xs, prob.y[S], Xe, [10.0])[0]

    pm, pt = prob.blocks["morph"].shape[1], prob.blocks["tx"].shape[1]
    Xall = np.hstack([prob.moa_code[:, None], prob.bits, prob.nuis, prob.blocks["morph"], prob.blocks["tx"]])
    c = np.cumsum([1, prob.bits.shape[1], prob.nuis.shape[1], pm, pt])
    ct = ColumnTransformer([
        ("moa", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0]),
        ("chem", TruncatedSVD(16, algorithm="arpack"), list(range(c[0], c[1]))),
        ("nuis", SimpleImputer(strategy="median"), list(range(c[1], c[2]))),
        ("morph", PCA(5, svd_solver="full"), list(range(c[2], c[3]))),
        ("tx", PCA(5, svd_solver="full"), list(range(c[3], c[4]))),
    ])
    pipe = Pipeline([("ct", ct), ("sc", StandardScaler()), ("ridge", Ridge(alpha=10.0))]).fit(Xall[S], prob.y[S])
    np.testing.assert_allclose(mine, pipe.predict(Xall[Ev]), rtol=1e-6, atol=1e-8)
