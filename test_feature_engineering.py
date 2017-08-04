import unittest
import pandas as pd
import feature_engineering
import numpy as np
import numpy.testing as npt

class TestEngine(unittest.TestCase):

    def test_dummify_columns(self):
        df = pd.DataFrame({'a': ['1', '2', '3', '2', '1'], 'b': [1, 2, 3, 4, 5]})
        engine = feature_engineering.Engine()
        df = engine.dummify_columns(df)
        npt.assert_equal(df['b'], np.array([1, 2, 3, 4, 5]))
        npt.assert_equal(df['1_dummy'], np.array([1, 0, 0, 0, 1]))
        npt.assert_equal(df['2_dummy'], np.array([0, 1, 0, 1, 0]))

    def test_evaluate(self):
        X = pd.DataFrame({'a': [1, 0, 1, 0, 1, 0, 1, 0, 1], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        y = pd.DataFrame({'label': [1, 1, 1, 0, 0, 1, 0, 0, 0]})
        engine = feature_engineering.Engine()
        engine.evaluate(X, y, max_time=0.001)

    def test_simplify(self):
        X = pd.DataFrame({'a': [1, 0, 1, 0, 1, 0, 1, 0, 1], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        y = pd.DataFrame({'label': [1, 1, 1, 0, 0, 1, 0, 0, 0]})
        engine = feature_engineering.Engine()
        engine.simplify(X, y)

    def test_squares(self):
        X = pd.DataFrame({'a': [1, 0, 1, 0, 1], 'b': [1, 2, 3, 4, 5]})
        engine = feature_engineering.Engine()
        X = engine.squares(X)
        npt.assert_equal(X.columns, np.array(['a', 'b', 'b^2']))
        npt.assert_equal(X['b^2'], np.array([1, 4, 9, 16, 25]))

    def test_logs(self):
        X = pd.DataFrame({'a': [1, 0, 1, 0, 1], 'b': [1, 2, 3, 4, 5]})
        engine = feature_engineering.Engine()
        X = engine.logs(X)
        npt.assert_equal(X.columns, np.array(['a', 'b', 'log(b)']))
        npt.assert_allclose(X['log(b)'], np.array([0., 0.69314718, 1.09861229, 1.38629436, 1.60943791]))

    def test_interactions(self):
        X = pd.DataFrame({'a': [1, 0, 1, 0, 1], 'b': [1, 2, 3, 4, 5]})
        engine = feature_engineering.Engine()
        X = engine.interactions(X)
        npt.assert_equal(X.columns, np.array(['a', 'b', 'a.b']))
        npt.assert_equal(X['a.b'], np.array([1, 0, 3, 0, 5]))

    def test_fit(self):
        X = pd.DataFrame({'a': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                          'b': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
        y = pd.DataFrame({'l': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]})
        engine = feature_engineering.Engine()
        X = engine.fit(X, y)
        self.assertTrue(engine.score(X, y) > 0.95)
