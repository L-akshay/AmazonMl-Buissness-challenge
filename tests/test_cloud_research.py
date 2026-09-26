import unittest
import numpy as np
from src.cloud_research import masks,ABLATIONS
from src.features import FEATURE_NAMES


class ResearchTests(unittest.TestCase):
    def test_country_threshold_and_fit_scope_exclude_foreign_labels_and_reserved(self):
        refs={'fold':np.tile(np.arange(5),2),'country_norm':np.array(['us']*5+['india']*5)}
        for country in ('us','india'):
            fit,test,columns=masks(refs,'country-'+country)
            self.assertFalse(np.any(fit & test))
            self.assertFalse(np.any((fit|test) & (refs['fold']==4)))
            self.assertTrue(np.all(refs['country_norm'][fit]==country))
            self.assertTrue(np.all(refs['country_norm'][test]!=country))
            self.assertEqual(len(columns),51)

    def test_ablations_keep_all_entities_and_remove_declared_features(self):
        refs={'fold':np.arange(5),'country_norm':np.array(['us']*5)}
        for name,removed in ABLATIONS.items():
            fit,test,columns=masks(refs,'ablation-'+name)
            np.testing.assert_array_equal(fit,refs['fold']!=4)
            np.testing.assert_array_equal(fit,test)
            self.assertEqual(set(FEATURE_NAMES)-{FEATURE_NAMES[i] for i in columns},removed)
